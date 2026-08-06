/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Adaptive Voronoi collective variable.

   Implements:
     - detector weights u_ih at LAMBDA_DET
     - participation coordinate Theta_h = 1 - sum_i u_ih^2
     - raised-cosine gate s(Theta_h)
     - geometrically interpolated per-hydrogen lambda_eff
     - adaptive weights w_ih
     - full analytic atomic and box derivatives, including the
       position-dependent-lambda gate correction

   Lambda convention in this action:

       weight = exp(-lambda * r)

   Thus LAMBDA_DET, LAMBDA_HIGH, and LAMBDA_LOW are positive inverse-length
   parameters, matching the accompanying Fortran implementation.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "Colvar.h"

#include "tools/Matrix.h"
#include "core/ActionRegister.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace PLMD {
namespace colvar {

class VoronoiD1Adaptive : public Colvar {
private:
  bool pbc;
  bool serial;
  bool invalidateList;
  bool firsttime;

  std::unique_ptr<NeighborList> nl;

  std::vector<PLMD::AtomNumber> list_a;
  std::vector<PLMD::AtomNumber> list_b;

  double lambdaDet;
  double lambdaHigh;
  double lambdaLow;
  double thetaOn;
  double thetaOff;

  double d0;
  double d1;
  double d2;
  double d3;

  int nrx;
  int numAtomsa;
  int numAtomsb;
  int numAtomso;

  static void smoothGate(const double theta,
                         const double thetaOn,
                         const double thetaOff,
                         double& s,
                         double& sprime);

public:
  explicit VoronoiD1Adaptive(const ActionOptions&);
  ~VoronoiD1Adaptive();

  void calculate() override;
  void prepare() override;

  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(VoronoiD1Adaptive, "VORONOID1_ADAPTIVE")


void VoronoiD1Adaptive::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);

  keys.addFlag(
      "SERIAL", false,
      "Perform neighbor-list construction in serial; mainly useful for "
      "debugging.");

  keys.addFlag(
      "NLIST", false,
      "Use a neighbor list for GROUPA-GROUPB pairs.");

  keys.add(
      "optional", "NL_CUTOFF",
      "Cutoff for the neighbor list. The cutoff must be sufficiently large "
      "to include every GROUPA site that contributes appreciably to the "
      "Voronoi normalization.");

  keys.add(
      "optional", "NL_STRIDE",
      "Frequency at which the neighbor list is updated.");

  keys.add(
      "atoms", "GROUPA",
      "Voronoi sites. The first size(GROUPA)-NRX atoms are the water oxygen "
      "sites entering the final pair-distance CV.");

  keys.add(
      "atoms", "GROUPB",
      "Atoms assigned among the Voronoi sites, normally the hydrogen atoms.");

  keys.add(
      "compulsory", "LAMBDA_DET", "1.0",
      "Positive inverse-length parameter used for the detector weights "
      "u_ih = exp(-LAMBDA_DET*r_ih)/sum_j exp(-LAMBDA_DET*r_jh).");

  keys.add(
      "compulsory", "LAMBDA_HIGH", "1.0",
      "Positive high-lambda endpoint of the adaptive interpolation.");

  keys.add(
      "compulsory", "LAMBDA_LOW", "1.0",
      "Positive low-lambda endpoint of the adaptive interpolation.");

  keys.add(
      "compulsory", "THETA_ON", "0.0",
      "Value of Theta_h below which the adaptive gate is zero.");

  keys.add(
      "compulsory", "THETA_OFF", "1.0",
      "Value of Theta_h above which the adaptive gate is one.");

  keys.add(
      "compulsory", "D_0", "0.0",
      "Charge shift applied to ordinary GROUPA sites.");

  keys.add(
      "compulsory", "D_1", "0.0",
      "Charge shift applied to the first reactive GROUPA site.");

  keys.add(
      "compulsory", "D_2", "0.0",
      "Charge shift applied to the second reactive GROUPA site.");

  keys.add(
      "compulsory", "D_3", "0.0",
      "Charge shift applied to the third reactive GROUPA site.");

  keys.add(
      "compulsory", "NRX", "0",
      "Number of reactive/non-water sites at the end of GROUPA. Only the "
      "first size(GROUPA)-NRX sites enter the pair-distance sum.");
   
  keys.setValueDescription("scalar",
      "the adaptive Voronoi collective variable (charge-weighted pair-distance sum)");
}


VoronoiD1Adaptive::VoronoiD1Adaptive(const ActionOptions& ao)
    : PLUMED_COLVAR_INIT(ao),
      pbc(true),
      serial(false),
      invalidateList(true),
      firsttime(true),
      lambdaDet(1.0),
      lambdaHigh(1.0),
      lambdaLow(1.0),
      thetaOn(0.0),
      thetaOff(1.0),
      d0(0.0),
      d1(0.0),
      d2(0.0),
      d3(0.0),
      nrx(0),
      numAtomsa(0),
      numAtomsb(0),
      numAtomso(0) {

  parseFlag("SERIAL", serial);

  std::vector<AtomNumber> gaLista;
  std::vector<AtomNumber> gbLista;

  parseAtomList("GROUPA", gaLista);
  parseAtomList("GROUPB", gbLista);

  list_a = gaLista;
  list_b = gbLista;

  numAtomsa = static_cast<int>(list_a.size());
  numAtomsb = static_cast<int>(list_b.size());

  if (numAtomsa <= 0) {
    error("GROUPA must contain at least one atom");
  }

  if (numAtomsb <= 0) {
    error("GROUPB must contain at least one atom");
  }

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  parse("LAMBDA_DET", lambdaDet);
  parse("LAMBDA_HIGH", lambdaHigh);
  parse("LAMBDA_LOW", lambdaLow);
  parse("THETA_ON", thetaOn);
  parse("THETA_OFF", thetaOff);

  parse("D_0", d0);
  parse("D_1", d1);
  parse("D_2", d2);
  parse("D_3", d3);
  parse("NRX", nrx);

  if (lambdaDet <= 0.0) {
    error("LAMBDA_DET must be positive");
  }

  if (lambdaHigh <= 0.0) {
    error("LAMBDA_HIGH must be positive");
  }

  if (lambdaLow <= 0.0) {
    error("LAMBDA_LOW must be positive");
  }

  if (nrx < 0 || nrx > numAtomsa) {
    error("NRX must satisfy 0 <= NRX <= size(GROUPA)");
  }

  numAtomso = numAtomsa - nrx;

  if (numAtomso <= 0) {
    error("GROUPA must contain at least one non-reactive/water site");
  }

  bool doneigh = false;
  double nlCut = 0.0;
  int nlStride = 0;

  parseFlag("NLIST", doneigh);

  if (doneigh) {
    parse("NL_CUTOFF", nlCut);
    parse("NL_STRIDE", nlStride);

    if (nlCut <= 0.0) {
      error("NL_CUTOFF must be explicitly specified and positive");
    }

    if (nlStride <= 0) {
      error("NL_STRIDE must be explicitly specified and positive");
    }
  }

  addValueWithDerivatives();
  setNotPeriodic();

  /*
   * PAIR is intentionally not supported. The adaptive detector and
   * Voronoi normalization require each GROUPB atom to be compared with
   * all relevant GROUPA sites.
   */
  if (doneigh) {
    nl = Tools::make_unique<NeighborList>(
        gaLista, gbLista, serial, false, pbc, getPbc(), comm,
        nlCut, nlStride);
  } else {
    nl = Tools::make_unique<NeighborList>(
        gaLista, gbLista, serial, false, pbc, getPbc(), comm);
  }

  requestAtoms(nl->getFullAtomList());

  log.printf("  adaptive Voronoi CV\n");
  log.printf("  GROUPA contains %u atoms\n",
             static_cast<unsigned>(gaLista.size()));
  log.printf("  GROUPB contains %u atoms\n",
             static_cast<unsigned>(gbLista.size()));
  log.printf("  number of water/non-reactive GROUPA sites: %d\n",
             numAtomso);
  log.printf("  number of reactive GROUPA sites: %d\n", nrx);

  log.printf("  LAMBDA_DET  = %f\n", lambdaDet);
  log.printf("  LAMBDA_HIGH = %f\n", lambdaHigh);
  log.printf("  LAMBDA_LOW  = %f\n", lambdaLow);
  log.printf("  THETA_ON    = %f\n", thetaOn);
  log.printf("  THETA_OFF   = %f\n", thetaOff);

  log.printf("  lambda convention: exp(-lambda*r)\n");

  if (pbc) {
    log.printf("  using periodic boundary conditions\n");
  } else {
    log.printf("  without periodic boundary conditions\n");
  }

  if (doneigh) {
    log.printf("  using a neighbor list with cutoff %f\n", nlCut);
    log.printf("  neighbor-list update stride: %d\n", nlStride);
    log.printf("  WARNING: the cutoff must include all GROUPA sites with "
               "non-negligible Voronoi weight\n");
  }

  checkRead();
}


VoronoiD1Adaptive::~VoronoiD1Adaptive() = default;


void VoronoiD1Adaptive::prepare() {
  if (nl->getStride() > 0) {
    if (firsttime || (getStep() % nl->getStride() == 0)) {
      requestAtoms(nl->getFullAtomList());
      invalidateList = true;
      firsttime = false;
    } else {
      /*
       * Keep the full list so that the local GROUPA indices remain
       * consistent with the order used for charge and A arrays.
       */
      requestAtoms(nl->getFullAtomList());
      invalidateList = false;

      if (getExchangeStep()) {
        error("Neighbor lists must be updated on exchange steps; choose an "
              "NL_STRIDE that divides the exchange stride");
      }
    }

    if (getExchangeStep()) {
      firsttime = true;
    }
  }
}


void VoronoiD1Adaptive::smoothGate(const double theta,
                                   const double thetaOn,
                                   const double thetaOff,
                                   double& s,
                                   double& sprime) {
  const double pi = 3.141592653589793238462643383279502884;

  if (thetaOff <= thetaOn) {
    /*
     * Degenerate gate: hard step. Its derivative is set to zero.
     */
    s = (theta >= thetaOn) ? 1.0 : 0.0;
    sprime = 0.0;
    return;
  }

  if (theta <= thetaOn) {
    s = 0.0;
    sprime = 0.0;
  } else if (theta >= thetaOff) {
    s = 1.0;
    sprime = 0.0;
  } else {
    const double x = (theta - thetaOn) / (thetaOff - thetaOn);

    s = 0.5 * (1.0 - std::cos(pi * x));

    sprime =
        pi /
        (2.0 * (thetaOff - thetaOn)) *
        std::sin(pi * x);
  }
}


void VoronoiD1Adaptive::calculate() {
  const double tinyR = 1.0e-14;
  const double infinity = std::numeric_limits<double>::infinity();

  if (nl->getStride() > 0 && invalidateList) {
    nl->update(getPositions());
  }

  const unsigned nAtoms = getNumberOfAtoms();
  const unsigned nn = nl->size();

  /*
   * The current implementation follows the same local-index convention as
   * VORONOID1:
   *
   *   local indices 0 ... numAtomsa-1 correspond to GROUPA.
   *
   * NeighborList pair indices are local requested-atom indices.
   */

  std::vector<unsigned> pairO(nn);
  std::vector<unsigned> pairH(nn);

  std::vector<double> pairR(nn, 0.0);
  std::vector<Vector> pairRvec(nn);
  std::vector<Vector> pairEOH(nn);

  /*
   * Detector/adaptive pair weights.
   */
  std::vector<double> u(nn, 0.0);
  std::vector<double> w(nn, 0.0);

  /*
   * Per-hydrogen quantities are indexed using the local requested-atom
   * index of the GROUPB atom.
   */
  std::vector<double> minR(nAtoms, infinity);
  std::vector<double> detectorNorm(nAtoms, 0.0);
  std::vector<double> adaptiveNorm(nAtoms, 0.0);

  std::vector<double> purity(nAtoms, 0.0);
  std::vector<double> theta(nAtoms, 0.0);
  std::vector<double> gate(nAtoms, 0.0);
  std::vector<double> gatePrime(nAtoms, 0.0);
  std::vector<double> lambdaEff(nAtoms, lambdaHigh);
  std::vector<double> kappa(nAtoms, 0.0);

  std::vector<double> rbar(nAtoms, 0.0);
  std::vector<double> abar(nAtoms, 0.0);
  std::vector<double> cad(nAtoms, 0.0);

  /*
   * Per-GROUPA quantities.
   */
  std::vector<double> charge(numAtomsa, 0.0);
  std::vector<double> A(numAtomsa, 0.0);

  /*
   * Atomic and box derivatives.
   */
  std::vector<Vector> deriv(nAtoms);
  Vector zero;
  zero.zero();
  std::fill(deriv.begin(), deriv.end(), zero);

  Tensor virial;

  /*
   * Geometric interpolation:
   *
   *   lambda_eff = lambdaHigh * exp[-s*log(lambdaHigh/lambdaLow)]
   *
   * and
   *
   *   kappa = -d(lambda_eff)/dTheta
   *         = log(lambdaHigh/lambdaLow)
   *           * lambda_eff * ds/dTheta.
   */
  const double logLambdaRatio = std::log(lambdaHigh / lambdaLow);

  //====================================================================
  // PASS 0: GROUPA-GROUPB distances and unit vectors
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned ind0 = nl->getClosePair(i).first;
    const unsigned ind1 = nl->getClosePair(i).second;

    if (ind0 >= static_cast<unsigned>(numAtomsa)) {
      error("Internal GROUPA index is inconsistent with the requested-atom "
            "ordering");
    }

    pairO[i] = ind0;
    pairH[i] = ind1;

    Vector rij;
    if (pbc) {
      rij = pbcDistance(getPosition(ind0), getPosition(ind1));
    } else {
      rij = delta(getPosition(ind0), getPosition(ind1));
    }

    /*
     * PLUMED pair vector:
     *
     *   rij = R_H - R_O
     *
     * The Fortran e_oh convention is:
     *
     *   e_oh = (R_O - R_H)/r = -rij/r.
     */
    const double r = rij.modulo();

    pairRvec[i] = rij;
    pairR[i] = r;

    if (r > tinyR) {
      pairEOH[i] = (-1.0 / r) * rij;
    } else {
      pairEOH[i] = zero;
    }

    minR[ind1] = std::min(minR[ind1], r);
  }

  //====================================================================
  // PASS 1: detector weights u_ih
  //
  // Stable evaluation:
  //
  //   exp[-lambda_det*(r_i-r_min)]
  //
  // The common exp(-lambda_det*r_min) cancels in the normalization.
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned h = pairH[i];

    const double shiftedR = pairR[i] - minR[h];
    const double value = std::exp(-lambdaDet * shiftedR);

    u[i] = value;
    detectorNorm[h] += value;
  }

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned h = pairH[i];

    if (detectorNorm[h] <= 0.0) {
      error("Zero detector normalization encountered");
    }

    u[i] /= detectorNorm[h];
    purity[h] += u[i] * u[i];
  }

  //====================================================================
  // PASS 2: Theta_h, gate, lambda_eff and kappa
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned h = pairH[i];

    /*
     * Process each GROUPB atom only once. detectorNorm[h] is set to a
     * negative marker after processing.
     */
    if (detectorNorm[h] < 0.0) {
      continue;
    }

    theta[h] = 1.0 - purity[h];

    smoothGate(theta[h], thetaOn, thetaOff,
               gate[h], gatePrime[h]);

    lambdaEff[h] =
        lambdaHigh * std::exp(-logLambdaRatio * gate[h]);

    kappa[h] =
        logLambdaRatio * lambdaEff[h] * gatePrime[h];

    detectorNorm[h] = -detectorNorm[h];
  }

  // Restore the signs in case detectorNorm is inspected below.
  for (unsigned i = 0; i < nAtoms; ++i) {
    detectorNorm[i] = std::fabs(detectorNorm[i]);
  }

  //====================================================================
  // PASS 3: adaptive weights w_ih
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned h = pairH[i];

    const double shiftedR = pairR[i] - minR[h];
    const double value = std::exp(-lambdaEff[h] * shiftedR);

    w[i] = value;
    adaptiveNorm[h] += value;
  }

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned o = pairO[i];
    const unsigned h = pairH[i];

    if (adaptiveNorm[h] <= 0.0) {
      error("Zero adaptive Voronoi normalization encountered");
    }

    w[i] /= adaptiveNorm[h];

    charge[o] += w[i];
    rbar[h] += w[i] * pairR[i];
  }

  //====================================================================
  // Shift the GROUPA charges.
  //
  // This retains the conventions of the original VORONOID1 action:
  //
  //   ordinary sites:        charge -= D_0
  //   first reactive site:   charge -= D_1
  //   second reactive site:  charge -= D_2
  //   third reactive site:   charge -= D_3
  //
  // Additional reactive sites, if present, receive D_0 as in the
  // original implementation.
  //====================================================================

  for (int j = 0; j < numAtomsa; ++j) {
    if (j == numAtomso) {
      charge[j] -= d1;
    } else if (j == numAtomso + 1) {
      charge[j] -= d2;
    } else if (j == numAtomso + 2) {
      charge[j] -= d3;
    } else {
      charge[j] -= d0;
    }
  }

  //====================================================================
  // CV and direct GROUPA-GROUPA derivatives
  //
  //   S = -sum_{a<b} charge[a]*charge[b]*r_ab
  //
  // Only the first numAtomso GROUPA sites enter the pair-distance sum.
  //
  //   A[a] = sum_{b != a} charge[b]*r_ab
  //====================================================================

  double cv = 0.0;

  for (int j = 0; j < numAtomso; ++j) {
    for (int k = j + 1; k < numAtomso; ++k) {
      Vector rjk;

      if (pbc) {
        rjk = pbcDistance(getPosition(j), getPosition(k));
      } else {
        rjk = delta(getPosition(j), getPosition(k));
      }

      const double r = rjk.modulo();
      const double qprod = charge[j] * charge[k];

      cv -= qprod * r;

      A[j] += charge[k] * r;
      A[k] += charge[j] * r;

      if (r > tinyR) {
        /*
         * rjk = R_k - R_j.
         *
         * For -q_j*q_k*r_jk:
         *
         *   dS/dR_j = q_j*q_k*(R_k-R_j)/r.
         */
        const Vector dd = (qprod / r) * rjk;

        deriv[j] += dd;
        deriv[k] -= dd;

        virial += Tensor(rjk, dd);
      }
    }
  }

  //====================================================================
  // Abar_h and C_ad,h
  //
  //   Abar_h = sum_i w_ih A_i
  //
  //   C_h = sum_i A_i w_ih (r_ih-rbar_h)
  //       = sum_i A_i w_ih r_ih - rbar_h*Abar_h
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned o = pairO[i];
    const unsigned h = pairH[i];

    abar[h] += w[i] * A[o];
    cad[h] += w[i] * A[o] * pairR[i];
  }

  /*
   * Process each hydrogen once. adaptiveNorm is temporarily used as a
   * visited marker.
   */
  for (unsigned i = 0; i < nn; ++i) {
    const unsigned h = pairH[i];

    if (adaptiveNorm[h] < 0.0) {
      continue;
    }

    cad[h] -= rbar[h] * abar[h];
    adaptiveNorm[h] = -adaptiveNorm[h];
  }

  for (unsigned i = 0; i < nAtoms; ++i) {
    adaptiveNorm[i] = std::fabs(adaptiveNorm[i]);
  }

  //====================================================================
  // Adaptive charge derivatives
  //
  // Base fixed-lambda contribution:
  //
  //   dS/dR_O =
  //       lambda_eff * w_oh * (A_o-Abar_h) * e_oh
  //
  // where
  //
  //   e_oh = (R_O-R_H)/r.
  //
  // Gate correction:
  //
  //   dS/dR_O |_gate =
  //       -2*lambda_det*kappa_h*C_h
  //       *u_oh*(u_oh-P_h)*e_oh
  //
  // The hydrogen derivative is the negative sum of the corresponding
  // oxygen derivatives, so each contribution is accumulated pairwise.
  //====================================================================

  for (unsigned i = 0; i < nn; ++i) {
    const unsigned o = pairO[i];
    const unsigned h = pairH[i];

    const double baseCoefficient =
        lambdaEff[h] * w[i] * (A[o] - abar[h]);

    const double gateCoefficient =
        -2.0 * lambdaDet * kappa[h] * cad[h] *
        u[i] * (u[i] - purity[h]);

    const Vector dd =
        (baseCoefficient + gateCoefficient) * pairEOH[i];

    deriv[o] += dd;
    deriv[h] -= dd;

    /*
     * pairRvec = R_H-R_O and dd is added to O and subtracted from H,
     * matching the convention used by the original VORONOID1 action.
     */
    virial += Tensor(pairRvec[i], dd);
  }

  for (unsigned i = 0; i < nAtoms; ++i) {
    setAtomsDerivatives(i, deriv[i]);
  }

  setValue(cv);
  setBoxDerivatives(virial);
}

} // namespace colvar
} // namespace PLMD
