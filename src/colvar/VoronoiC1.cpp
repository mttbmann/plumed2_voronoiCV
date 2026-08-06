/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2020 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Communicator.h"
#include "tools/NeighborList.h"
#include "tools/OpenMP.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace PLMD {
namespace colvar {

class VoronoiC1 : public Colvar {
private:
  bool pbc;
  bool serial;

  std::unique_ptr<NeighborList> nl;

  bool invalidateList;
  bool firsttime;

  double lambda;

  int nrx;
  int num_atomsa;
  int num_atomso;

  double d0;
  double d1;
  double d2;
  double d3;

public:
  explicit VoronoiC1(const ActionOptions&);
  ~VoronoiC1();

  void calculate() override;
  void prepare() override;

  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(VoronoiC1, "VORONOIC1")


void VoronoiC1::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);

  keys.addFlag(
      "SERIAL",
      false,
      "Perform the calculation in serial - for debug purpose"
  );

  keys.addFlag(
      "PAIR",
      false,
      "Pair only 1st element of the 1st group with 1st element in the "
      "second, etc"
  );

  keys.addFlag(
      "NLIST",
      false,
      "Use a neighbor list to speed up the calculation"
  );

  keys.add(
      "optional",
      "NL_CUTOFF",
      "The cutoff for the neighbor list"
  );

  keys.add(
      "optional",
      "NL_STRIDE",
      "The frequency with which we are updating the atoms in the "
      "neighbor list"
  );

  keys.add(
      "atoms",
      "GROUPA",
      "First list of atoms"
  );

  keys.add(
      "atoms",
      "GROUPB",
      "Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are "
      "counted)"
  );

  keys.add(
      "compulsory",
      "LAMBDA",
      "1",
      "The lambda parameter of the sum_exp function; 0 implies 1"
  );

  keys.add(
      "compulsory",
      "D_0",
      "0.0",
      "The d_0 parameter of the switching function"
  );

  /*
   * These keywords are retained for compatibility with the original
   * action interface. The current VORONOIC1 calculation uses D_0 for
   * the active GROUPA atoms.
   */
  keys.add(
      "compulsory",
      "D_1",
      "0.0",
      "The d_1 parameter of the switching function"
  );

  keys.add(
      "compulsory",
      "D_2",
      "0.0",
      "The d_2 parameter of the switching function"
  );

  keys.add(
      "compulsory",
      "D_3",
      "0.0",
      "The d_3 parameter of the switching function"
  );

  keys.add(
      "compulsory",
      "NRX",
      "0.0",
      "The number of reactive sites"
  );

  keys.setValueDescription(
      "the Voronoi collective variable: number of ions"
  );
}


VoronoiC1::VoronoiC1(const ActionOptions& ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false),
  invalidateList(true),
  firsttime(true),
  lambda(1.0),
  nrx(0),
  num_atomsa(0),
  num_atomso(0),
  d0(0.0),
  d1(0.0),
  d2(0.0),
  d3(0.0)
{
  parseFlag("SERIAL", serial);

  std::vector<AtomNumber> ga_lista;
  std::vector<AtomNumber> gb_lista;

  parseAtomList("GROUPA", ga_lista);
  parseAtomList("GROUPB", gb_lista);

  num_atomsa = static_cast<int>(ga_lista.size());

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  parse("D_0", d0);
  parse("D_1", d1);
  parse("D_2", d2);
  parse("D_3", d3);
  parse("NRX", nrx);
  parse("LAMBDA", lambda);

  /*
   * Preserve the original definition: the first
   * size(GROUPA)-NRX atoms are active in the final charge sum.
   */
  num_atomso = num_atomsa - nrx;

  // Pair handling.
  bool dopair = false;
  parseFlag("PAIR", dopair);

  // Neighbor-list handling.
  bool doneigh = false;
  double nl_cut = 0.0;
  int nl_st = 0;

  parseFlag("NLIST", doneigh);

  if(doneigh) {
    parse("NL_CUTOFF", nl_cut);

    if(nl_cut <= 0.0) {
      error("NL_CUTOFF should be explicitly specified and positive");
    }

    parse("NL_STRIDE", nl_st);

    if(nl_st <= 0) {
      error("NL_STRIDE should be explicitly specified and positive");
    }
  }

  addValueWithDerivatives();
  setNotPeriodic();

  if(!gb_lista.empty()) {
    if(doneigh) {
      nl = Tools::make_unique<NeighborList>(
          ga_lista,
          gb_lista,
          serial,
          dopair,
          pbc,
          getPbc(),
          comm,
          nl_cut,
          nl_st
      );
    } else {
      nl = Tools::make_unique<NeighborList>(
          ga_lista,
          gb_lista,
          serial,
          dopair,
          pbc,
          getPbc(),
          comm
      );
    }
  } else {
    if(doneigh) {
      nl = Tools::make_unique<NeighborList>(
          ga_lista,
          serial,
          pbc,
          getPbc(),
          comm,
          nl_cut,
          nl_st
      );
    } else {
      nl = Tools::make_unique<NeighborList>(
          ga_lista,
          serial,
          pbc,
          getPbc(),
          comm
      );
    }
  }

  requestAtoms(nl->getFullAtomList());

  log.printf(
      "  between two groups of %u and %u atoms\n",
      static_cast<unsigned>(ga_lista.size()),
      static_cast<unsigned>(gb_lista.size())
  );

  log.printf("  first group:\n");

  for(std::size_t i = 0; i < ga_lista.size(); ++i) {
    if((i + 1) % 25 == 0) {
      log.printf("  \n");
    }

    log.printf("  %d", ga_lista[i].serial());
  }

  log.printf("  \n  second group:\n");

  for(std::size_t i = 0; i < gb_lista.size(); ++i) {
    if((i + 1) % 25 == 0) {
      log.printf("  \n");
    }

    log.printf("  %d", gb_lista[i].serial());
  }

  log.printf("  \n");

  if(pbc) {
    log.printf("  using periodic boundary conditions\n");
  } else {
    log.printf("  without periodic boundary conditions\n");
  }

  if(dopair) {
    log.printf("  with PAIR option\n");
  }

  if(doneigh) {
    log.printf("  using neighbor lists with\n");
    log.printf(
        "  update every %d steps and cutoff %f\n",
        nl_st,
        nl_cut
    );
  }
}


VoronoiC1::~VoronoiC1() {
  // Destructor retained for ownership of the NeighborList.
}


void VoronoiC1::prepare() {
  if(nl->getStride() > 0) {
    if(firsttime || getStep() % nl->getStride() == 0) {
      requestAtoms(nl->getFullAtomList());

      invalidateList = true;
      firsttime = false;
    } else {
      /*
       * Preserve the original behavior: always request the full list so
       * that the local atom ordering remains fixed.
       */
      requestAtoms(nl->getFullAtomList());

      invalidateList = false;

      if(getExchangeStep()) {
        error(
            "Neighbor lists should be updated on exchange steps - choose "
            "a NL_STRIDE which divides the exchange stride!"
        );
      }
    }

    if(getExchangeStep()) {
      firsttime = true;
    }
  }
}


void VoronoiC1::calculate() {
  double totcharge = 0.0;

  Tensor virial;
  virial.zero();

  std::vector<Vector> deriv(getNumberOfAtoms());

  Vector zeros;
  zeros.zero();

  std::fill(deriv.begin(), deriv.end(), zeros);

  if(nl->getStride() > 0 && invalidateList) {
    nl->update(getPositions());
  }

  /*
   * Preserve the original use of communicator size in the OpenMP
   * thread-count heuristic. The loops themselves are not distributed
   * over MPI ranks.
   */
  const unsigned stride = serial
                          ? 1u
                          : static_cast<unsigned>(comm.Get_size());

  unsigned nt = OpenMP::getNumThreads();

  // Number of pairs in the NeighborList pair list.
  const unsigned nn = nl->size();

  if(nt * stride * 10u > nn) {
    nt = 1;
  }

  // Exponential weight for every GROUPA-GROUPB pair.
  std::vector<double> nnexp(nn, 0.0);

  // Local atom indices for every pair.
  std::vector<unsigned> nni0(nn, 0u);
  std::vector<unsigned> nni1(nn, 0u);

  // Normalization for each local GROUPB atom.
  std::vector<double> nnexpnorm(
      getNumberOfAtoms(),
      0.0
  );

  // Normalized Voronoi weights c[o][h].
  std::vector<std::vector<double>> c(
      getNumberOfAtoms(),
      std::vector<double>(getNumberOfAtoms(), 0.0)
  );

  // Voronoi-assigned charge on each GROUPA atom.
  std::vector<double> charge(
      static_cast<std::size_t>(num_atomsa),
      0.0
  );

  // GROUPA-GROUPB displacement vectors.
  std::vector<std::vector<Vector>> distAB(
      static_cast<std::size_t>(num_atomsa),
      std::vector<Vector>(getNumberOfAtoms())
  );

  // Inverse GROUPA-GROUPB distances.
  std::vector<std::vector<double>> distABinvmod(
      static_cast<std::size_t>(num_atomsa),
      std::vector<double>(getNumberOfAtoms(), 0.0)
  );

  /*
   * A(m) is the shifted charge for active GROUPA atoms and zero for
   * inactive/reactive GROUPA atoms.
   *
   * abar(h) = sum_o c[o][h] A[o].
   */
  std::vector<double> A(
      static_cast<std::size_t>(num_atomsa),
      0.0
  );

  std::vector<double> abar(
      getNumberOfAtoms(),
      0.0
  );

#pragma omp parallel num_threads(nt)
  {
    /*
     * Thread-local derivative and virial accumulators.
     */
    std::vector<Vector> omp_deriv(getPositions().size());
    std::fill(omp_deriv.begin(), omp_deriv.end(), zeros);

    Tensor omp_virial;
    omp_virial.zero();

    std::vector<double> charget(
        static_cast<std::size_t>(num_atomsa),
        0.0
    );

    std::vector<double> nnexpnormt(
        getNumberOfAtoms(),
        0.0
    );

    /*
     * LOOP1:
     * Compute GROUPA-GROUPB displacements, exponential weights, and
     * normalization denominators.
     */
#pragma omp for
    for(unsigned i = 0; i < nn; ++i) {
      const unsigned i0 = nl->getClosePair(i).first;
      const unsigned i1 = nl->getClosePair(i).second;

      if(pbc) {
        distAB[i0][i1] = pbcDistance(
            getPosition(i0),
            getPosition(i1)
        );
      } else {
        distAB[i0][i1] = delta(
            getPosition(i0),
            getPosition(i1)
        );
      }

      const double distance = distAB[i0][i1].modulo();

      distABinvmod[i0][i1] = 1.0 / distance;
      nnexp[i] = std::exp(lambda * distance);

      nni0[i] = i0;
      nni1[i] = i1;

      nnexpnormt[i1] += nnexp[i];
    }

#pragma omp critical
    {
      for(std::size_t i = 0; i < getNumberOfAtoms(); ++i) {
        nnexpnorm[i] += nnexpnormt[i];
      }
    }

#pragma omp barrier

    /*
     * LOOP2:
     * Compute normalized Voronoi weights and unshifted GROUPA charges.
     */
#pragma omp for
    for(unsigned i = 0; i < nn; ++i) {
      c[nni0[i]][nni1[i]] =
          nnexp[i] / nnexpnorm[nni1[i]];

      charget[nni0[i]] += c[nni0[i]][nni1[i]];
    }

#pragma omp critical
    {
      for(int i = 0; i < num_atomsa; ++i) {
        const std::size_t is = static_cast<std::size_t>(i);
        charge[is] += charget[is];
      }
    }

#pragma omp barrier

    /*
     * LOOP2.5:
     * Shift active GROUPA charges by D_0, calculate the squared-charge
     * CV, and populate A.
     *
     * This preserves the original definition:
     *
     *     totcharge = sum_{j < num_atomso} (charge[j] - d0)^2
     */
#pragma omp for reduction(+:totcharge)
    for(int j = 0; j < num_atomso; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);

      charge[js] -= d0;
      totcharge += std::pow(charge[js], 2);
      A[js] = charge[js];
    }

    /*
     * LOOP4, pass 1:
     *
     *     abar(h) = sum_o c[o][h] A[o]
     */
    std::vector<double> abart(
        getNumberOfAtoms(),
        0.0
    );

#pragma omp for
    for(unsigned i = 0; i < nn; ++i) {
      const unsigned ind0 = nni0[i];
      const unsigned ind1 = nni1[i];

      abart[ind1] += c[ind0][ind1] * A[ind0];
    }

#pragma omp critical
    {
      for(std::size_t i = 0; i < getNumberOfAtoms(); ++i) {
        abar[i] += abart[i];
      }
    }

#pragma omp barrier

    /*
     * LOOP4, pass 2:
     * Compute the analytical derivatives using A(k)-abar(h).
     *
     * The minus sign is retained exactly from the original implementation:
     *
     *     buf = -2 lambda c[k][h] (A[k] - abar[h])
     */
#pragma omp for
    for(unsigned i = 0; i < nn; ++i) {
      const unsigned k = nni0[i];
      const unsigned h = nni1[i];

      const double buf =
          -2.0
          * lambda
          * c[k][h]
          * (A[k] - abar[h]);

      const Vector dd(
          buf
          * distABinvmod[k][h]
          * distAB[k][h]
      );

      omp_deriv[k] += dd;
      omp_deriv[h] -= dd;

      /*
       * Preserve the original virial convention.
       */
      omp_virial += Tensor(distAB[k][h], dd);
    }

#pragma omp critical
    {
      for(std::size_t i = 0; i < getPositions().size(); ++i) {
        deriv[i] += omp_deriv[i];
      }

      virial += omp_virial;
    }

#pragma omp barrier
  }

  for(std::size_t i = 0; i < deriv.size(); ++i) {
    setAtomsDerivatives(i, deriv[i]);
  }

  setValue(totcharge);
  setBoxDerivatives(virial);
}

} // namespace colvar
} // namespace PLMD
