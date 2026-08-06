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

class VoronoiD1 : public Colvar {
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
  explicit VoronoiD1(const ActionOptions&);
  ~VoronoiD1();

  void calculate() override;
  void prepare() override;

  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(VoronoiD1, "VORONOID1")


void VoronoiD1::registerKeywords(Keywords& keys) {
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
      "0",
      "The number of reactive sites"
  );

  keys.setValueDescription(
      "the Voronoi collective variable "
      "(charge-weighted pair-distance sum)"
  );
}


VoronoiD1::VoronoiD1(const ActionOptions& ao):
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
   * This subtraction is deliberately kept equivalent to the original
   * implementation. The last NRX entries of GROUPA are treated as
   * reactive/non-water sites.
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


VoronoiD1::~VoronoiD1() {
  // Destructor retained for ownership of the NeighborList.
}


void VoronoiD1::prepare() {
  if(nl->getStride() > 0) {
    if(firsttime || getStep() % nl->getStride() == 0) {
      requestAtoms(nl->getFullAtomList());

      invalidateList = true;
      firsttime = false;
    } else {
      /*
       * The full list is intentionally requested at every step. Using the
       * reduced list can change the local atom ordering and would require
       * additional local-to-absolute index bookkeeping in calculate().
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


void VoronoiD1::calculate() {
  double IonDistance = 0.0;

  /*
   * getNumberOfAtoms() gives the number of currently requested atoms.
   * Indices used below are local PLUMED atom indices.
   */

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
   * The original implementation used the communicator size only in the
   * OpenMP thread-count heuristic. It did not distribute the loops over
   * MPI ranks. That behavior is retained here.
   */
  const unsigned stride = serial
                          ? 1u
                          : static_cast<unsigned>(comm.Get_size());

  unsigned nt = OpenMP::getNumThreads();

  // Number of pairs in the NeighborList pair list.
  const unsigned nn = nl->size();

  /*
   * The pair list is ordered with a GROUPA atom followed by its GROUPB
   * neighbors.
   */
  if(nt * stride * 10u > nn) {
    nt = 1;
  }

  std::vector<double> nnexp(nn);
  std::vector<unsigned> nni0(nn);
  std::vector<unsigned> nni1(nn);

  // Normalization for each local GROUPB atom.
  std::vector<double> nnexpnorm(getNumberOfAtoms(), 0.0);

  // Voronoi weights stored using local atom indices.
  std::vector<std::vector<double>> c(
      getNumberOfAtoms(),
      std::vector<double>(getNumberOfAtoms(), 0.0)
  );

  /*
   * The full atom list is always requested, so the first num_atomsa local
   * indices correspond to GROUPA.
   */
  std::vector<double> charge(
      static_cast<std::size_t>(num_atomsa),
      0.0
  );

  // GROUPA-GROUPA distances.
  std::vector<std::vector<double>> distGA(
      static_cast<std::size_t>(num_atomsa),
      std::vector<double>(
          static_cast<std::size_t>(num_atomsa),
          0.0
      )
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
   * A(m) = sum over water oxygen n != m of
   *        distGA(m,n) * charge(n)
   *
   * abar(h) = sum over o of c[o][h] * A[o]
   */
  std::vector<double> A(
      static_cast<std::size_t>(num_atomsa),
      0.0
  );

  std::vector<double> abar(getNumberOfAtoms(), 0.0);

#pragma omp parallel num_threads(nt)
  {
    /*
     * Each thread accumulates its own derivative and virial contributions.
     * They are merged at the end of the parallel region.
     */
    std::vector<Vector> omp_deriv(getPositions().size());

    std::fill(
        omp_deriv.begin(),
        omp_deriv.end(),
        zeros
    );

    Tensor omp_virial;
    omp_virial.zero();

    std::vector<double> nnexpnormt(
        getNumberOfAtoms(),
        0.0
    );

    std::vector<double> charget(
        static_cast<std::size_t>(num_atomsa),
        0.0
    );

    /*
     * LOOP1:
     * Compute GROUPA-GROUPB distances, exponentials, and normalization
     * denominators for each GROUPB atom.
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
        charge[static_cast<std::size_t>(i)] +=
            charget[static_cast<std::size_t>(i)];
      }
    }

#pragma omp barrier

    /*
     * LOOP2.5:
     * Apply the requested charge shifts.
     */
#pragma omp for
    for(int j = 0; j < num_atomsa; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);

      if(j == num_atomso) {
        charge[js] -= d1;
      } else if(j == num_atomso + 1) {
        charge[js] -= d2;
      } else if(j == num_atomso + 2) {
        charge[js] -= d3;
      } else {
        charge[js] -= d0;
      }
    }

    /*
     * LOOP3:
     * Compute the CV from pairs of water GROUPA atoms, calculate the direct
     * GROUPA derivatives, and accumulate A(m).
     */
    std::vector<double> At(
        static_cast<std::size_t>(num_atomsa),
        0.0
    );

#pragma omp for reduction(-:IonDistance)
    for(int j = 0; j < num_atomsa; ++j) {
      for(int k = j + 1; k < num_atomsa; ++k) {
        const std::size_t js = static_cast<std::size_t>(j);
        const std::size_t ks = static_cast<std::size_t>(k);

        Vector distance_jk;

        if(pbc) {
          distance_jk = pbcDistance(
              getPosition(js),
              getPosition(ks)
          );
        } else {
          distance_jk = delta(
              getPosition(js),
              getPosition(ks)
          );
        }

        distGA[js][ks] = distance_jk.modulo();

        double buffer;
        double buffer2;

        /*
         * Only distances between water GROUPA sites contribute to the
         * pair-distance CV.
         */
        if(j < num_atomso && k < num_atomso) {
          buffer =
              distGA[js][ks] * charge[js] * charge[ks];

          buffer2 =
              charge[js] * charge[ks] / distGA[js][ks];

          At[js] += charge[ks] * distGA[js][ks];
          At[ks] += charge[js] * distGA[js][ks];
        } else {
          buffer = 0.0;
          buffer2 = 0.0;
        }

        IonDistance -= buffer;

        const Vector dd(buffer2 * distance_jk);

        omp_deriv[js] += dd;
        omp_deriv[ks] -= dd;

        /*
         * Pair virial convention:
         * distance_jk = pbcDistance(position(j), position(k)),
         * derivative on j is +dd, and derivative on k is -dd.
         */
        omp_virial += Tensor(distance_jk, dd);
      }
    }

#pragma omp critical
    {
      for(int i = 0; i < num_atomsa; ++i) {
        const std::size_t is = static_cast<std::size_t>(i);
        A[is] += At[is];
      }
    }

#pragma omp barrier

    /*
     * LOOP4:
     *
     * dCV/d(O-H pair i) =
     *     lambda * c[ind0][ind1] * (A(ind0) - abar(ind1))
     *
     * This is the O(nn) implementation from the original source.
     */
    std::vector<double> abart(
        getNumberOfAtoms(),
        0.0
    );

    // Pass 1: abar(h) = sum_o c[o][h] * A[o].
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

    // Pass 2: calculate GROUPA-GROUPB derivatives.
#pragma omp for
    for(unsigned i = 0; i < nn; ++i) {
      const unsigned ind0 = nni0[i];
      const unsigned ind1 = nni1[i];

      const double buf =
          lambda
          * c[ind0][ind1]
          * (A[ind0] - abar[ind1]);

      const Vector dd(
          buf
          * distABinvmod[ind0][ind1]
          * distAB[ind0][ind1]
      );

      omp_deriv[ind0] += dd;
      omp_deriv[ind1] -= dd;

      omp_virial += Tensor(
          distAB[ind0][ind1],
          dd
      );
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

  setValue(IonDistance);
  setBoxDerivatives(virial);
}

} // namespace colvar
} // namespace PLMD
