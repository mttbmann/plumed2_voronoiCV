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
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "Colvar.h"

#include "tools/Matrix.h"
#include "core/ActionRegister.h"
#include <string>
#include <cmath>
#include <iostream>
using namespace std;

namespace PLMD {
namespace colvar {

class VoronoiC1 : public Colvar {
  bool pbc;
  bool serial;
  std::unique_ptr<NeighborList> nl;
  std::vector<PLMD::AtomNumber> list_a,list_b,list_c;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  bool invalidateList;
  bool firsttime;
  double lambda;
  int nrx, num_atomsa,num_atomsb,num_atoms,num_atomso;
  double d0, d1, d2, d3;

public:
  explicit VoronoiC1(const ActionOptions&);
  ~VoronoiC1();
// active methods:
  void calculate() override;
  void prepare() override;
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(VoronoiC1,"VORONOIC1")

void VoronoiC1::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbor list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbor list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbor list");
  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
  keys.add("compulsory","LAMBDA","1","The lambda parameter of the sum_exp function; 0 implies 1");  
  keys.add("compulsory","D_0","0.0","The d_0 parameter of the switching function");
  keys.add("compulsory","D_1","0.0","The d_1 parameter of the switching function");
  keys.add("compulsory","D_2","0.0","The d_2 parameter of the switching function");
  keys.add("compulsory","D_3","0.0","The d_3 parameter of the switching function");
  keys.add("compulsory","NRX","0.0","The number of reactive sites");
  keys.setValueDescription("scalar","the Voronoi collective variable: number of ions");
}

VoronoiC1::VoronoiC1(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false),
  invalidateList(true),
  firsttime(true)
{

  parseFlag("SERIAL",serial);

  std::vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);

  list_a = ga_lista;
  list_b = gb_lista;
  
  num_atomsa = list_a.size();
  num_atomsb = list_b.size(); 
  num_atoms = num_atomsa + num_atomsb;  

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;  
  
  parse("D_0",d0);
  parse("D_1",d1);
  parse("D_2",d2);
  parse("D_3",d3);
  parse("NRX",nrx);
  parse("LAMBDA",lambda);  
  num_atomso = num_atomsa - nrx;  

// pair stuff
  bool dopair=false;
  parseFlag("PAIR",dopair);

// neighbor list stuff
  bool doneigh=false;
  double nl_cut=0.0;
  int nl_st=0;
  parseFlag("NLIST",doneigh);
  if(doneigh) {
    parse("NL_CUTOFF",nl_cut);
    if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
    parse("NL_STRIDE",nl_st);
    if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  addValueWithDerivatives(); setNotPeriodic();
  if(gb_lista.size()>0) {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,dopair,pbc,getPbc(),comm,nl_cut,nl_st);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,dopair,pbc,getPbc(),comm);
  } else {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm,nl_cut,nl_st);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm);
  }

  requestAtoms(nl->getFullAtomList()); 

  log.printf("  between two groups of %u and %u atoms\n",static_cast<unsigned>(ga_lista.size()),static_cast<unsigned>(gb_lista.size()));
  log.printf("  first group:\n");
  for(unsigned int i=0; i<ga_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0; i<gb_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", gb_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh) {
    log.printf("  using neighbor lists with\n");
    log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }
}

VoronoiC1::~VoronoiC1() {
// destructor required to delete forward declared class
}

void VoronoiC1::prepare() {
  if(nl->getStride()>0) {
    if(firsttime || (getStep()%nl->getStride()==0)) {
      requestAtoms(nl->getFullAtomList()); 
      invalidateList=true;
      firsttime=false;
    } else {
      requestAtoms(nl->getFullAtomList());      
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

// calculator
void VoronoiC1::calculate()
{

  double totcharge=0.0;
  
  Tensor virial;
  vector<Vector> deriv(getNumberOfAtoms());
  Vector zeros;
  zeros.zero();
  fill(deriv.begin(), deriv.end(), zeros);

  if(nl->getStride()>0 && invalidateList) {
    nl->update(getPositions());
  }

  unsigned stride;
  unsigned rank;
  if(serial) {
    stride=1;
    rank=0;
  } else {
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }

  unsigned nt=OpenMP::getNumThreads();
  const unsigned nn=nl->size(); 
  if(nt*stride*10>nn) nt=1;

  vector<double> nnexp(nn);
  vector<unsigned> nni0(nn);
  vector<unsigned> nni1(nn);
  vector<double> nnexpnorm(getNumberOfAtoms()); 
  vector<vector<double>> c(getNumberOfAtoms(),vector<double>(getNumberOfAtoms()));  
  vector<double> charge(num_atomsa); 
  vector<vector<Vector>> distAB(num_atomsa,vector<Vector>(getNumberOfAtoms())); 
  vector<vector<double>> distABinvmod(num_atomsa,vector<double>(getNumberOfAtoms())); 
  
  // --- precomputed quantities to make LOOP4 O(nn) instead of O(nn*num_atomsa) ---
  vector<double> A(num_atomsa, 0.0);          // A(m) = shifted charge(m) for relevant atoms, 0 otherwise
  vector<double> abar(getNumberOfAtoms(), 0.0); // abar(h) = sum_o c[o][h]*A[o]

  #pragma omp parallel num_threads(nt)
  {
    std::vector<Vector> omp_deriv(getPositions().size());  
    Tensor omp_virial;                                      
    vector<double> charget(num_atomsa); 
    vector<double> nnexpnormt(getNumberOfAtoms());

    //LOOP1 compute the distances, the exponentials and the normalizations over groupb elements
    #pragma omp for
    for(unsigned int i=0; i<nn; i+=1) {

        unsigned i0=nl->getClosePair(i).first;  
        unsigned i1=nl->getClosePair(i).second;

        if(pbc) {
          distAB[i0][i1]=pbcDistance(getPosition(i0),getPosition(i1));
        } else {
          distAB[i0][i1]=delta(getPosition(i0),getPosition(i1));
        }

        distABinvmod[i0][i1]=1.0/distAB[i0][i1].modulo();
        nnexp[i]=exp(lambda * distAB[i0][i1].modulo());
        nni0[i]=i0;  
        nni1[i]=i1;
        nnexpnormt[i1]+=nnexp[i]; 
    }
    #pragma omp critical
    for(unsigned i=0; i<getNumberOfAtoms(); i++) nnexpnorm[i]+=nnexpnormt[i];
    #pragma omp barrier

    //LOOP2 compute the unshifted charge on atoms A
    #pragma omp for
    for(unsigned int i=0; i<nn; i+=1) {
      c[nni0[i]][nni1[i]]=nnexp[i]/nnexpnorm[nni1[i]];  
      charget[nni0[i]]+=c[nni0[i]][nni1[i]];
    }
    #pragma omp critical
    for(unsigned i=0; i<num_atomsa; i++) charge[i]+=charget[i];
    #pragma omp barrier

    //LOOP2.5 shift charge by d0, accumulate total charge, populate vector A
    #pragma omp for reduction(+:totcharge)
    for(unsigned int j=0; j<num_atomso; j+=1) {    
      charge[j]-=d0;
      totcharge+=pow(charge[j],2);
      A[j] = charge[j]; // Only the active atoms contribute to the analytical gradient scalar A
    }    

    // --- LOOP4 O(nn) optimization pass ---
    
    // Pass 1: build abar(h) = sum_o c[o][h] * A[o]
    vector<double> abart(getNumberOfAtoms(), 0.0); // thread-local version of abar
    #pragma omp for
    for(unsigned int i=0; i<nn; i+=1) {
      unsigned ind0=nni0[i];
      unsigned ind1=nni1[i];
      abart[ind1] += c[ind0][ind1] * A[ind0];
    }
    #pragma omp critical
    for(unsigned i=0; i<getNumberOfAtoms(); i++) abar[i]+=abart[i];
    #pragma omp barrier

    // Pass 2: compute forces using A(k) - abar(h)
    #pragma omp for
    for(unsigned int i=0; i<nn; i+=1) {
      unsigned k=nni0[i];
      unsigned h=nni1[i];

      // Note: Reversing sign ensures compatibility with D1's correct virial PLUMED convention
      double buf = -2.0 * lambda * c[k][h] * ( A[k] - abar[h] );
	
      Vector dd(buf*distABinvmod[k][h]*distAB[k][h]);
      
      omp_deriv[k]+=dd;
      omp_deriv[h]-=dd;
      omp_virial += Tensor(distAB[k][h],dd); // Add proper thread-local virial tensor
    }

    #pragma omp critical
    {
      for(unsigned i=0; i<getPositions().size(); i++) deriv[i]+=omp_deriv[i];
      virial += omp_virial; // Merge thread-local virial into the global one
    }
    #pragma omp barrier

  }

  for(unsigned i=0; i<deriv.size(); ++i) setAtomsDerivatives(i,deriv[i]);
  setValue           (totcharge);
  setBoxDerivatives  (virial);

}
}
}
