#ifndef PROB_H
#define PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "mechanism.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "Constants.H"
#include "PelePhysics.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
  const auto prob_low  = geomdata.ProbLo();
  const auto prob_high = geomdata.ProbHi();
  const auto dx =  geomdata.CellSize();
  AMREX_D_TERM(
	       const amrex::Real x = prob_low[0] + (i + 0.5) * dx[0];
	       , const amrex::Real y = prob_low[1] + (j + 0.5) * dx[1];
	       , const amrex::Real z = prob_low[2] + (k + 0.5) * dx[2];);
  auto eos = pele::physics::PhysicsType::eos();
    
  if(y<2.0 && x<20.0){
    double massfracl[NUM_SPECIES]={0};
    double mnew = .7655+.2345*(1+prob_parm.phi/8);
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfracl[n]= prob_parm.massfrac[n]/mnew;
    }
    massfracl[0] = 1- massfracl[1]- massfracl[8];
    
    amrex::Real pres = prob_parm.p;
    amrex::Real rho = 0.0;
    amrex::Real energy = 0.0;
    amrex::Real T = prob_parm.T;
    eos.PYT2RE(pres, massfracl, T, rho, energy);
    state(i, j, k, URHO) = rho;
    state(i, j, k, UMX) = rho * prob_parm.vx_in_1;
    state(i, j, k, UMY) = rho * prob_parm.vy_in;
    state(i, j, k, UMZ) = 0.0;
    state(i, j, k, UEINT) = rho * energy;
    state(i, j, k, UEDEN) =rho *(energy + 0.5 *(prob_parm.vx_in_1
	    * prob_parm.vx_in_1 + prob_parm.vy_in * prob_parm.vy_in));
    state(i, j, k, UTEMP) = T;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i,j,k,UFS + n) = rho * massfracl[n];
    }

    

    
  }else if(y<2.0 && x>=20.0 && x<20.5 ){
    double massfracl[NUM_SPECIES]={0};
    double mnew = .7655+.2345*(1+prob_parm.phi/8);
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfracl[n]= prob_parm.massfrac[n]/mnew;
    }
    massfracl[0] = 1- massfracl[1]- massfracl[8];
    
    amrex::Real pres = prob_parm.p;
    amrex::Real rho = 0.0;
    amrex::Real energy = 0.0;
    amrex::Real T = 1000;
    eos.PYT2RE(pres, massfracl, T, rho, energy);
    state(i, j, k, URHO) = rho;
    state(i, j, k, UMX) = rho * prob_parm.vx_in_1;
    state(i, j, k, UMY) = rho * prob_parm.vy_in;
    state(i, j, k, UMZ) = 0.0;
    state(i, j, k, UEINT) = rho * energy;
    state(i, j, k, UEDEN) =rho *(energy + 0.5 *(prob_parm.vx_in_1
	    * prob_parm.vx_in_1 + prob_parm.vy_in * prob_parm.vy_in));
    state(i, j, k, UTEMP) = T;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i,j,k,UFS + n) = rho * massfracl[n];
    }
  }else{
    double massfracl[NUM_SPECIES]={0};
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfracl[n]= prob_parm.massfrac[n];
    }
    amrex::Real pres = prob_parm.p;
    amrex::Real rho = 0.0;
    amrex::Real energy = 0.0;
    amrex::Real T = prob_parm.T;
    eos.PYT2RE(pres, massfracl, T, rho, energy);
    state(i, j, k, URHO) = rho;
    state(i, j, k, UMX) = rho * prob_parm.vx_in_2;
    state(i, j, k, UMY) = rho * prob_parm.vy_in;
    state(i, j, k, UMZ) = 0.0;
    state(i, j, k, UEINT) = rho * energy;
    state(i, j, k, UEDEN) = rho * (energy + 0.5 * (prob_parm.vx_in_2 *
	      prob_parm.vx_in_2 + prob_parm.vy_in * prob_parm.vy_in));
    state(i, j, k, UTEMP) = T;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, UFS + n) = rho * massfracl[n];
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real* /*s_int[NVAR]*/,
  amrex::Real s_ext[NVAR],
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
  auto eos = pele::physics::PhysicsType::eos();

  if (x[1] <= 2.0) {
    double massfracl[NUM_SPECIES]={0};
    double mnew = .7655+.2345*(1+prob_parm.phi/8);
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfracl[n]= prob_parm.massfrac[n]/mnew;
    }
    massfracl[0] = 1- massfracl[1]- massfracl[8];
    
    amrex::Real pres = prob_parm.p;
    amrex::Real rho = 0.0;
    amrex::Real energy = 0.0;
    amrex::Real T = prob_parm.T;
    eos.PYT2RE(pres, massfracl, T, rho, energy);
    s_ext[URHO] = rho;
    s_ext[UMX] = rho * prob_parm.vx_in_1;
    s_ext[UMY] = rho * prob_parm.vy_in;
    s_ext[UMZ] = 0.0;
    s_ext[UEINT] = rho * energy;
    s_ext[UEDEN] =rho *(energy + 0.5 *(prob_parm.vx_in_1
	    * prob_parm.vx_in_1 + prob_parm.vy_in * prob_parm.vy_in));
    s_ext[UTEMP] = T;
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[UFS + n] = rho * massfracl[n];
    }
  }else{
    double massfracl[NUM_SPECIES]={0};
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfracl[n]= prob_parm.massfrac[n];
    }
    amrex::Real pres = prob_parm.p;
    amrex::Real rho = 0.0;
    amrex::Real energy = 0.0;
    amrex::Real T = prob_parm.T;
    eos.PYT2RE(pres, massfracl, T, rho, energy);
    s_ext[URHO] = rho;
    s_ext[UMX] = rho * prob_parm.vx_in_2;
    s_ext[UMY] = rho * prob_parm.vy_in;
    s_ext[UMZ] = 0.0;
    s_ext[UEINT] = rho * energy;
    s_ext[UEDEN] =rho *(energy + 0.5 *(prob_parm.vx_in_2
	    * prob_parm.vx_in_2 + prob_parm.vy_in * prob_parm.vy_in));
    s_ext[UTEMP] = T;
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[UFS + n] = rho * massfracl[n];
    }
  }
}

void pc_prob_close();

using ProblemTags = EmptyProbTagStruct;
using ProblemDerives = EmptyProbDeriveStruct;

#endif
