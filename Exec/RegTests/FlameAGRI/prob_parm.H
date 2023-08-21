#ifndef PROB_PARM_H
#define PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  amrex::Real p = 1013250.0;
  amrex::Real T = 300.0;
  amrex::Real rho = 0.001293;
  amrex::Real eint = 0.0;
  amrex::Real vx_in_1 = 112.7;
  amrex::Real vx_in_2 = 43;
  amrex::Real vy_in = 0.0;
  amrex::Real Re_L = 700.0;
  amrex::Real Pr = 0.7;
  amrex::Real phi = 3.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.,.2345,0.,0.,0.,0.,0.,0.,0.7655};
};

struct ProbParmHost
{
  ProbParmHost() = default;
};

#endif
