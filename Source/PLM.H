#ifndef _PLM_H_
#define _PLM_H_

#include <cmath>

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>

#include "Constants.H"
#include "IndexDefines.H"
#include "PelePhysics.H"
#include "Riemann.H"

// This header file contains functions and declarations for the 3D hydro portion
// of PeleC on the GPU. As per the convention of AMReX, inlined functions are
// defined here. Where as non-inline functions are declared here.

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
plm_slope(
  const int i,
  const int j,
  const int k,
  const int n,
  const int dir,
  amrex::Array4<const amrex::Real> const& q,
  const amrex::Real rho_scale = false)
{
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivm2(iv - 2 * amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ivm(iv - amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ivp(iv + amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ivp2(iv + 2 * amrex::IntVect::TheDimensionVector(dir));

  amrex::Real dlft, drgt, dcen, dfm, dfp, dlim, dsgn, dtemp;
  const amrex::Real qc = q(iv, n);
  const amrex::Real qm2 = q(ivm2, n);
  const amrex::Real qm = q(ivm, n);
  const amrex::Real qp = q(ivp, n);
  const amrex::Real qp2 = q(ivp2, n);

  if(rho_scale){
    qc*=q(iv,QRHO);
    qm2*=q(ivm2, QRHO);
    qm*=q(ivm, QRHO);
    qp*=q(ivp, QRHO);
    qp2*=q(ivp2, QRHO);
  }
  
  dlft = qm - qm2;
  drgt = qc - qm;
  dcen = 0.5 * (dlft + drgt);
  dsgn = amrex::Math::copysign(1.0, dcen);
  dlim = (dlft * drgt >= 0.0)
           ? 2.0 * amrex::min<amrex::Real>(
                     amrex::Math::abs(dlft), amrex::Math::abs(drgt))
           : 0.0;
  dfm = dsgn * amrex::min<amrex::Real>(dlim, amrex::Math::abs(dcen));

  dlft = qp - qc;
  drgt = qp2 - qp;
  dcen = 0.5 * (dlft + drgt);
  dsgn = amrex::Math::copysign(1.0, dcen);
  dlim = (dlft * drgt >= 0.0)
           ? 2.0 * amrex::min<amrex::Real>(
                     amrex::Math::abs(dlft), amrex::Math::abs(drgt))
           : 0.0;
  dfp = dsgn * amrex::min<amrex::Real>(dlim, amrex::Math::abs(dcen));

  dlft = qc - qm;
  drgt = qp - qc;
  dcen = 0.5 * (dlft + drgt);
  dsgn = amrex::Math::copysign(1.0, dcen);
  dlim = (dlft * drgt >= 0.0)
           ? 2.0 * amrex::min<amrex::Real>(
                     amrex::Math::abs(dlft), amrex::Math::abs(drgt))
           : 0.0;
  dcen = dsgn * amrex::min<amrex::Real>(dlim, amrex::Math::abs(dcen));

  const amrex::Real dtemp = 4.0 / 3.0 * dcen - 1.0 / 6.0 * (dfp + dfm);

  // Flattening could be done here (see Nyx if we want to do it)
  return dsgn * amrex::min<amrex::Real>(dlim, amrex::Math::abs(dtemp));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_plm_d(
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Real const slope[QVAR],
  amrex::Array4<const amrex::Real> const& q,
  amrex::Real const cc,
  amrex::Real const dx,
  amrex::Real const dt,
  PassMap const& pmap)
{
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivp(iv + amrex::IntVect::TheDimensionVector(dir));
  const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
  const amrex::GpuArray<const int, 3> l_idx{
    {bdim[0] * 0 + bdim[1] * 1 + bdim[2] * 2,
     bdim[0] * 1 + bdim[1] * 0 + bdim[2] * 0,
     bdim[0] * 2 + bdim[1] * 2 + bdim[2] * 1}};

  const amrex::Real dtdx = dt / dx;
  const amrex::Real cs = cc * cc;
  amrex::Real rho = 0.0;
  amrex::Real rhoY[NUM_SPECIES] = {0.0};
  amrex::Real drho = 0.0;
  amrex::Real drhoY[NUM_SPECIES] = {0.0};
  for (int sp = 0; sp < NUM_SPECIES; sp++) {
    rhoY[sp] = q(iv, QRHO) * q(iv, QFS + sp);
    drhoY[sp] = slope[QFS + sp];
    rho += rhoY[sp];
    drho += drhoY[sp];
  }
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel{
    AMREX_D_DECL(q(iv, QU), q(iv, QV), q(iv, QW))};
  const amrex::Real p = q(iv, QPRES);
  const amrex::Real rhoe = q(iv, QREINT);
  const amrex::Real enth = ((rhoe + p) / rho) / cs;
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dvel{
    AMREX_D_DECL(slope[QU], slope[QV], slope[QW])};
  const amrex::Real dp = slope[QPRES];
  const amrex::Real drhoe = slope[QREINT];
  const amrex::Real alpham = 0.5 * (dp / (rho * cc) - dvel[dir]) * rho / cc;
  const amrex::Real alphap = 0.5 * (dp / (rho * cc) + dvel[dir]) * rho / cc;

  amrex::Real alpha0r = 0.0;
  amrex::Real alpha0rY[NUM_SPECIES] = {0.0};
  for(int sp=0; sp<NUM_SPECIES; sp++){
    alpha0rY[sp] = drhoY[sp] - q(i,j,k,QFS+sp) * dp/cs;
    alpha0r += alpha0rY[sp];
  }
  const amrex::Real alpha0e = drhoe - dp * enth;
  AMREX_D_TERM(, const amrex::Real alpha0v = dvel[l_idx[1]];
               , const amrex::Real alpha0w = dvel[l_idx[2]];)
  const amrex::GpuArray<amrex::Real, 3> wv = {
    vel[dir] - cc, vel[dir], vel[dir] + cc};

  //  Construct the right state on the i-1/2 interface

  amrex::Real rho_ref = 0.0;
  amrex::Real rhoY_ref[NUM_SPECIES] = {0.0};
  for(int sp=0; sp<NUM_SPECIES; sp++){
    rhoY_ref[sp] = rhoY[sp] - 0.5 * (1.0 + dtdx * amrex::min<amrex::Real>(wv[0], 0.0)) * drhoY[sp];
    rho_ref += rhoY_ref[sp];
  }
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_ref{
    AMREX_D_DECL(0.0, 0.0, 0.0)};
  for (int dim = 0; dim < AMREX_SPACEDIM; dim++) {
    vel_ref[dim] =
      vel[dim] -
      0.5 * (1.0 + dtdx * amrex::min<amrex::Real>(wv[0], 0.0)) * dvel[dim];
  }
  amrex::Real p_ref =
    p - 0.5 * (1.0 + dtdx * amrex::min<amrex::Real>(wv[0], 0.0)) * dp;
  amrex::Real rhoe_ref =
    rhoe - 0.5 * (1.0 + dtdx * amrex::min<amrex::Real>(wv[0], 0.0)) * drhoe;

  const amrex::Real apright = 0.25 * dtdx * (wv[0] - wv[2]) *
                              (1.0 - amrex::Math::copysign(1.0, wv[2])) *
                              alphap;
  const amrex::Real amright = 0.0;

  amrex::Real azrright = 0.0;
  amrex::Real azrYright[NUM_SPECIES] = {0.0};
  for(int sp=0; sp<NUM_SPECIES; sp++){
    azrYright[sp] = 0.25 * dtdx * (wv[0] - wv[1]) *
                               (1.0 - amrex::Math::copysign(1.0, wv[1])) *
                               alpha0rY[sp];
    azrright += azrYright[sp];
  }
  const amrex::Real azeright = 0.25 * dtdx * (wv[0] - wv[1]) *
                               (1.0 - amrex::Math::copysign(1.0, wv[1])) *
                               alpha0e;
  AMREX_D_TERM(, const amrex::Real azv1rght =
                   0.25 * dtdx * (wv[0] - wv[1]) *
                   (1.0 - amrex::Math::copysign(1.0, wv[1])) * alpha0v;
               , const amrex::Real azw1rght =
                   0.25 * dtdx * (wv[0] - wv[1]) *
                   (1.0 - amrex::Math::copysign(1.0, wv[1])) * alpha0w;)

  qp(iv, QRHO) = 0.0 ;
  for(int sp=0; sp<NUM_SPECIES; sp++){ // MHDF rhoY or rhoY_ref?
    qp(iv, QFS+sp) = rhoY_ref[sp] + rhoY[sp] * (apright + amright) + azrYright[sp];
    qp(iv, QRHO) += qp(iv, QFS+sp);
  }
  qp(iv, QRHO) = amrex::max<amrex::Real>(
    std::numeric_limits<amrex::Real>::min(), qp(iv, QRHO));
  for(int sp=0; sp<NUM_SPECIES; sp++){
    qp(iv, QFS+sp) /=  qp(iv, QRHO);
  }
  AMREX_D_TERM(qp(iv, QU + l_idx[0]) =
                 vel_ref[l_idx[0]] + (apright - amright) * cc / rho;
               qp(iv, QU + l_idx[1]) = 0.; qp(iv, QU + l_idx[2]) = 0.;
               , qp(iv, QU + l_idx[1]) = vel_ref[l_idx[1]] + azv1rght;
               , qp(iv, QU + l_idx[2]) = vel_ref[l_idx[2]] + azw1rght;);
  qp(iv, QPRES) = p_ref + (apright + amright) * cs;
  qp(iv, QPRES) = amrex::max<amrex::Real>(
    qp(iv, QPRES), std::numeric_limits<amrex::Real>::min());
  qp(iv, QREINT) = rhoe_ref + (apright + amright) * enth * cs + azeright;

  //  Construct the left state on the i+1/2 interface

  rho_ref = 0.0;
  rhoY_ref[NUM_SPECIES] = {0.0};
  for(int sp=0; sp<NUM_SPECIES; sp++){
    rhoY_ref[sp] = rhoY[sp] + 0.5 * (1.0 + dtdx * amrex::min<amrex::Real>(wv[2], 0.0)) * drhoY[sp];
    rho_ref += rhoY_ref[sp];
  }
  for (int dim = 0; dim < AMREX_SPACEDIM; dim++) {
    vel_ref[dim] =
      vel[dim] +
      0.5 * (1.0 - dtdx * amrex::max<amrex::Real>(wv[2], 0.0)) * dvel[dim];
  }
  p_ref = p + 0.5 * (1.0 - dtdx * amrex::max<amrex::Real>(wv[2], 0.0)) * dp;
  rhoe_ref =
    rhoe + 0.5 * (1.0 - dtdx * amrex::max<amrex::Real>(wv[2], 0.0)) * drhoe;

  const amrex::Real apleft = 0.0;
  const amrex::Real amleft = 0.25 * dtdx * (wv[2] - wv[0]) *
                             (1.0 + amrex::Math::copysign(1.0, wv[0])) * alpham;

  amrex::Real azrleft = 0.0;
  amrex::Real azrYleft[NUM_SPECIES] = {0.0};
  for(int sp=0; sp<NUM_SPECIES; sp++){
    azrYleft[sp] = 0.25 * dtdx * (wv[2] - wv[1]) *
                               (1.0 + amrex::Math::copysign(1.0, wv[1])) *
                               alpha0rY[sp];
    azrleft += azrYleft[sp];
  }
  const amrex::Real azeleft = 0.25 * dtdx * (wv[2] - wv[1]) *
                              (1.0 + amrex::Math::copysign(1.0, wv[1])) *
                              alpha0e;
  AMREX_D_TERM(, const amrex::Real azv1left =
                   0.25 * dtdx * (wv[2] - wv[1]) *
                   (1.0 + amrex::Math::copysign(1.0, wv[1])) * alpha0v;
               , const amrex::Real azw1left =
                   0.25 * dtdx * (wv[2] - wv[1]) *
                   (1.0 + amrex::Math::copysign(1.0, wv[1])) * alpha0w;)

  qm(ivp, QRHO) = 0.0 ;
  for(int sp=0; sp<NUM_SPECIES; sp++){ // MHDF rhoY or rhoY_ref?
    qm(ivp, QFS+sp) = rhoY_ref[sp] + rhoY[sp] * (apleft + amleft) + azrYleft[sp];
    qm(ivp, QRHO) += qm(ivp, QFS+sp);
  }
  qm(ivp, QRHO) = amrex::max<amrex::Real>(
    qm(ivp, QRHO), std::numeric_limits<amrex::Real>::min());
  for (int sp = 0; sp < NUM_SPECIES; sp++) {
    qm(ivp, QFS + sp) /= qm(ivp, QRHO);
  }
  AMREX_D_TERM(qm(ivp, QU + l_idx[0]) =
                 vel_ref[l_idx[0]] + (apleft - amleft) * cc / rho;
               qm(ivp, QU + l_idx[1]) = 0.; qm(ivp, QU + l_idx[2]) = 0.;
               , qm(ivp, QU + l_idx[1]) = vel_ref[l_idx[1]] + azv1left;
               , qm(ivp, QU + l_idx[2]) = vel_ref[l_idx[2]] + azw1left;);
  qm(ivp, QPRES) = p_ref + (apleft + amleft) * cs;
  qm(ivp, QPRES) = amrex::max<amrex::Real>(
    qm(ivp, QPRES), std::numeric_limits<amrex::Real>::min());
  qm(ivp, QREINT) = rhoe_ref + (apleft + amleft) * enth * cs + azeleft;

  // Upwind the passive variables
  for (int ipassive = 0; ipassive < NPASSIVE; ++ipassive) {
    const int n = pmap.qpassMap[ipassive];
    // Right state
    const amrex::Real vel_ad = q(iv, QU + l_idx[0]);
    amrex::Real spzero = (vel_ad > 0) ? -1.0 : vel_ad * dtdx;
    const amrex::Real acmprght = 0.5 * (-1.0 - spzero) * slope[n];
    qp(iv, n) = q(iv, n) + acmprght;

    spzero = vel_ad >= 0 ? vel_ad * dtdx : 1.0;
    const amrex::Real acmpleft = 0.5 * (1.0 - spzero) * slope[n];
    qm(ivp, n) = q(iv, n) + acmpleft;
  }
}

#endif
