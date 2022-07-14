#ifndef RIEMANN_H
#define RIEMANN_H
#include "PeleC.H"
#include "PelePhysics.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
riemann(
  const amrex::Real rl,
  const amrex::Real ul,
  const amrex::Real vl,
  const amrex::Real v2l,
  const amrex::Real pl,
  const amrex::Real spl[NUM_SPECIES],
  const amrex::Real rr,
  const amrex::Real ur,
  const amrex::Real vr,
  const amrex::Real v2r,
  const amrex::Real pr,
  const amrex::Real spr[NUM_SPECIES],
  const int bc_test_val,
  const amrex::Real cav,
  amrex::Real& ustar,
  amrex::Real& uflx_rho,
  amrex::Real uflx_rhoY[NUM_SPECIES],
  amrex::Real& uflx_u,
  amrex::Real& uflx_v,
  amrex::Real& uflx_w,
  amrex::Real& uflx_eden,
  amrex::Real& uflx_eint,
  amrex::Real& qint_iu,
  amrex::Real& qint_iv1,
  amrex::Real& qint_iv2,
  amrex::Real& qint_gdpres,
  amrex::Real& qint_gdgame)
{
  const amrex::Real wsmall = std::numeric_limits<amrex::Real>::min();

  auto eos = pele::physics::PhysicsType::eos();

  amrex::Real gdnv_state_massfrac[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spl[n];
  }
  amrex::Real cl = 0.0;
  eos.RPY2Cs(rl, pl, gdnv_state_massfrac, cl);

  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = spr[n];
  }
  amrex::Real cr = 0.0;
  eos.RPY2Cs(rr, pr, gdnv_state_massfrac, cr);

  const amrex::Real wl = amrex::max<amrex::Real>(wsmall, cl * rl);
  const amrex::Real wr = amrex::max<amrex::Real>(wsmall, cr * rr);
  const amrex::Real pstar = amrex::max<amrex::Real>(
    std::numeric_limits<amrex::Real>::min(),
    ((wr * pl + wl * pr) + wl * wr * (ul - ur)) / (wl + wr));
  ustar = ((wl * ul + wr * ur) + (pl - pr)) / (wl + wr);

  bool mask = ustar > 0.0;
  amrex::Real ro = 0.0;
  amrex::Real rspo[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspo[n] = mask ? rl * spl[n] : rr * spr[n];
    ro += rspo[n];
  }
  amrex::Real uo = mask ? ul : ur;
  amrex::Real po = mask ? pl : pr;

  mask =
    amrex::Math::abs(ustar) < constants::smallu() * 0.5 *
                                (amrex::Math::abs(ul) + amrex::Math::abs(ur)) ||
    ustar == 0.0;
  ustar = mask ? 0.0 : ustar;
  ro = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspo[n] = mask ? 0.5 * (rl * spl[n] + rr * spr[n]) : rspo[n];
    ro += rspo[n];
  }
  uo = mask ? 0.5 * (ul + ur) : uo;
  po = mask ? 0.5 * (pl + pr) : po;

  amrex::Real gdnv_state_rho = ro;
  amrex::Real gdnv_state_p = po;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspo[n] / ro;
  }
  amrex::Real gdnv_state_e;
  eos.RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  amrex::Real co;
  eos.RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, co);

  const amrex::Real drho = (pstar - po) / (co * co);
  amrex::Real rstar = 0.0;
  amrex::Real rspstar[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    const amrex::Real spon = rspo[n] / ro;
    rspstar[n] = amrex::max<amrex::Real>(0.0, rspo[n] + drho * spon);
    rstar += rspstar[n];
  }
  gdnv_state_rho = rstar;
  gdnv_state_p = pstar;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspstar[n] / rstar;
  }
  eos.RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  amrex::Real cstar;
  eos.RPY2Cs(gdnv_state_rho, gdnv_state_p, gdnv_state_massfrac, cstar);

  const amrex::Real sgnm = amrex::Math::copysign(1.0, ustar);

  amrex::Real spout = co - sgnm * uo;
  amrex::Real spin = cstar - sgnm * ustar;
  const amrex::Real ushock = 0.5 * (spin + spout);

  mask = pstar < po;
  spout = mask ? spout : ushock;
  spin = mask ? spin : ushock;

  const amrex::Real scr =
    (spout == spin) ? constants::small_num() * cav : spout - spin;
  const amrex::Real frac = amrex::max<amrex::Real>(
    0.0, amrex::min<amrex::Real>(1.0, (1.0 + (spout + spin) / scr) * 0.5));

  mask = ustar > 0.0;
  qint_iv1 = mask ? vl : vr;
  qint_iv2 = mask ? v2l : v2r;

  mask = (ustar == 0.0);
  qint_iv1 = mask ? 0.5 * (vl + vr) : qint_iv1;
  qint_iv2 = mask ? 0.5 * (v2l + v2r) : qint_iv2;
  amrex::Real rgd = 0.0;

  amrex::Real rspgd[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = frac * rspstar[n] + (1.0 - frac) * rspo[n];
    rgd += rspgd[n];
  }
  qint_iu = frac * ustar + (1.0 - frac) * uo;
  qint_gdpres = frac * pstar + (1.0 - frac) * po;
  gdnv_state_rho = rgd;
  gdnv_state_p = qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  eos.RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);

  mask = (spout < 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspo[n] : rspgd[n];
    rgd += rspgd[n];
  }
  qint_iu = mask ? uo : qint_iu;
  qint_gdpres = mask ? po : qint_gdpres;

  mask = (spin >= 0.0);
  rgd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rspgd[n] = mask ? rspstar[n] : rspgd[n];
    rgd += rspgd[n];
  }
  qint_iu = mask ? ustar : qint_iu;
  qint_gdpres = mask ? pstar : qint_gdpres;

  gdnv_state_rho = rgd;
  gdnv_state_p = qint_gdpres;
  for (int n = 0; n < NUM_SPECIES; n++) {
    gdnv_state_massfrac[n] = rspgd[n] / rgd;
  }
  eos.RYP2E(gdnv_state_rho, gdnv_state_massfrac, gdnv_state_p, gdnv_state_e);
  amrex::Real regd = gdnv_state_rho * gdnv_state_e;

  qint_gdgame = qint_gdpres / regd + 1.0;
  qint_iu = bc_test_val * qint_iu;
  uflx_rho = rgd * qint_iu;
  for (int n = 0; n < NUM_SPECIES; n++) {
    uflx_rhoY[n] = rspgd[n] * qint_iu;
  }
  uflx_u = uflx_rho * qint_iu + qint_gdpres;
  uflx_v = uflx_rho * qint_iv1;
  uflx_w = uflx_rho * qint_iv2;
  const amrex::Real rhoetot =
    regd +
    0.5 * rgd * (qint_iu * qint_iu + qint_iv1 * qint_iv1 + qint_iv2 * qint_iv2);
  uflx_eden = qint_iu * (rhoetot + qint_gdpres);
  uflx_eint = qint_iu * regd;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
laxfriedrich_flux(
  const amrex::Real rl,
  const amrex::Real ul,
  const amrex::Real vl,
  const amrex::Real v2l,
  const amrex::Real pl,
  const amrex::Real spl[NUM_SPECIES],
  const amrex::Real rr,
  const amrex::Real ur,
  const amrex::Real vr,
  const amrex::Real v2r,
  const amrex::Real pr,
  const amrex::Real spr[NUM_SPECIES],
  const int /*unused*/,
  const amrex::Real /*unused*/,
  amrex::Real& ustar,
  amrex::Real& maxeigval,
  amrex::Real& uflx_rho,
  amrex::Real uflx_rhoY[NUM_SPECIES],
  amrex::Real& uflx_u,
  amrex::Real& uflx_v,
  amrex::Real& uflx_w,
  amrex::Real& uflx_eden,
  amrex::Real& uflx_eint)
{

  const amrex::Real wsmall = std::numeric_limits<amrex::Real>::min();

  auto eos = pele::physics::PhysicsType::eos();

  amrex::Real cl = 0.0;
  amrex::Real mfrac_l[NUM_SPECIES];
  amrex::Real mfrac_r[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    mfrac_l[n] = spl[n];
    mfrac_r[n] = spr[n];
  }
  eos.RPY2Cs(rl, pl, mfrac_l, cl);

  amrex::Real cr = 0.0;
  eos.RPY2Cs(rr, pr, mfrac_r, cr);

  amrex::Real el = 0.0;
  eos.RYP2E(rl, mfrac_l, pl, el);

  amrex::Real er = 0.0;
  eos.RYP2E(rr, mfrac_r, pr, er);

  // rho e total
  amrex::Real r_elt = rl * (el + 0.5 * (ul * ul + vl * vl + v2l * v2l));
  amrex::Real r_ert = rr * (er + 0.5 * (ur * ur + vr * vr + v2r * v2r));

  const amrex::Real wl = amrex::max<amrex::Real>(wsmall, cl * rl);
  const amrex::Real wr = amrex::max<amrex::Real>(wsmall, cr * rr);
  ustar = ((wl * ul + wr * ur) + (pl - pr)) / (wl + wr);

  bool mask =
    amrex::Math::abs(ustar) < constants::smallu() * 0.5 *
                                (amrex::Math::abs(ul) + amrex::Math::abs(ur)) ||
    ustar == 0.0;
  ustar = mask ? 0.0 : ustar;

  amrex::Real chalf = 0.5 * (cr + cl);
  amrex::Real max_wavespd = amrex::Math::abs(ustar) + chalf;

  maxeigval = max_wavespd;

  // density and mass frac eqn
  uflx_rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    uflx_rhoY[n] = 0.5 * ((rl * ul * spl[n] + rr * ur * spr[n]) -
                          max_wavespd * (rr * spr[n] - rl * spl[n]));
    uflx_rho += uflx_rhoY[n];
  }

  // three momentum eqns
  uflx_u = 0.5 * ((rl * ul * ul + pl + rr * ur * ur + pr) -
                  max_wavespd * (rr * ur - rl * ul));
  uflx_v =
    0.5 * ((rl * vl * ul + rr * vr * ur) - max_wavespd * (rr * vr - rl * vl));
  uflx_w = 0.5 * ((rl * v2l * ul + rr * v2r * ur) -
                  max_wavespd * (rr * v2r - rl * v2l));

  // total energy eqn
  uflx_eden = 0.5 * (((r_elt + pl) * ul + (r_ert + pr) * ur) -
                     max_wavespd * (r_ert - r_elt));

  // not really used
  uflx_eint =
    0.5 * ((rl * el * ul + rr * er * ur) - max_wavespd * (rr * er - rl * el));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
massflux(
  const amrex::Real rho,
  const amrex::Real cs,
  const amrex::Real pQ,
  const amrex::Real pm,
  const amrex::Real gamma)
{
  // Returns the mass flux used for the exact Riemann solver. Inspired from:
  // Katate Masatsuka, February 2009. http://www.cfdbooks.com
  const amrex::Real eps = 1e2*std::numeric_limits<amrex::Real>::epsilon();
  const amrex::Real gam1 = 0.5 * (gamma + 1) / gamma;
  const amrex::Real gam2 = 0.5 * (gamma - 1) / gamma;

  return (pm / pQ >= 1 - eps)
           ? rho * cs * std::sqrt(1 + gam1 * (pm / pQ - 1))
           : rho * cs * gam2 * (1 - pm / pQ) / (1 - std::pow(pm / pQ, gam2));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
sonic(
  const amrex::Real u1,
  const amrex::Real c1,
  const amrex::Real P1,
  const amrex::Real u2,
  const amrex::Real c2,
  const amrex::Real a1,
  const amrex::Real a2,
  const amrex::Real gamma,
  amrex::Real& US1,
  amrex::Real& US2,
  amrex::Real& US3,
  amrex::Real& US4)
{ // Returns solutions at sonic points --- used in Godunov's flux. Inspired
  // from: Katate Masatsuka, February 2009. http://www.cfdbooks.com

  const amrex::Real R1 = a2 / (a2 - a1);
  const amrex::Real R2 = -a1 / (a2 - a1);
  const amrex::Real us = R1 * u1 + R2 * u2;
  const amrex::Real cs = R1 * c1 + R2 * c2;
  const amrex::Real Ps = std::pow(cs / c1, (2.0 * gamma / (gamma - 1))) * P1;
  const amrex::Real rs = gamma * Ps / (cs * cs);

  US1 = rs;
  US2 = rs * us;
  US3 = Ps / (gamma - 1) + 0.5 * rs * us * us;
  US4 = Ps / (gamma - 1);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
riemann_exact(
  const amrex::Real rl,
  const amrex::Real ul,
  const amrex::Real vl,
  const amrex::Real v2l,
  const amrex::Real pl,
  const amrex::Real spl[NUM_SPECIES],
  const amrex::Real rr,
  const amrex::Real ur,
  const amrex::Real vr,
  const amrex::Real v2r,
  const amrex::Real pr,
  const amrex::Real spr[NUM_SPECIES],
  const int bc_test_val,
  const amrex::Real cav,
  amrex::Real& ustar,
  amrex::Real& uflx_rho,
  amrex::Real uflx_rhoY[NUM_SPECIES],
  amrex::Real& uflx_u,
  amrex::Real& uflx_v,
  amrex::Real& uflx_w,
  amrex::Real& uflx_eden,
  amrex::Real& uflx_eint,
  amrex::Real& qint_iu,
  amrex::Real& qint_iv1,
  amrex::Real& qint_iv2,
  amrex::Real& qint_gdpres,
  amrex::Real& qint_gdgame)
{
  // Exact Riemann solver. Inspired from: Katate Masatsuka, February 2009.
  // http://www.cfdbooks.com
  auto eos = pele::physics::PhysicsType::eos();

  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    NUM_SPECIES == 1, "Exact Riemann solver only works with single-species");

  amrex::Real ml[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    ml[n] = spl[n];
  }
  amrex::Real cl = 0.0, el = 0.0, Tl = 0.0, gammal = 0.0;
  eos.RYP2E(rl, ml, pl, el);
  eos.RPY2Cs(rl, pl, ml, cl);
  eos.REY2T(rl, el, ml, Tl);
  eos.RTY2G(rl, Tl, ml, gammal);
  const amrex::Real rel = rl * el;
  const amrex::Real rEl = rel + 0.5 * rl * (ul * ul + vl * vl + v2l * v2l);

  amrex::Real mr[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; n++) {
    mr[n] = spr[n];
  }
  amrex::Real cr = 0.0, er = 0.0, Tr = 0.0, gammar = 0.0;
  eos.RYP2E(rr, mr, pr, er);
  eos.RPY2Cs(rr, pr, mr, cr);
  eos.REY2T(rr, er, mr, Tr);
  eos.RTY2G(rr, Tr, mr, gammar);
  const amrex::Real rer = rr * er;
  const amrex::Real rEr = rer + 0.5 * rr * (ur * ur + vr * vr + v2r * v2r);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    amrex::Math::abs(gammar - gammal) <
      std::numeric_limits<amrex::Real>::epsilon(),
    "Exact Riemann solver only works with single gamma");
  const amrex::Real gamma = gammal;

  // Supersonic flow to the right
  if (ul / cl >= 1.0) {
    ustar = ul;
    qint_iu = ul;
    qint_iv1 = vl;
    qint_iv2 = v2l;

    uflx_rho = rl * ul;
    uflx_rhoY[0] = uflx_rho;
    uflx_u = uflx_rho * ul + pl;
    uflx_v = uflx_rho * vl;
    uflx_w = uflx_rho * v2l;
    uflx_eden = (rEl + pl) * ul;
    uflx_eint = rel * ul;
  }
  // Supersonic flow to the left
  else if (vr / cr <= -1) {
    ustar = ur;
    qint_iu = ur;
    qint_iv1 = vr;
    qint_iv2 = v2r;

    uflx_rho = rr * ur;
    uflx_rhoY[0] = uflx_rho;
    uflx_u = uflx_rho * ur + pr;
    uflx_v = uflx_rho * vr;
    uflx_w = uflx_rho * v2r;
    uflx_eden = (rEr + pr) * ur;
    uflx_eint = rer * ur;
  } else {
    // Initial solution: (intersection of two linearized integral curves, which
    // is actually the upper bound of the solution.)
    amrex::Real pm1 = std::pow(
      (0.5 * (ul - ur) * (gamma - 1) + cl + cr) /
        (cl * std::pow(pl, ((1 - gamma) / gamma * 0.5)) +
         cr * std::pow(pr, ((1 - gamma) / gamma * 0.5))),
      (2 * gamma / (gamma - 1)));
    amrex::Real pm2 = 0;
    // amrex::Print() << "L: " << rl << " "<< ul << " " << pl << " " << rEl << " " << el << " " << cl << " " << Tl << std::endl;
    // amrex::Print() << "R: " << rr << " "<< ur << " " << pr << " " << rEr << " " << er << " " << cr << " " << Tr << std::endl;

    // Fixed-point iteration to find the pressure and velocity in the middle
    // (i.e., find the intersection of two nonlinear integral curves.)
    const amrex::Real tol = 1e-6;

    const int kmax = 1000;
    for (int k = 0; k < kmax + 2; k++) {
      const amrex::Real mL = massflux(rl, cl, pl, pm1, gamma);
      const amrex::Real mR = massflux(rr, cr, pr, pm1, gamma);
      pm2 = (mL * pr + mR * pl - mL * mR * (ur - ul)) / (mL + mR);

      // Test for fixed point convergence
      if (amrex::Math::abs(pm2 - pm1) < tol) {
        break;
      }

      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        k < kmax,
        "Godunov fixed-point iteration did not converge. Exiting");

      // Set old value to new value
      pm1 = pm2;
    }

    //  Calculate the new fluxes
    const amrex::Real mL = massflux(rl, cl, pl, pm2, gamma);
    const amrex::Real mR = massflux(rr, cr, pr, pm2, gamma);
    const amrex::Real vm = (mL * ul + mR * ur - (pr - pl)) / (mL + mR);

    // Density in the middle
    const amrex::Real r[2] = {rl, rr};
    const amrex::Real P[2] = {pl, pr};
    const amrex::Real gam = (gamma + 1) / (gamma - 1);
    amrex::Real rm[2] = {0.0};
    for (int k = 0; k < 2; k++) {
      if ((pm2 / P[k]) >= 1) {
        rm[k] = r[k] * (1 + gam * pm2 / P[k]) / (gam + pm2 / P[k]);
      } else {
        rm[k] = r[k] * std::pow(pm2 / P[k], (1.0 / gamma));
      }
    }

    // Contact wave to the right or left?
    amrex::Real rmI = (vm >= 0) ? rm[0] : rm[1];

    // Wave speeds at the interface, x/t = 0
    const amrex::Real amL = std::sqrt(gamma * pm2 / rm[0]);
    const amrex::Real amR = std::sqrt(gamma * pm2 / rm[1]);
    const amrex::Real SmL = vm - amL;
    const amrex::Real SmR = vm + amR;

    // Sonic case
    amrex::Real Um2 = 0.0, Um3 = 0.0, Um4 = 0.0;
    if ((SmL <= 0) && (SmR >= 0)) {
      Um2 = rmI * vm;
      Um3 = pm2 / (gamma - 1) + 0.5 * rmI * vm * vm;
      Um4 = pm2 / (gamma - 1);
    } else if ((SmL > 0) && (ul - cl < 0)) {
      sonic(ul, cl, pl, vm, amL, ul - cl, SmL, gamma, rmI, Um2, Um3, Um4);
    } else if ((SmR < 0) && (ur + cr > 0)) {
      sonic(ur, cr, pr, vm, amR, ur + cr, SmR, gamma, rmI, Um2, Um3, Um4);
    }

    // Compute the flux: evaluate the physical flux at the interface (middle)
    const amrex::Real pm = (gamma - 1) * (Um3 - 0.5 * Um2 * Um2 / rmI);
    ustar = Um2 / rmI;
    qint_iu = Um2 / rmI;
    bool mask = ustar > 0.0;
    qint_iv1 = mask ? vl : vr;
    qint_iv2 = mask ? v2l : v2r;
    mask = (ustar == 0.0);
    qint_iv1 = mask ? 0.5 * (vl + vr) : qint_iv1;
    qint_iv2 = mask ? 0.5 * (v2l + v2r) : qint_iv2;
    qint_gdpres = pm;
    qint_gdgame = qint_gdpres / Um4 + 1.0;

    uflx_rho = Um2;
    uflx_rhoY[0] = uflx_rho;
    uflx_u = uflx_rho * qint_iu + pm;
    uflx_v = uflx_rho * qint_iv1;
    uflx_w = uflx_rho * qint_iv2;
    uflx_eden = (Um3 + pm) * qint_iu;
    uflx_eint = Um4 * qint_iu;
  }
}

#endif
