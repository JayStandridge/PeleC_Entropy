#include "mechanism.H"

#include "PelePhysics.H"
#include "Derive.H"
#include "PeleC.H"
#include "IndexDefines.H"

#include<math.h>
#include<stdlib.h>

#include "PhysicsConstants.H"
#include "TransportParams.H"

void
pc_dervelx(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto velx = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    velx(i, j, k) = dat(i, j, k, UMX) / dat(i, j, k, URHO);
  });
}

void
pc_dervely(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vely = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    vely(i, j, k) = dat(i, j, k, UMY) / dat(i, j, k, URHO);
  });
}

void
pc_dervelz(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto velz = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    velz(i, j, k) = dat(i, j, k, UMZ) / dat(i, j, k, URHO);
  });
}

void
pc_dermagvel(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto magvel = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real datinv = 1.0 / dat(i, j, k, URHO);
    const amrex::Real dat1 = (dat(i, j, k, UMX) * datinv);
    const amrex::Real dat2 = (dat(i, j, k, UMY) * datinv);
    const amrex::Real dat3 = (dat(i, j, k, UMZ) * datinv);
    magvel(i, j, k) = sqrt((dat1 * dat1) + (dat2 * dat2) + (dat3 * dat3));
  });
}

void
pc_dermagmom(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto magmom = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    magmom(i, j, k) = sqrt(
      dat(i, j, k, UMX) * dat(i, j, k, UMX) +
      dat(i, j, k, UMY) * dat(i, j, k, UMY) +
      dat(i, j, k, UMZ) * dat(i, j, k, UMZ));
  });
}

void
pc_derkineng(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto kineng = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real datxsq = dat(i, j, k, UMX) * dat(i, j, k, UMX);
    const amrex::Real datysq = dat(i, j, k, UMY) * dat(i, j, k, UMY);
    const amrex::Real datzsq = dat(i, j, k, UMZ) * dat(i, j, k, UMZ);
    kineng(i, j, k) = 0.5 / dat(i, j, k, URHO) * (datxsq + datysq + datzsq);
  });
}

void
pc_dereint1(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // Compute internal energy from (rho E).
  auto const dat = datfab.const_array();
  auto e = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
    const amrex::Real ux = dat(i, j, k, UMX) * rhoInv;
    const amrex::Real uy = dat(i, j, k, UMY) * rhoInv;
    const amrex::Real uz = dat(i, j, k, UMZ) * rhoInv;
    e(i, j, k) =
      dat(i, j, k, UEDEN) * rhoInv - 0.5 * (ux * ux + uy * uy + uz * uz);
  });
}

void
pc_dereint2(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // Compute internal energy from (rho e).
  auto const dat = datfab.const_array();
  auto e = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    e(i, j, k) = dat(i, j, k, UEINT) / dat(i, j, k, URHO);
  });
}

void
pc_derlogden(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto logden = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    logden(i, j, k) = log10(dat(i, j, k));
  });
}

void
pc_derspec(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto spec = derfab.array();

  amrex::ParallelFor(
    bx, NUM_SPECIES, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
      spec(i, j, k, n) = dat(i, j, k, UFS + n) / dat(i, j, k, URHO);
    });
}

void
pc_deradv(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto adv = derfab.array();

  amrex::ParallelFor(
    bx, NUM_ADV, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
      adv(i, j, k, n) = dat(i, j, k, UFA + n) / dat(i, j, k, URHO);
    });
}

void
pc_dermagvort(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vort = derfab.array();

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
    AMREX_D_TERM(larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
                 , larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
                 , larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;)
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSize(0);
               , const amrex::Real dy = geomdata.CellSize(1);
               , const amrex::Real dz = geomdata.CellSize(2););

  // Calculate vorticity.
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

    // if fab is all regular -> call regular idx and weights
    // otherwise
    AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
    AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      vort(i, j, k) = 0.0 * dx;
      ,
      const amrex::Real vx = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx;
      const amrex::Real uy = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy;
      const amrex::Real v3 = vx - uy;
      ,
      const amrex::Real wx = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx;
      const amrex::Real wy = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy;
      const amrex::Real uz = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz;
      const amrex::Real vz = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz;
      const amrex::Real v1 = wy - vz; const amrex::Real v2 = uz - wx;);
    vort(i, j, k) = sqrt(AMREX_D_TERM(0., +v3 * v3, +v1 * v1 + v2 * v2));
  });
}

void
pc_derdivu(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto divu = derfab.array();

  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSize(0);
               , const amrex::Real dy = geomdata.CellSize(1);
               , const amrex::Real dz = geomdata.CellSize(2););

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)
    AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
    AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      const amrex::Real uhi = dat(ip, j, k, UMX) / dat(ip, j, k, URHO);
      const amrex::Real ulo = dat(im, j, k, UMX) / dat(im, j, k, URHO);
      , const amrex::Real vhi = dat(i, jp, k, UMY) / dat(i, jp, k, URHO);
      const amrex::Real vlo = dat(i, jm, k, UMY) / dat(i, jm, k, URHO);
      , const amrex::Real whi = dat(i, j, kp, UMZ) / dat(i, j, kp, URHO);
      const amrex::Real wlo = dat(i, j, km, UMZ) / dat(i, j, km, URHO););
    divu(i, j, k) = AMREX_D_TERM(
      wi * (uhi - ulo) / dx, +wj * (vhi - vlo) / dy, +wk * (whi - wlo) / dz);
  });
}

void
pc_derenstrophy(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  // This routine will derive enstrophy  = 1/2 rho (x_vorticity^2 +
  // y_vorticity^2 + z_vorticity^2)
  auto const dat = datfab.const_array();
  auto enstrophy = derfab.array();

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
    larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
    larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
    larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSize(0);
               , const amrex::Real dy = geomdata.CellSize(1);
               , const amrex::Real dz = geomdata.CellSize(2););

  // Calculate enstrophy.
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)
    AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
    AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      enstrophy(i, j, k) = 0.0 * dx;
      ,
      const amrex::Real vx = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx;
      const amrex::Real uy = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy;
      const amrex::Real v3 = vx - uy;
      ,
      const amrex::Real wx = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx;

      const amrex::Real wy = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy;

      const amrex::Real uz = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz;
      const amrex::Real vz = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz;

      const amrex::Real v1 = wy - vz; const amrex::Real v2 = uz - wx;);
    enstrophy(i, j, k) = 0.5 * dat(i, j, k, URHO) *
                         (AMREX_D_TERM(0., +v3 * v3, +v1 * v1 + v2 * v2));
  });
}

void
pc_dernull(
  const amrex::Box& /*bx*/,
  amrex::FArrayBox& /*derfab*/,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& /*datfab*/,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // This routine is used by particle_count.  Yes it does nothing.
}

void
pc_dermolefrac(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // Derive the mole fractions of the species
  auto const dat = datfab.const_array();
  auto spec = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real mass[NUM_SPECIES];
    amrex::Real mole[NUM_SPECIES];
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);

    for (int n = 0; n < NUM_SPECIES; n++) {
      mass[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    eos.Y2X(mass, mole);
    for (int n = 0; n < NUM_SPECIES; n++) {
      spec(i, j, k, n) = mole[n];
    }
  });
}

void
pc_dersoundspeed(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto cfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real massfrac[NUM_SPECIES];
    amrex::Real c;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2Cs(rho, T, massfrac, c);
    cfab(i, j, k) = c;
  });
}

void
pc_derentropy(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& /*datfab*/,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto sfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real s;
    auto eos = pele::physics::PhysicsType::eos();
    eos.S(s);
    sfab(i, j, k) = s;
  });
}

void
pc_dermachnumber(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto mach = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real massfrac[NUM_SPECIES];
    amrex::Real c;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2Cs(rho, T, massfrac, c);
    const amrex::Real datxsq = dat(i, j, k, UMX) * dat(i, j, k, UMX);
    const amrex::Real datysq = dat(i, j, k, UMY) * dat(i, j, k, UMY);
    const amrex::Real datzsq = dat(i, j, k, UMZ) * dat(i, j, k, UMZ);
    mach(i, j, k) = sqrt(datxsq + datysq + datzsq) / dat(i, j, k, URHO) / c;
  });
}

void
pc_derpres(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto pfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real p;
    amrex::Real massfrac[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2P(rho, T, massfrac, p);
    pfab(i, j, k) = p;
  });
}

void
pc_dertemp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto tfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    tfab(i, j, k) = dat(i, j, k, UTEMP);
  });
}

void
pc_derspectrac(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/,
  const int idx)
{
  auto const dat = datfab.const_array();
  auto spectrac = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    spectrac(i, j, k) = dat(i, j, k, UFS + idx) / dat(i, j, k, URHO);
  });
}

void
pc_derradialvel(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto rvel = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
    geomdata.ProbHiArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();
  AMREX_D_TERM(const amrex::Real centerx = 0.5 * (prob_lo[0] + prob_hi[0]);
               , const amrex::Real centery = 0.5 * (prob_lo[1] + prob_hi[1]);
               , const amrex::Real centerz = 0.5 * (prob_lo[2] + prob_hi[2]));

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0] - centerx;
      , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1] - centery;
      , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2] - centerz;)
    const amrex::Real r = sqrt(AMREX_D_TERM(x * x, +y * y, +z * z));
    rvel(i, j, k) = (AMREX_D_TERM(
                      dat(i, j, k, UMX) * x, +dat(i, j, k, UMY) * y,
                      +dat(i, j, k, UMZ) * z)) /
                    (dat(i, j, k, URHO) * r);
  });
}

void
pc_dercp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto cp_arr = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real mass[NUM_SPECIES];
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);

    for (int n = 0; n < NUM_SPECIES; n++) {
      mass[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real cp = 0.0;
    eos.RTY2Cp(dat(i, j, k, URHO), dat(i, j, k, UTEMP), mass, cp);
    cp_arr(i, j, k) = cp;
  });
}

void
pc_dercv(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto cv_arr = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real mass[NUM_SPECIES];
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);

    for (int n = 0; n < NUM_SPECIES; n++) {
      mass[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real cv = 0.0;
    eos.RTY2Cv(dat(i, j, k, URHO), dat(i, j, k, UTEMP), mass, cv);
    cv_arr(i, j, k) = cv;
  });
}

void
pc_dercoord(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& /*datfab*/,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto coord_arr = derfab.array();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(coord_arr(i, j, k, 0) = prob_lo[0] + (i + 0.5) * dx[0];
                 , coord_arr(i, j, k, 1) = prob_lo[1] + (j + 0.5) * dx[1];
                 , coord_arr(i, j, k, 2) = prob_lo[2] + (k + 0.5) * dx[2];);
  });
}

void
PeleC::pc_derviscosity(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto mu_arr = derfab.array();
  auto const* ltransparm = trans_parms.device_trans_parm();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real massfrac[NUM_SPECIES];
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);

    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto trans = pele::physics::PhysicsType::transport();
    amrex::Real mu = 0.0, dum1 = 0.0, dum2 = 0.0;
    const bool get_xi = false, get_mu = true, get_lam = false,
               get_Ddiag = false, get_chi = false;
    trans.transport(
      get_xi, get_mu, get_lam, get_Ddiag, get_chi, T, rho, massfrac, nullptr,
      nullptr, mu, dum1, dum2, ltransparm);
    mu_arr(i, j, k) = mu;
  });
}

void
PeleC::pc_derbulkviscosity(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto xi_arr = derfab.array();
  auto const* ltransparm = trans_parms.device_trans_parm();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real massfrac[NUM_SPECIES];
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);

    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto trans = pele::physics::PhysicsType::transport();
    amrex::Real xi = 0.0, dum1 = 0.0, dum2 = 0.0;
    const bool get_xi = true, get_mu = false, get_lam = false,
               get_Ddiag = false, get_chi = false;
    trans.transport(
      get_xi, get_mu, get_lam, get_Ddiag, get_chi, T, rho, massfrac, nullptr,
      nullptr, dum1, xi, dum2, ltransparm);
    xi_arr(i, j, k) = xi;
  });
}

void
PeleC::pc_derconductivity(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto lam_arr = derfab.array();
  auto const* ltransparm = trans_parms.device_trans_parm();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real massfrac[NUM_SPECIES];
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);

    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto trans = pele::physics::PhysicsType::transport();
    amrex::Real lam = 0.0, dum1 = 0.0, dum2 = 0.0;
    const bool get_xi = false, get_mu = false, get_lam = true,
               get_Ddiag = false, get_chi = false;
    trans.transport(
      get_xi, get_mu, get_lam, get_Ddiag, get_chi, T, rho, massfrac, nullptr,
      nullptr, dum1, dum2, lam, ltransparm);
    lam_arr(i, j, k) = lam;
  });
}

void
PeleC::pc_derdiffusivity(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto d_arr = derfab.array();
  auto const* ltransparm = trans_parms.device_trans_parm();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real massfrac[NUM_SPECIES];
    amrex::Real ddiag[NUM_SPECIES] = {0.0};
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);

    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    auto trans = pele::physics::PhysicsType::transport();
    amrex::Real dum1 = 0.0, dum2 = 0.0, dum3 = 0.0;
    const bool get_xi = false, get_mu = false, get_lam = false,
               get_Ddiag = true, get_chi = false;
    trans.transport(
      get_xi, get_mu, get_lam, get_Ddiag, get_chi, T, rho, massfrac, ddiag,
      nullptr, dum1, dum2, dum3, ltransparm);
    for (int n = 0; n < NUM_SPECIES; n++) {
      d_arr(i, j, k, n) = ddiag[n];
    }
  });
}

void
pc_vel_ders(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vel_ders = derfab.array();
#if false

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
    larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
    larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
    larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSize(0);
               , const amrex::Real dy = geomdata.CellSize(1);
               , const amrex::Real dz = geomdata.CellSize(2););

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			   AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;);
			   AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
					, get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
					, get_idx(k, 2, all_regular, flags(i, j, k), km, kp););
			   AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
					, const amrex::Real wj = get_weight(jm, jp);
					, const amrex::Real wk = get_weight(km, kp););
			   

			   vel_ders(i, j, k, 0) = wi * (larr(ip, j, k, 0) - larr(im, j, k, 0)) / dx; // dudx
			   vel_ders(i, j, k, 1) = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy; // dudy
			   vel_ders(i, j, k, 2) = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz; // dudz

			   vel_ders(i, j, k, 3) = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx; // dvdx
			   vel_ders(i, j, k, 4) = wj * (larr(i, jp, k, 1) - larr(i, jm, k, 1)) / dy; // dvdy
			   vel_ders(i, j, k, 5) = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz; // dvdz

			   vel_ders(i, j, k, 6) = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx; // dwdx
			   vel_ders(i, j, k, 7) = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy; // dwdy
			   vel_ders(i, j, k, 8) = wk * (larr(i, j, kp, 2) - larr(i, j, km, 2)) / dz; // dwdz
			 });
}

#endif
}


#if NUM_SPECIES > 1

void
PeleC::pc_entropyInequality(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto entropyInequality = derfab.array();
  // Declaring dat
  auto const dat = datfab.const_array();

  const int dim = amrex::SpaceDim;
  
  // Declaring local arrays
  amrex::FArrayBox local_mu(bx, 1, amrex::The_Async_Arena());
  auto mu_arr       = local_mu.array();
  amrex::FArrayBox local_divu(bx, 1, amrex::The_Async_Arena());
  auto divu         = local_divu.array();
  amrex::FArrayBox local_gxx(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto gradXx       = local_gxx.array();
  amrex::FArrayBox local_gxy(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto gradXy       = local_gxy.array();
 
  amrex::FArrayBox local_gyx(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto gradYx       = local_gyx.array();
  amrex::FArrayBox local_gyy(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto gradYy       = local_gyy.array();
  
  amrex::FArrayBox local_dti(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto Dti          = local_dti.array();

  amrex::FArrayBox local_gdkx(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto dkx          = local_gdkx.array();
  amrex::FArrayBox local_gdky(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto dky          = local_gdky.array();
  

  amrex::FArrayBox local_gvkx(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto vkx          = local_gvkx.array();
  amrex::FArrayBox local_gvky(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto vky          = local_gvky.array();
 

  amrex::FArrayBox local_gjkx(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto jkx          = local_gjkx.array();
  amrex::FArrayBox local_gjky(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto jky          = local_gjky.array();
   
  

  amrex::FArrayBox local_gt(bx, 3, amrex::The_Async_Arena());
  auto gradT        = local_gt.array();
  amrex::FArrayBox local_gp(bx, 3, amrex::The_Async_Arena());
  auto gradP        = local_gp.array();
  amrex::FArrayBox local_vd(bx, 9, amrex::The_Async_Arena());
  auto vel_ders     = local_vd.array();
  amrex::FArrayBox local_lam(bx, 1, amrex::The_Async_Arena());
  auto lam_arr      = local_lam.array();
  amrex::FArrayBox local_dd(bx, NUM_SPECIES, amrex::The_Async_Arena());
  auto d_arr        = local_dd.array();
  amrex::FArrayBox local_dij(bx, NUM_SPECIES*NUM_SPECIES, amrex::The_Async_Arena());
  auto Dij       = local_dij.array();

  amrex::FArrayBox local_dija(bx, NUM_SPECIES*NUM_SPECIES, amrex::The_Async_Arena());
  auto Dija       = local_dija.array();

  amrex::FArrayBox local_omega_d(bx,1, amrex::The_Async_Arena());
  auto omega_d       = local_omega_d.array();
  
  amrex::FArrayBox local_logt(bx, 3, amrex::The_Async_Arena());
  auto logt       = local_logt.array();

  amrex::FArrayBox local_ct(bx, 1, amrex::The_Async_Arena());
  auto c_tot        = local_ct.array();
  amrex::FArrayBox local_mmm(bx, 1, amrex::The_Async_Arena());
  auto mmm          = local_mmm.array();

  #if true
    amrex::FArrayBox local_gyz(bx, NUM_SPECIES, amrex::The_Async_Arena());
    auto gradYz       = local_gyz.array();
    
    amrex::FArrayBox local_gxz(bx, NUM_SPECIES, amrex::The_Async_Arena());
    auto gradXz       = local_gxz.array();
    
    amrex::FArrayBox local_gdkz(bx, NUM_SPECIES, amrex::The_Async_Arena());
    auto dkz          = local_gdkz.array();
    
    amrex::FArrayBox local_gvkz(bx, NUM_SPECIES, amrex::The_Async_Arena());
    auto vkz          = local_gvkz.array();
    
    amrex::FArrayBox local_gjkz(bx, NUM_SPECIES, amrex::The_Async_Arena());
    auto jkz          = local_gjkz.array();
  #endif

  // CONTROLS
  const bool do_soret             = false;
  const bool do_dufour            = false;
  const bool do_barodiffusion     = true;
  const bool do_enthalpy_diffusion = true;


  //Stole this from pc_derdiffusivity, will need mu, k, as well as ddiag later on
  auto const* ltransparm = trans_parms.device_trans_parm();


  //Get molecular weight/ inverse molecular weight prepared
  amrex::Real mw_arr[NUM_SPECIES]  = {0.0};
  amrex::Real imw_arr[NUM_SPECIES] = {0.0};
  amrex::Real eps[NUM_SPECIES]  = {0.0};
  amrex::Real sig[NUM_SPECIES] = {0.0};
  amrex::Real eps_ij[NUM_SPECIES][NUM_SPECIES]  = {0.0};
  amrex::Real sig_ij[NUM_SPECIES][NUM_SPECIES]  = {0.0};
  auto eos = pele::physics::PhysicsType::eos();
  eos.inv_molecular_weight(imw_arr);
  eos.molecular_weight(mw_arr);
  egtransetEPS(eps);
  egtransetSIG(sig);
  
  for(int i =0; i<NUM_SPECIES;i++){
    for(int j=0; j<NUM_SPECIES;j++){
      eps_ij[i][j]=pow(eps[i]*eps[j],.5);
      sig_ij[i][j]=.5*(sig[i]+sig[j]);
    }
  }
  

  // Get flow quantities that I do not need derivatives of
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			   amrex::Real massfrac[NUM_SPECIES] = {0};
			   amrex::Real molefrac[NUM_SPECIES] = {0};
			   const amrex::Real rho = dat(i, j, k, URHO);
			   const amrex::Real rhoInv = 1.0 / rho;
			   amrex::Real ddiag[NUM_SPECIES] = {0.0};
			   amrex::Real imw_local[NUM_SPECIES] = {0.0};
			   const amrex::Real T = dat(i, j, k, UTEMP);
			   c_tot(i,j,k)=0;
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
			     c_tot(i,j,k)+=dat(i,j,k,UFS+n)*imw_arr[n];
			   }
			   eos.Y2X(massfrac, molefrac);
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     mmm(i,j,k) += mw_arr[n]*molefrac[n];
			   }
			   auto trans = pele::physics::PhysicsType::transport();
			   amrex::Real mu = 0.0, lam = 0.0, dum1 = 0.0;
			   const bool get_xi = false, get_mu = true, get_lam = true,
			     get_Ddiag = true, get_chi = false;
			   trans.transport(
					   get_xi, get_mu, get_lam, get_Ddiag, get_chi, T, rho,
					   massfrac, ddiag, nullptr, mu, dum1,
					   lam, ltransparm);
			   mu_arr(i, j, k) = mu;
			   lam_arr(i, j, k) = lam;
			   amrex::Real temp1 = 0.0;
			   amrex::Real temp2 = 0.0;

			   for (int n = 0; n< NUM_SPECIES; n++){
			     temp1 += pow(mw_arr[n],.511) * molefrac[n];
			     temp2 += pow(mw_arr[n],.489) * molefrac[n];
			   }
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     d_arr(i, j, k, n) = ddiag[n];
			     if (do_soret){
			       Dti(i,j,k,n) = -2.59 * pow(10,-7) * pow(T,.659) * (mw_arr[n] * molefrac[n] / temp1 - massfrac[n]) * (temp1 / temp2);
			     } else {
			       Dti(i,j,k,n) = 0.0;
			     }
			   }
			   
			 });
  
  
  // Declaring gbx arrays that I will need for the gradient calculations later
  const amrex::Box& gbx = amrex::grow(bx, 1);
  amrex::FArrayBox glocal(gbx, 5, amrex::The_Async_Arena());
  auto larr = glocal.array();
  amrex::FArrayBox glocalx(gbx, NUM_SPECIES, amrex::The_Async_Arena());
  auto specX= glocalx.array();
  amrex::FArrayBox glocaly(gbx, NUM_SPECIES, amrex::The_Async_Arena());
  auto specY= glocaly.array();

  // Stolen from @nickwimer 
  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;


  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			    amrex::Real mass[NUM_SPECIES];
			    amrex::Real mole[NUM_SPECIES];
			    amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);


			    // Section added to get mass/mole fraction gradients
			    for (int n = 0; n < NUM_SPECIES; n++) {
			      mass[n] = dat(i, j, k, UFS + n) * rhoInv;
			    }
			    eos.Y2X(mass, mole);
	 
			    for (int n = 0; n < NUM_SPECIES; n++) {
			      specX(i, j, k, n) = mole[n];
			      specY(i, j, k, n) = dat(i, j, k, UFS + n) * rhoInv;
			    }
			    larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
			    larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
			    if (dim > 2)
			      larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;

			    // Temperature also added
			    larr(i, j, k, 3) = dat(i, j, k, UTEMP);
			    amrex::Real pressure;
			    eos.RTY2P(dat(i,j,k,URHO),larr(i,j,k,3), mass, pressure);
			    larr(i, j, k, 4) = pressure;
			    
			  });
  
  // As in @nickwimer branch
  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSize(0);
               , const amrex::Real dy = geomdata.CellSize(1);
               , const amrex::Real dz = geomdata.CellSize(2););
  
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
  			   AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;);
  			   AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
  			   		, get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
  			   		, get_idx(k, 2, all_regular, flags(i, j, k), km, kp););
  			   AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
  			   		, const amrex::Real wj = get_weight(jm, jp);
  			   		, const amrex::Real wk = get_weight(km, kp););
			   const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);

  			   vel_ders(i, j, k, 0) = wi * (larr(ip, j, k, 0) - larr(im, j, k, 0)) / dx; // dudx
  			   vel_ders(i, j, k, 1) = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy; // dudy

#if dim > 2 
			   vel_ders(i, j, k, 2) = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz; // dudz
#endif
  			   vel_ders(i, j, k, 3) = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx; // dvdx
  			   vel_ders(i, j, k, 4) = wj * (larr(i, jp, k, 1) - larr(i, jm, k, 1)) / dy; // dvdy
#if dim > 2 
			   vel_ders(i, j, k, 5) = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz; // dvdz
#endif
			   vel_ders(i, j, k, 6) = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx; // dwdx
  			   vel_ders(i, j, k, 7) = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy; // dwdy
#if dim > 2 
			     vel_ders(i, j, k, 8) = wk * (larr(i, j, kp, 2) - larr(i, j, km, 2)) / dz; // dwdz
#endif
			   // My code

			   gradT(i, j, k, 0) = wi * (larr(ip, j, k, 3) - larr(im, j, k, 3)) / dx; // dTdx
  			   gradT(i, j, k, 1) = wj * (larr(i, jp, k, 3) - larr(i, jm, k, 3)) / dy; // dTdy
#if dim > 2 
			     gradT(i, j, k, 2) = wk * (larr(i, j, kp, 3) - larr(i, j, km, 3)) / dz; // dTdz
#endif
			   gradP(i, j, k, 0) = wi * (larr(ip, j, k, 4) - larr(im, j, k, 4)) / dx; // dTdx
  			   gradP(i, j, k, 1) = wj * (larr(i, jp, k, 4) - larr(i, jm, k, 4)) / dy; // dTdy
#if dim > 2
			     gradP(i, j, k, 2) = wk * (larr(i, j, kp, 4) - larr(i, j, km, 4)) / dz; // dTdz
#endif
			   for (int n = 0; n < NUM_SPECIES; n++) {
			      gradXx(i, j, k, n) = wi * (specX(ip, j, k, n) - specX(im, j, k, n)) / dx; // dXdx
			      gradXy(i, j, k, n) = wj * (specX(i, jp, k, n) - specX(i, jm, k, n)) / dy; // dXdy
#if dim > 2
				gradXz(i, j, k, n) = wk * (specX(i, j, kp, n) - specX(i, j, km, n)) / dz; // dXdz
#endif
			   }
			   for (int n = 0; n < NUM_SPECIES; n++) {
			      gradYx(i, j, k, n) = wi * (specY(ip, j, k, n) - specY(im, j, k, n)) / dx; // dYdx
			      gradYy(i, j, k, n) = wj * (specY(i, jp, k, n) - specY(i, jm, k, n)) / dy; // dYdy
#if dim > 2
				gradYz(i, j, k, n) = wk * (specY(i, j, kp, n) - specY(i, j, km, n)) / dz; // dYdz
#endif
			   }
			   

			   divu(i,j,k,0) = vel_ders(i, j, k, 0) + vel_ders(i, j, k, 4);
#if dim >2
			   divu(i,j,k,0) += vel_ders(i, j, k, 8);
#endif
			   // First term of differential entropy inequality
			   //EIterm1
#if  dim == 2
			   entropyInequality(i,j,k,0)=(2/3*pow(divu(i,j,k),2)
						       -2*( pow(vel_ders(i, j, k, 0),2)
							    + pow(vel_ders(i, j, k, 4),2))
						       -(pow(vel_ders(i, j, k, 3),2)
							 + pow( vel_ders(i, j, k, 6),2)
							 + pow(vel_ders(i, j, k, 7),2)));
#endif
#if  dim > 2
			   entropyInequality(i,j,k,0)=(2/3*pow(divu(i,j,k),2)
						       -2*( pow(vel_ders(i, j, k, 0),2)
							    + pow(vel_ders(i, j, k, 4),2)
							    +pow( vel_ders(i, j, k, 8),2))
						       -(pow(vel_ders(i, j, k, 2) + vel_ders(i, j, k, 3),2)
							 + pow(vel_ders(i, j, k, 2) + vel_ders(i, j, k, 6),2)
							 + pow(vel_ders(i, j, k, 7) + vel_ders(i, j, k, 5),2)));
#endif       					 
			       //EITerm2
			       //using AUX1, AUX2, and AUX3 to store the energy flux vector
			   entropyInequality(i,j,k,5)=0; //AUX1
			   entropyInequality(i,j,k,6)=0; //AUX2
			   entropyInequality(i,j,k,7)=0; //AUX3
			   float sden;
			   amrex::Real barodiffusion = 0.0;
			   if (do_barodiffusion) {
			     barodiffusion = 1.0;
			   }
			   // Diffusion driving force
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     dkx(i,j,k,n) = gradXx(i,j,k,n) + barodiffusion * (specX(i,j,k,n) - specY(i,j,k,n))*gradP(i,j,k,0)/larr(i,j,k,4);
			     dky(i,j,k,n) = gradXy(i,j,k,n) + barodiffusion * (specX(i,j,k,n) - specY(i,j,k,n))*gradP(i,j,k,1)/larr(i,j,k,4);
#if dim >2
			     dkz(i,j,k,n) = gradXz(i,j,k,n) + barodiffusion * (specX(i,j,k,n) - specY(i,j,k,n))*gradP(i,j,k,2)/larr(i,j,k,4);
#endif
			     }
			   // Diffusion Velocity
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     sden=std::max(specX(i,j,k,n),pow(10,-8));
			     vkx(i,j,k,n) = -d_arr(i,j,k,n) * dkx(i,j,k,n) / sden;
			     vky(i,j,k,n) = -d_arr(i,j,k,n) * dky(i,j,k,n) / sden;
# if dim > 2 
			     vkz(i,j,k,n) = -d_arr(i,j,k,n) * dkz(i,j,k,n) / sden;
#endif
			   }
			   if (do_soret) {
			     for (int n = 0; n < NUM_SPECIES; n++) {
			       sden=std::max(specY(i,j,k,n),pow(10,-8));
			       vkx(i,j,k,n) -= Dti(i,j,k,n) * gradT(i,j,k,0) / ( dat(i,j,k,URHO) * sden * larr(i,j,k,3));
			       vky(i,j,k,n) -= Dti(i,j,k,n) * gradT(i,j,k,1) / ( dat(i,j,k,URHO) * sden * larr(i,j,k,3));
#if dim > 2
			       vkz(i,j,k,n) -= Dti(i,j,k,n) * gradT(i,j,k,2) / ( dat(i,j,k,URHO) * sden * larr(i,j,k,3));
#endif
			     }
			   }
			   // multicomponent species flux
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     jkx(i,j,k,n) = dat(i,j,k,URHO) * specY(i,j,k,n) * vkx(i,j,k,n);
			     jky(i,j,k,n) = dat(i,j,k,URHO) * specY(i,j,k,n) * vky(i,j,k,n);
#if dim > 2 
			     jkz(i,j,k,n) = dat(i,j,k,URHO) * specY(i,j,k,n) * vkz(i,j,k,n);
#endif
			   }
			   
			   
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     sden=std::max(dat(i,j,k,UFS+n),pow(10,-8)); // No dividing by zero
			     entropyInequality(i,j,k,5)+= dkx(i,j,k,n) * Dti(i,j,k,n)/sden;
			     entropyInequality(i,j,k,6)+= dky(i,j,k,n) * Dti(i,j,k,n)/sden;
#if dim > 2
			     entropyInequality(i,j,k,7)+= dkz(i,j,k,n) * Dti(i,j,k,n)/sden;
#endif
			   }
			   entropyInequality(i,j,k,5)*=-c_tot(i,j,k)*pele::physics::Constants::RU*larr(i,j,k,3);
			   entropyInequality(i,j,k,6)*=-c_tot(i,j,k)*pele::physics::Constants::RU*larr(i,j,k,3);
#if dim > 2 
			   entropyInequality(i,j,k,7)*=-c_tot(i,j,k)*pele::physics::Constants::RU*larr(i,j,k,3);
#endif
			   
			   
			   if ( do_enthalpy_diffusion ){
			     amrex::Real h_specific[NUM_SPECIES] = {0.0};
			     amrex::Real y_temp[NUM_SPECIES] = {0.0};
			     for (int n = 0; n< NUM_SPECIES; n++) {
			       for (int m = 0; m < NUM_SPECIES; m++) {
				 y_temp[m] = 0.0;
			       }
			       y_temp[n] = 1.0;
			       CKHBMS(larr(i,j,k,3),y_temp,h_specific[n]); //is in ergs/g
			       h_specific[n] *= pow(10,-7); //ergs/g -> j/g
			       entropyInequality(i,j,k,5)+=h_specific[n]*jkx(i,j,k,n);
			       entropyInequality(i,j,k,6)+=h_specific[n]*jky(i,j,k,n);
#if dim > 2 
			       entropyInequality(i,j,k,7)+=h_specific[n]*jkz(i,j,k,n);
#endif
			     }
			   }
			   
			   entropyInequality(i,j,k,5)-=lam_arr(i,j,k)*gradT(i,j,k,0);
			   entropyInequality(i,j,k,6)-=lam_arr(i,j,k)*gradT(i,j,k,1);
#if dim > 2
			   entropyInequality(i,j,k,7)-=lam_arr(i,j,k)*gradT(i,j,k,2);
#endif
#if dim == 2 
			   entropyInequality(i,j,k,1)=(entropyInequality(i,j,k,5)*gradT(i,j,k,0)+
						       entropyInequality(i,j,k,6)*gradT(i,j,k,1))/larr(i,j,k,3);
#endif
#if dim > 2
			   entropyInequality(i,j,k,1)=(entropyInequality(i,j,k,5)*gradT(i,j,k,0)+
						       entropyInequality(i,j,k,6)*gradT(i,j,k,1)
						       +entropyInequality(i,j,k,7)*gradT(i,j,k,2))/larr(i,j,k,3);
#endif
			   // EITerm3
			   // Start by getting the Dij matrix:
			   logt(i,j,k,0) = std::log(larr(i,j,k,3));
			   logt(i,j,k,1) = logt(i,j,k,0) * logt(i,j,k,0);
			   logt(i,j,k,2) = logt(i,j,k,0) * logt(i,j,k,1);
			   const amrex::Real scale = pele::physics::Constants::PATM / (pele::physics::Constants::RU * larr(i,j,k,3));
			   
			   for (int ii = 0; ii < NUM_SPECIES; ++ii)
			     {
			       for (int jj = 0; jj < NUM_SPECIES; ++jj)
				 {
				   const int four_idx_ij = 4 * (ii + NUM_SPECIES * jj);
				   const amrex::Real dbintemp =
				 ltransparm->fitdbin[four_idx_ij] +
				     ltransparm->fitdbin[1 + four_idx_ij] * logt(i,j,k,0) +
				     ltransparm->fitdbin[2 + four_idx_ij] * logt(i,j,k,1) +
				     ltransparm->fitdbin[3 + four_idx_ij] * logt(i,j,k,2);
				   Dij(i,j,k,ii+NUM_SPECIES*jj)=ltransparm->wt[ii] * std::exp(dbintemp) * scale;
				 }
			     }
			   // Verify Dij matrix with an analytical expression from Fluent:
			   amrex::Real massfrac[NUM_SPECIES];
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
			   }
			   amrex::Real pabs;
			   eos.RTY2P(dat(i,j,k,URHO),larr(i,j,k,3), massfrac, pabs);
			   for (int ii = 0; ii < NUM_SPECIES; ++ii)
			     {
			       for (int jj = 0; jj < NUM_SPECIES; ++jj)
				 {
				   amrex::Real t_star=larr(i,j,k,3)*pow(eps_ij[ii][jj],-1);
				   omega_d(i,j,k)=1.16145*pow(t_star,-.14874) +
				     .52487*std::exp(-.7732*t_star) +
				     2.16178*std::exp(-2.43787*t_star);
				   Dija(i,j,k,ii+NUM_SPECIES*jj) = 0.0188*pow( pow(larr(i,j,k,3),3) * (imw_arr[ii] + imw_arr[jj])  , .5) /
				     (pow(sig_ij[ii][jj],2)*omega_d(i,j,k) * pabs);
				   
				 }
			     }
			   // Now that we are done with the energy flux vector, we will construct the
			   // multicomponent species flux. Starting with the implementation from Kee and Warnatz (1986)
			   
			   entropyInequality(i,j,k,2) = 0;
			   
			   for (int n = 0; n < NUM_SPECIES-1; n++) {
			     sden=std::max(dat(i,j,k,UFS+n),pow(10,-8)); // No dividing by zero
			     entropyInequality(i,j,k,2) += jkx(i,j,k,n)*(dkx(i,j,k,n) / sden - dkx(i,j,k,NUM_SPECIES-1) / dat(i,j,k,UFS+NUM_SPECIES-1));
			     entropyInequality(i,j,k,2) += jky(i,j,k,n)*(dky(i,j,k,n) / sden - dky(i,j,k,NUM_SPECIES-1) / dat(i,j,k,UFS+NUM_SPECIES-1));
			     if(dim > 2)
			       entropyInequality(i,j,k,2) += jkz(i,j,k,n)*(dkz(i,j,k,n) / sden - dkz(i,j,k,NUM_SPECIES-1) / dat(i,j,k,UFS+NUM_SPECIES-1));
			   }
			   
			   entropyInequality(i,j,k,2) *= c_tot(i,j,k) * pele::physics::Constants::RU * larr(i,j,k,3);
			   
			   // EITERM4
			   amrex::Real tc[5] = {0.0};
			   amrex::Real gibbs_fe[NUM_SPECIES] = {0.0};
         amrex::Real gibbs_standard[NUM_SPECIES] = {0.0};
			   amrex::Real prod_rate[NUM_SPECIES] = {0.0};
			   amrex::Real prod_rate2[NUM_SPECIES] = {0.0};
			   
			   amrex::Real rho = dat(i, j, k, URHO);
			   
			   tc[1] = larr(i,j,k,3);
			   tc[0] = std::log(tc[1]);
			   tc[2] = pow(tc[1],2);
			   tc[3] = pow(tc[1],3);
			   tc[4] = pow(tc[1],4);
			   gibbs(gibbs_fe,tc);
			   
			   // Get rate of production of each species
			   amrex::Real sc[NUM_SPECIES] = {0.0};
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
			   }
			   CKYTCR(rho,rho,massfrac, sc); //Need species concentration first
			     
			   // Get chemical potential, or molar gibbs
			   for (int n = 0; n < NUM_SPECIES; n++) {
           gibbs_standard[n] = 8.314 * larr(i,j,k,3) * gibbs_fe[n];
			     gibbs_fe[n] += std::log(std::max(1e6*sc[n]*8.31446 * larr(i,j,k,3)/101325,1e-200));
			     gibbs_fe[n] *= 8.31446 * larr(i,j,k,3);// * pow(10,-7); keep in CGS
			   }
			   
			   for (int i = 0; i < NUM_SPECIES; i++) { 
			     sc[i] *= 1e6; // in SI units for productionRate

			   }

			   productionRate(prod_rate, sc, tc[1]);
         CKWYR(dat(i,j,k,URHO), larr(i,j,k,3), massfrac, prod_rate2);

			   //Calculate fourth term
			   entropyInequality(i,j,k,3) = 0.0;
         entropyInequality(i,j,k,5) = 0.0;
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     prod_rate[n] *= 1e-6;
			     entropyInequality(i,j,k,3) += prod_rate[n] * gibbs_fe[n];
           entropyInequality(i,j,k,5) += prod_rate[n] * gibbs_fe[n];
			   }

			   // Sum of all terms
			   
			   entropyInequality(i,j,k,4) = entropyInequality(i,j,k,0) +
			                                entropyInequality(i,j,k,1) +
			                                entropyInequality(i,j,k,2) +
			                                entropyInequality(i,j,k,3);
			   
			   //Check each species portion of fourth term
			   for (int n = 0; n < NUM_SPECIES; n++) {
			     entropyInequality(i,j,k,9+n)= prod_rate[n] *gibbs_fe[n];

			     
			   }
			   amrex::Real q_f_temp[NUM_REACTIONS] = {0.0};
			   amrex::Real q_r_temp[NUM_REACTIONS] = {0.0};
			   amrex::Real q_f[NUM_REACTIONS] = {0.0};
			   amrex::Real q_r[NUM_REACTIONS] = {0.0};
			   amrex::Real prod_rate_2[NUM_SPECIES] = {0.0};

			   amrex::Real wdot[NUM_REACTIONS] = {0.0};
			   amrex::Real nu_spec = 0.0;
			   int nspec = 0;
			   int& nspecr = nspec;
			   int* temp;
			   int* temp2;
			   CKINU(0, nspecr, temp, temp2);
			   int ki[nspec] = {0};
			   int nu[nspec] = {0};
			   int* kip = ki;
			   int* nup = nu;
			   int rmap[21] = {0};
			   int* rmapp = rmap;

			   CKYTCR(rho,rho,massfrac, sc); //Need species concentration first
			     
			   for (int i = 0; i < NUM_SPECIES; i++) { 
			     sc[i] *= 1e6; // in SI units for productionRate

			   }
			   
			   GET_RMAP(rmapp);

			   progressRateFR(q_f_temp, q_r_temp, sc, tc[1]);
			   for (int n = 0; n < NUM_REACTIONS; n++) { 
			     q_f[rmap[n]] = q_f_temp[n];
			     q_r[rmap[n]] = q_r_temp[n];
           entropyInequality(i,j,k,9+NUM_SPECIES+n) = 0;
			   }
        double DG_j[NUM_REACTIONS] = {0.0};
        double DG_j_s[NUM_REACTIONS] = {0.0};
        double EI_j[NUM_REACTIONS] = {0.0};
			   for (int n = 0; n < NUM_REACTIONS; n++) {
			     CKINU(n+1, nspecr, kip, nup);
			     wdot[n] = 1e-6 * (q_f[n] - q_r[n]);
			     for (int m = 0; m < nspec; m++){
			      //  entropyInequality(i,j,k,9+NUM_SPECIES+n) +=  nu[m] * gibbs_fe[ki[m]-1];
			       entropyInequality(i,j,k,9+NUM_SPECIES+n) += wdot[n]* nu[m] * gibbs_fe[ki[m]-1];
			       DG_j[n] += nu[m] * gibbs_fe[ki[m]-1];
             DG_j_s[n] += nu[m] * gibbs_standard[ki[m]-1];
			       EI_j[n] += wdot[n] * nu[m] * gibbs_fe[ki[m]-1];
             entropyInequality(i,j,k,5) -= wdot[n] * nu[m] * gibbs_fe[ki[m]-1];
			      }
         }
            if((i==0)&&(j==0)){
              // Open the file in append mode
              std::ofstream outputFile("output2.txt", std::ios_base::app);

              // Check if the file is open
              if (!outputFile.is_open()) {
                std::cerr << "Error opening the file." << std::endl;
              }

              // Append the values to the file
              int jj = {15};
                outputFile << "DGs : " << DG_j_s[jj]/(8.31446*larr(i,j,k,3)) << ", ";
                outputFile << "Dgs are " <<gibbs_fe[5] << " and " << gibbs_fe[7] <<", ";
                outputFile << "SCs are " <<sc[5] << " and " << sc[7] <<", ";
                outputFile << "omega " <<jj<<": " << wdot[jj] << ", ";
                outputFile << "DG " <<jj<<": " << DG_j[jj] << ", ";
                outputFile << "EI " <<jj<<": " << EI_j[jj] << ", ";
              
              outputFile << "EI Term 4: " << entropyInequality(0,0,0,3) << "\n";
              // Close the file
              outputFile.close();
            }      
			   });
          

}
#else
void
PeleC::pc_entropyInequality(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto entropyInequality = derfab.array();
}
#endif






























#ifdef PELEC_USE_MASA
void
pc_derrhommserror(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto rhommserror = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

    const amrex::Real rho = masa_eval_3d_exact_rho(x, y, z);
    rhommserror(i, j, k) = dat(i, j, k, URHO) - rho;
  });
}

void
pc_derummserror(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto ummserror = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  // const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
  // geomdata.ProbHiArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

    const amrex::Real u = masa_eval_3d_exact_u(x, y, z);
    ummserror(i, j, k) = dat(i, j, k, UMX) / dat(i, j, k, URHO) - u;
  });
}

void
pc_dervmmserror(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vmmserror = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  // const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
  // geomdata.ProbHiArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

    const amrex::Real v = masa_eval_3d_exact_v(x, y, z);
    vmmserror(i, j, k) = dat(i, j, k, UMY) / dat(i, j, k, URHO) - v;
  });
}

void
pc_derwmmserror(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto wmmserror = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  // const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
  // geomdata.ProbHiArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

    const amrex::Real w = masa_eval_3d_exact_w(x, y, z);
    wmmserror(i, j, k) = dat(i, j, k, UMZ) / dat(i, j, k, URHO) - w;
  });
}

void
pc_derpmmserror(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto pmmserror = derfab.array();

  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
    geomdata.ProbLoArray();
  // const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi =
  // geomdata.ProbHiArray();
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    geomdata.CellSizeArray();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

    auto eos = pele::physics::PhysicsType::eos();
    const amrex::Real rho = dat(i, j, k, URHO);
    const amrex::Real rhoinv = 1.0 / rho;
    amrex::Real eint = dat(i, j, k, UEINT) * rhoinv;
    amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoinv;
    }

    amrex::Real pdat;
    eos.RYET2P(rho, massfrac, eint, T, pdat);
    const amrex::Real p = masa_eval_3d_exact_p(x, y, z);
    pmmserror(i, j, k) = pdat - p;
  });
}
#endif
