#include "PeleC_diffterm_3D.H" 
#include "PeleC_gradutil_3D.H"
#include "PeleC_diffusion.H"  
/*
    This file contains the driver for generating the diffusion fluxes, which are then used to generate the diffusion flux divergence.
    PeleC_compute_diffusion_flux utilizes functions from the PeleC_diffterm_2D header: 
    PeleC_move_transcoefs_to_ec -> Moves Cell Centered Transport Coefficients to Edge Centers 
    PeleC_compute_tangential_vel_derivs -> Computes the Tangential Velocity Derivatives 
    PeleC_diffusion_flux -> Computes the diffusion flux per direction with the coefficients and velocity derivatives. 
*/

void 
PeleC_compute_diffusion_flux(const Box& box, const amrex::Array4<const amrex::Real> &q,
                             const amrex::Array4<const amrex::Real> &coef, const amrex::Array4<amrex::Real> &flx1, 
                             const amrex::Array4<amrex::Real> &flx2, const amrex::Array4<amrex::Real> &flx3, 
                             const amrex::Array4<const amrex::Real> &a1, const amrex::Array4<const amrex::Real> &a2, 
                             const amrex::Array4<const amrex::Real> &a3, const int nCompTr, const amrex::Real del[], 
                             const int do_harmonic, const int diffuse_vel) 
{        
        Box exbox = amrex::surroundingNodes(box,0);
        Box eybox = amrex::surroundingNodes(box,1); 
        Box ezbox = amrex::surroundingNodes(box,2); 
        Gpu::AsyncFab cx_ec(exbox, nCompTr); 
        Gpu::AsyncFab cy_ec(eybox, nCompTr);
        Gpu::AsyncFab cz_ec(ezbox, nCompTr);  
        auto const &cx = cx_ec.array();
        auto const &cy = cy_ec.array(); 
        auto const &cz = cz_ec.array(); 
        const amrex::Real dx = del[0]; 
        const amrex::Real dy = del[1]; 
        const amrex::Real dz = del[2]; 

        // Get Edge-centered transport coefficients
        {
            BL_PROFILE("PeleC::pc_move_transport_coeffs_to_ec call");
            AMREX_PARALLEL_FOR_4D (box, nCompTr, i, j, k, n, { 
                   PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cx, 0, do_harmonic); 
                   PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cy, 1, do_harmonic);
                   PeleC_move_transcoefs_to_ec(i,j,k,n, coef, cz, 2, do_harmonic);  
            });       
        }
        int nCompTan = 4;
        Gpu::AsyncFab tx_der(exbox, nCompTan);
        Gpu::AsyncFab ty_der(eybox, nCompTan); 
        Gpu::AsyncFab tz_der(ezbox, nCompTan); 
        auto const &tx = tx_der.array(); 
        auto const &ty = ty_der.array(); 
        auto const &tz = tz_der.array(); 
        // Tangential derivatives on faces only needed for velocity diffusion
        if (diffuse_vel == 0) { // needs to be changed. 
          (tx_der.fab()).setVal(0);
          (ty_der.fab()).setVal(0);
          (tz_der.fab()).setVal(0); 
        } 
        else
        {
            {
                BL_PROFILE("PeleC::pc_compute_tangential_vel_derivs call");
        // Tangential derivs 
        // X 
                AMREX_PARALLEL_FOR_3D(exbox, i, j, k, {
                    PeleC_compute_tangential_vel_derivs(i,j,k,tx, q, 0, dy, dz); 
                });
        // Y
                AMREX_PARALLEL_FOR_3D(eybox, i, j, k, {
                    PeleC_compute_tangential_vel_derivs(i,j,k,ty, q, 1, dx, dz); 
                });
        //Z 
                AMREX_PARALLEL_FOR_3D(eybox, i, j, k, {
                    PeleC_compute_tangential_vel_derivs(i,j,k,tz, q, 2, dx, dy); 
                });
            }

        }  // diffuse_vel

        //Compute Extensive diffusion fluxes
        {
            BL_PROFILE("PeleC::diffusion_flux()"); 
            AMREX_PARALLEL_FOR_3D(exbox, i, j, k,  {
                PeleC_diffusion_flux(i,j,k, q, cx, tx, a1, flx1, dx, 0); 
            });
            AMREX_PARALLEL_FOR_3D(eybox, i, j, k,  {
                PeleC_diffusion_flux(i,j,k, q, cy, ty, a2, flx2, dy, 1);
            });
            AMREX_PARALLEL_FOR_3D(ezbox, i, j, k,  {
                PeleC_diffusion_flux(i,j,k, q, cz, tz, a3, flx3, dz, 2);
            });
        }
}
