/**
 * work.h
 *
 * PAWSIM work arrays used by the first tderiv porting steps.
 *
 */
#ifndef _PAWSIM_WORK_H_
#define _PAWSIM_WORK_H_

#include "state.h"

#define PAWSIM_UMOM_NTERMS 16
#define PAWSIM_VMOM_NTERMS 16
#define PAWSIM_THIC_NTERMS 2
#define PAWSIM_ENERGY_NTERMS 15
#define PAWSIM_TRAC_NTERMS 6

enum
{
  PAWSIM_UMOM_Q = 0,
  PAWSIM_UMOM_GRADM,
  PAWSIM_UMOM_GRADKE,
  PAWSIM_UMOM_DHDT,
  PAWSIM_UMOM_A2,
  PAWSIM_UMOM_A4,
  PAWSIM_UMOM_RDRAG,
  PAWSIM_UMOM_RSURF,
  PAWSIM_UMOM_CDBOT,
  PAWSIM_UMOM_CDSURF,
  PAWSIM_UMOM_WIND,
  PAWSIM_UMOM_BUOY,
  PAWSIM_UMOM_RELAX,
  PAWSIM_UMOM_WDIA,
  PAWSIM_UMOM_FBARO,
  PAWSIM_UMOM_DIAVISC
};

enum
{
  PAWSIM_VMOM_Q = 0,
  PAWSIM_VMOM_GRADM,
  PAWSIM_VMOM_GRADKE,
  PAWSIM_VMOM_DHDT,
  PAWSIM_VMOM_A2,
  PAWSIM_VMOM_A4,
  PAWSIM_VMOM_RDRAG,
  PAWSIM_VMOM_RSURF,
  PAWSIM_VMOM_CDBOT,
  PAWSIM_VMOM_CDSURF,
  PAWSIM_VMOM_WIND,
  PAWSIM_VMOM_BUOY,
  PAWSIM_VMOM_RELAX,
  PAWSIM_VMOM_WDIA,
  PAWSIM_VMOM_FBARO,
  PAWSIM_VMOM_DIAVISC
};

enum
{
  PAWSIM_THIC_ADV = 0,
  PAWSIM_THIC_RELAX
};

enum
{
  PAWSIM_ENERGY_ADV = 0,
  PAWSIM_ENERGY_GRADM,
  PAWSIM_ENERGY_WIND,
  PAWSIM_ENERGY_RDRAG,
  PAWSIM_ENERGY_RSURF,
  PAWSIM_ENERGY_CDBOT,
  PAWSIM_ENERGY_CDSURF,
  PAWSIM_ENERGY_A2,
  PAWSIM_ENERGY_A4,
  PAWSIM_ENERGY_WDIAPE,
  PAWSIM_ENERGY_WDIAKE,
  PAWSIM_ENERGY_FBARO,
  PAWSIM_ENERGY_BUOY,
  PAWSIM_ENERGY_RELAX,
  PAWSIM_ENERGY_DIAVISC
};

enum
{
  PAWSIM_TRAC_ADV = 0,
  PAWSIM_TRAC_WDIA,
  PAWSIM_TRAC_K2,
  PAWSIM_TRAC_K4,
  PAWSIM_TRAC_DIADIFF,
  PAWSIM_TRAC_RELAX
};

typedef struct pawsim_work
{
  uint Nlay;
  /* Ghosted copies of the prognostic state used by stencil code. */
  pawsim_field2d * u_w;
  pawsim_field2d * v_w;
  pawsim_field2d * h_w;
  pawsim_field2d * b_w;
  /* Tendency outputs for the current derivative evaluation. */
  pawsim_field2d * dt_u;
  pawsim_field2d * dt_v;
  pawsim_field2d * dt_h;
  pawsim_field2d * dt_b;
  /* Layer face thicknesses and surface/bottom forcing fractions. */
  pawsim_field2d * h_west;
  pawsim_field2d * h_south;
  pawsim_field2d * hFsurf_west;
  pawsim_field2d * hFsurf_south;
  pawsim_field2d * hFbot_west;
  pawsim_field2d * hFbot_south;
  /* q-grid PV and momentum-scheme coefficient fields. */
  pawsim_field2d * hh_q;
  pawsim_field2d * zeta;
  pawsim_field2d * qq;
  pawsim_field2d * alpha;
  pawsim_field2d * beta;
  pawsim_field2d * gamma;
  pawsim_field2d * delta;
  pawsim_field2d * epsilon;
  pawsim_field2d * phi;
  pawsim_field2d * lambda;
  pawsim_field2d * mu;
  pawsim_field2d * pp;
  pawsim_field2d * KE_B;
  pawsim_field2d * MM_B;
  /* Interface heights from surface through bottom. */
  pawsim_field2d * eta_w;
  /* Diapycnal velocity at layer interfaces, plus u/v-face averages for momentum. */
  pawsim_field2d * wdia;
  pawsim_field2d * wdia_u;
  pawsim_field2d * wdia_v;
  /* Static geometry and rigid-lid column thickness on cell/face locations. */
  pawsim_field2d hhs_w;
  pawsim_field2d hhb_w;
  pawsim_field2d hhs_west;
  pawsim_field2d hhs_south;
  pawsim_field2d hhb_west;
  pawsim_field2d hhb_south;
  pawsim_field2d Hc;
  pawsim_field2d Hw;
  pawsim_field2d Hs;
  /* Interpolated forcing and surface/bottom drag speed work arrays. */
  pawsim_field2d taux_w;
  pawsim_field2d tauy_w;
  pawsim_field2d usq_surf;
  pawsim_field2d vsq_surf;
  pawsim_field2d uabs_surf;
  pawsim_field2d usq_bot;
  pawsim_field2d vsq_bot;
  pawsim_field2d uabs_bot;
  /* Downward diapycnal momentum fluxes at layer interfaces. */
  pawsim_field2d * Fdia_u;
  pawsim_field2d * Fdia_v;
  pawsim_field2d * Fdia_b;
  pawsim_field2d * gprime;
  /* Viscosity and reconstruction work arrays. */
  pawsim_field2d DD_T;
  pawsim_field2d DD_S;
  pawsim_field2d DD4_T;
  pawsim_field2d DD4_S;
  pawsim_field2d A2_h;
  pawsim_field2d A2_q;
  pawsim_field2d A4sqrt_h;
  pawsim_field2d A4sqrt_q;
  pawsim_field2d UU4;
  pawsim_field2d VV4;
  pawsim_field2d BB4;
  pawsim_field2d hz;
  pawsim_field2d d2hx;
  pawsim_field2d d2hy;
  pawsim_field2d d2bx;
  pawsim_field2d d2by;
  pawsim_field2d dh_dx;
  pawsim_field2d dh_dy;
  pawsim_field2d db_dx;
  pawsim_field2d db_dy;
  pawsim_field2d udb_dx;
  pawsim_field2d vdb_dy;
  pawsim_field2d hub;
  pawsim_field2d hvb;
  /* Layer mass fluxes h*u and h*v reused by momentum and thickness tendencies. */
  pawsim_field2d * huu;
  pawsim_field2d * hvv;
  /* AWSIM-compatible budget accumulators, allocated only for enabled diagnostics. */
  pawsim_field2d ** diag_umom;
  pawsim_field2d ** diag_vmom;
  pawsim_field2d ** diag_thic;
  pawsim_field2d ** diag_energy;
  pawsim_field2d ** diag_trac;
}
pawsim_work;

/* Allocate all derived work arrays and optional budget accumulators. */
bool pawsim_work_alloc (pawsim_work * work, const pawsim_config * cfg,
                        const pawsim_domain * dom);

/* Free all arrays owned by the work object. */
void pawsim_work_free (pawsim_work * work);

/* Zero work arrays before reuse or diagnostics initialization. */
void pawsim_work_zero (pawsim_work * work);

/* Build static surface/bottom geometry and pressure-operator thicknesses. */
void pawsim_work_init_static_geometry (pawsim_work * work, const pawsim_state * state,
                                       const pawsim_config * cfg,
                                       const pawsim_domain * dom);

/* Copy the current prognostic state into ghosted AWSIM-style work arrays. */
void pawsim_work_load_state (pawsim_work * work, const pawsim_state * state,
                             const pawsim_domain * dom);

/* Compute interface heights eta from layer thickness and bathymetry. */
void pawsim_work_calc_eta (pawsim_work * work, const pawsim_domain * dom);

/* Compute h on u/v faces using the selected AWSIM thickness scheme. */
void pawsim_work_calc_face_thickness (pawsim_work * work, const pawsim_config * cfg,
                                      const pawsim_domain * dom);

/* Compute face thicknesses from gathered no-ghost fields for AWSIM diagnostics. */
bool pawsim_work_calc_face_thickness_no_ghost (pawsim_work * work,
                                               const pawsim_config * cfg,
                                               const pawsim_domain * dom);

/* Compute q-grid vorticity, q-point thickness, and potential vorticity. */
void pawsim_work_calc_pv (pawsim_work * work, const pawsim_state * state,
                          const pawsim_domain * dom, real dx, real dy);

/* Build AL81/HK83/TW81 interpolation coefficients from q-grid PV. */
void pawsim_work_calc_pv_coefficients (pawsim_work * work, const pawsim_config * cfg,
                                       const pawsim_domain * dom);

/* Compute layer pressure, kinetic energy, and Montgomery/Bernoulli functions. */
void pawsim_work_calc_bernoulli (pawsim_work * work, const pawsim_state * state,
                                 const pawsim_config * cfg,
                                 const pawsim_domain * dom);

/* Compute horizontal momentum RHS terms and accumulate momentum/energy budgets. */
void pawsim_work_calc_momentum_tendency (pawsim_work * work,
                                         const pawsim_config * cfg,
                                         const pawsim_domain * dom,
                                         real dx, real dy);

/* Compute thickness RHS and thickness-related diagnostic budget terms. */
void pawsim_work_calc_thickness_tendency (pawsim_work * work,
                                          const pawsim_state * state,
                                          const pawsim_config * cfg,
                                          const pawsim_domain * dom, real dx, real dy);

/* Compute optional tracer RHS and layer-integrated tracer diagnostics. */
void pawsim_work_calc_tracer_tendency (pawsim_work * work, const pawsim_state * state,
                                       const pawsim_config * cfg,
                                       const pawsim_domain * dom, real dx, real dy);

/* Compute layer fractions participating in surface forcing/drag. */
void pawsim_work_calc_surface_forcing_thickness (pawsim_work * work,
                                                 const pawsim_config * cfg,
                                                 const pawsim_domain * dom);

/* Compute layer fractions participating in bottom drag. */
void pawsim_work_calc_bottom_forcing_thickness (pawsim_work * work,
                                                const pawsim_config * cfg,
                                                const pawsim_domain * dom);

#endif
