/**
 * state.h
 *
 * PAWSIM model-state initialization interface.
 *
 */
#ifndef _PAWSIM_STATE_H_
#define _PAWSIM_STATE_H_

#include "config.h"
#include "field.h"

typedef struct pawsim_state
{
  uint Nlay;
  /* Prognostic layer fields on the C grid. */
  pawsim_field2d * u;
  pawsim_field2d * v;
  pawsim_field2d * h;
  pawsim_field2d * b;
  /* Static bathymetry/surface geometry and q-grid planetary vorticity. */
  pawsim_field2d hhb;
  pawsim_field2d hhs;
  pawsim_field2d omega_z;
  /* Optional wind records, interpolated in forcing.c. */
  pawsim_field2d * taux;
  pawsim_field2d * tauy;
  /* Static lid velocities and barotropic accelerations. */
  pawsim_field2d uLid;
  pawsim_field2d vLid;
  pawsim_field2d Fbaro_x;
  pawsim_field2d Fbaro_y;
  /* Newtonian restoring targets and time scales. Times <= 0 disable restoring. */
  pawsim_field2d * uRelax;
  pawsim_field2d * vRelax;
  pawsim_field2d * hRelax;
  pawsim_field2d * eRelax;
  pawsim_field2d * bRelax;
  pawsim_field2d * uTime;
  pawsim_field2d * vTime;
  pawsim_field2d * eTime;
  pawsim_field2d * bTime;
  pawsim_field2d hTime;
  /* Prescribed diapycnal velocity records, tracer diffusivity, and momentum viscosities. */
  pawsim_field2d * wdia_ff;
  pawsim_field2d * kappa_dia;
  pawsim_field2d Fsurf_b;
  pawsim_field2d * nu_u_dia;
  pawsim_field2d * nu_v_dia;
  uint tauNrecs;
  uint wDiaNrecs;
  /* Layer reduced gravities and their cumulative effective values. */
  real * gg;
  real * geff;
}
pawsim_state;

/* Allocate prognostic, static, forcing, and optional tracer/diapycnal fields. */
bool pawsim_state_alloc (pawsim_state * state, const pawsim_config * cfg,
                         const pawsim_domain * dom);

/* Free all arrays owned by a state object. */
void pawsim_state_free (pawsim_state * state);

/* Zero only U/V/H/B prognostic fields, leaving static inputs intact. */
void pawsim_state_zero_prognostic (pawsim_state * state);

/* Copy owned prognostic fields between states with the same local layout. */
void pawsim_state_copy_prognostic (pawsim_state * dst, const pawsim_state * src);

/* Update prognostic halos after time stepping or tendency-history shifts. */
void pawsim_state_exchange_prognostic_halos (pawsim_state * state,
                                             const pawsim_domain * dom);

/* Read initial AWSIM input files, scatter to ranks, and fill halos. */
bool pawsim_state_init (pawsim_state * state, const pawsim_config * cfg,
                        const pawsim_domain * dom, FILE * errstrm);

/* Read restart U/V/H/B files from an existing output directory. */
bool pawsim_state_read_restart (pawsim_state * state, const pawsim_config * cfg,
                                const pawsim_domain * dom, const char * outdir,
                                uint startIdx, FILE * errstrm);

#endif
