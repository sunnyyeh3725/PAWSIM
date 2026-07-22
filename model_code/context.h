/**
 * context.h
 *
 * Shared PAWSIM model context.
 *
 */
#ifndef _PAWSIM_CONTEXT_H_
#define _PAWSIM_CONTEXT_H_

#include "config.h"
#include "domain.h"
#include "field.h"
#include "state.h"
#include "work.h"

typedef struct pawsim_context
{
  /* Long-lived model configuration, domain decomposition, and state. */
  pawsim_config cfg;
  pawsim_domain dom;
  pawsim_state state;
  /* Adams-Bashforth tendency history. */
  pawsim_state tendency;
  pawsim_state tendency_1;
  pawsim_state tendency_2;
  /* Shared stencil work arrays and rigid-lid pressure fields. */
  pawsim_work work;
  pawsim_field2d pi;
  pawsim_field2d pi_rhs;
  /*
   * AWSIM-compatible online averages. These are allocated only when
   * savefreqAvg > 0 and are written with the same *_avg file names.
   */
  pawsim_field2d * avg_u;
  pawsim_field2d * avg_v;
  pawsim_field2d * avg_h;
  pawsim_field2d * avg_M;
  pawsim_field2d * avg_b;
  pawsim_field2d * avg_wdia;
  pawsim_field2d * avg_hu;
  pawsim_field2d * avg_hv;
  pawsim_field2d * avg_huu;
  pawsim_field2d * avg_hvv;
  pawsim_field2d * avg_huv;
  pawsim_field2d avg_pi;
  const char * infname;
  const char * outdir;
  real dx;
  real dy;
  real t;
  /* Pressure diagnostics accumulated by pawsim.c when pressureTiming is set. */
  double pressure_walltime_total;
  real pressure_last_residual_max;
  real pressure_last_residual_rms;
  real pressure_last_divergence_max;
  real pressure_last_divergence_rms;
  uint pressure_last_solver;
  bool pressure_last_fallback;
  uint pressure_solve_count;
  uint pressure_fallback_count;
  uint pressure_iters_total;
  uint n_saves;
  uint n_avg;
  uint n_EZ;
  uint n_avg_hu;
  uint n_avg_hv;
  uint n_avg_h;
  uint n_avg_e;
  uint n_avg_b;
  uint n_prev_avg_hu;
  uint n_prev_avg_hv;
  uint n_prev_avg_h;
  uint n_prev_avg_e;
  uint n_prev_avg_b;
  real t_next_avg;
  real t_next_EZ;
  real t_next_avg_hu;
  real t_next_avg_hv;
  real t_next_avg_h;
  real t_next_avg_e;
  real t_next_avg_b;
}
pawsim_context;

#endif
