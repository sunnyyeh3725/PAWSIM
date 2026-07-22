/**
 * tendency.c
 *
 * PAWSIM deterministic tendency setup, corresponding to the first pieces of
 * AWSIM's tderiv routine.
 *
 * This is the PAWSIM analogue of AWSIM's derivative callback. The integrator
 * asks for tendencies at a time and state; this module rebuilds work arrays,
 * applies deterministic physics, and copies the resulting dt_u/dt_v/dt_h
 * fields into the supplied tendency state.
 *
 */
#include "tendency.h"
#include "diapycnal.h"
#include "forcing.h"
#include "restoring.h"

/** Copy only owned cell values from a work-array tendency into a state buffer. */
static void copy_field_interior (pawsim_field2d * dst, const pawsim_field2d * src)
{
  uint i,j,g_dst,g_src;

  g_dst = dst->nghost;
  g_src = src->nghost;
  for (i = 0; i < dst->nx; i ++)
  {
    for (j = 0; j < dst->ny; j ++)
    {
      dst->a[i+g_dst][j+g_dst] = src->a[i+g_src][j+g_src];
    }
  }
}

/*
 * Rebuild diagnostic/work arrays without producing a derivative state. This is
 * used for initialization diagnostics and pressure-corrected preliminaries.
 */
void pawsim_tendency_prepare_work_arrays (pawsim_context * ctx)
{
  /** Used for diagnostics/preliminaries when the tendency state is not needed. */
  pawsim_work_load_state(&ctx->work,&ctx->state,&ctx->dom);
  pawsim_work_init_static_geometry(&ctx->work,&ctx->state,&ctx->cfg,&ctx->dom);
}

/*
 * Compute AWSIM's deterministic tderiv contents on ctx->work. Random forcing
 * is deliberately absent; prescribed deterministic forcing is handled here.
 */
void pawsim_tendency_compute_deterministic_core (pawsim_context * ctx,
                                                const pawsim_state * state)
{
  if ((ctx == NULL) || (state == NULL))
  {
    return;
  }

  /*
   * Preserve the AWSIM tderiv ordering: geometry and interface heights first,
   * then q/PV pressure work, momentum tendencies plus additive physics, and
   * finally the thickness tendency using the same mass fluxes and wdia.
   */
  pawsim_work_calc_face_thickness(&ctx->work,&ctx->cfg,&ctx->dom);
  pawsim_work_calc_eta(&ctx->work,&ctx->dom);
  pawsim_restoring_calc_wdia(ctx);
  pawsim_diapycnal_prepare_tracer_fluxes(ctx);
  pawsim_work_calc_pv(&ctx->work,state,&ctx->dom,ctx->dx,ctx->dy);
  pawsim_work_calc_pv_coefficients(&ctx->work,&ctx->cfg,&ctx->dom);
  pawsim_work_calc_bernoulli(&ctx->work,state,&ctx->cfg,&ctx->dom);
  pawsim_work_calc_momentum_tendency(&ctx->work,&ctx->cfg,&ctx->dom,ctx->dx,ctx->dy);
  pawsim_diapycnal_apply_momentum(ctx);
  pawsim_forcing_apply(ctx);
  pawsim_restoring_apply_momentum(ctx);
  pawsim_work_calc_thickness_tendency(&ctx->work,state,&ctx->cfg,&ctx->dom,ctx->dx,ctx->dy);
  pawsim_work_calc_tracer_tendency(&ctx->work,state,&ctx->cfg,&ctx->dom,ctx->dx,ctx->dy);
}

/*
 * Derivative callback used by the integrator. The explicit state is loaded
 * into work arrays, physics tendencies are computed, then copied to tendency.
 */
bool pawsim_tendency_eval (pawsim_context * ctx, real t,
                           const pawsim_state * state,
                           pawsim_state * tendency)
{
  uint k;

  if ((ctx == NULL) || (state == NULL) || (tendency == NULL))
  {
    return false;
  }

  if (!pawsim_forcing_update(ctx,t))
  {
    return false;
  }

  pawsim_state_zero_prognostic(tendency);
  pawsim_work_load_state(&ctx->work,state,&ctx->dom);
  pawsim_work_init_static_geometry(&ctx->work,state,&ctx->cfg,&ctx->dom);
  pawsim_tendency_compute_deterministic_core(ctx,state);

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    copy_field_interior(&tendency->u[k],&ctx->work.dt_u[k]);
    copy_field_interior(&tendency->v[k],&ctx->work.dt_v[k]);
    copy_field_interior(&tendency->h[k],&ctx->work.dt_h[k]);
    if (ctx->cfg.useTracer)
    {
      copy_field_interior(&tendency->b[k],&ctx->work.dt_b[k]);
    }
  }
  pawsim_state_exchange_prognostic_halos(tendency,&ctx->dom);

  return true;
}
