/**
 * integrator.c
 *
 * PAWSIM time-integration implementation.
 *
 * AWSIM passes a tderiv function pointer into its Adams-Bashforth routines.
 * PAWSIM currently keeps the same conceptual split: tendency.c computes the
 * derivative, while this file owns AB startup and history rotation.
 *
 */
#include "integrator.h"
#include "tendency.h"

/** Apply a forward-Euler/AB1 update to one field. */
static void update_field_ab1 (pawsim_field2d * state,
                              const pawsim_field2d * f0,
                              real dt)
{
  uint i,j,g;

  g = state->nghost;
  for (i = 0; i < state->nx; i ++)
  {
    for (j = 0; j < state->ny; j ++)
    {
      state->a[i+g][j+g] += dt * f0->a[i+g][j+g];
    }
  }
}

/** Apply the second-order Adams-Bashforth combination to one field. */
static void update_field_ab2 (pawsim_field2d * state,
                              const pawsim_field2d * f0,
                              const pawsim_field2d * f1,
                              real dt)
{
  uint i,j,g;

  g = state->nghost;
  for (i = 0; i < state->nx; i ++)
  {
    for (j = 0; j < state->ny; j ++)
    {
      state->a[i+g][j+g] += (3*dt/2) * f0->a[i+g][j+g]
                          - (dt/2) * f1->a[i+g][j+g];
    }
  }
}

/** Apply the third-order Adams-Bashforth combination to one field. */
static void update_field_ab3 (pawsim_field2d * state,
                              const pawsim_field2d * f0,
                              const pawsim_field2d * f1,
                              const pawsim_field2d * f2,
                              real dt)
{
  uint i,j,g;

  g = state->nghost;
  for (i = 0; i < state->nx; i ++)
  {
    for (j = 0; j < state->ny; j ++)
    {
      state->a[i+g][j+g] += (23*dt/12) * f0->a[i+g][j+g]
                          - (4*dt/3) * f1->a[i+g][j+g]
                          + (5*dt/12) * f2->a[i+g][j+g];
    }
  }
}

/** Apply AB1 to every prognostic variable in the model state. */
static void update_state_ab1 (pawsim_context * ctx)
{
  uint k;

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    update_field_ab1(&ctx->state.u[k],&ctx->tendency.u[k],ctx->cfg.dt);
    update_field_ab1(&ctx->state.v[k],&ctx->tendency.v[k],ctx->cfg.dt);
    update_field_ab1(&ctx->state.h[k],&ctx->tendency.h[k],ctx->cfg.dt);
    if (ctx->cfg.useTracer)
    {
      update_field_ab1(&ctx->state.b[k],&ctx->tendency.b[k],ctx->cfg.dt);
    }
  }
}

/** Apply AB2 to every prognostic variable using the previous tendency buffer. */
static void update_state_ab2 (pawsim_context * ctx)
{
  uint k;

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    update_field_ab2(&ctx->state.u[k],&ctx->tendency.u[k],&ctx->tendency_1.u[k],ctx->cfg.dt);
    update_field_ab2(&ctx->state.v[k],&ctx->tendency.v[k],&ctx->tendency_1.v[k],ctx->cfg.dt);
    update_field_ab2(&ctx->state.h[k],&ctx->tendency.h[k],&ctx->tendency_1.h[k],ctx->cfg.dt);
    if (ctx->cfg.useTracer)
    {
      update_field_ab2(&ctx->state.b[k],&ctx->tendency.b[k],&ctx->tendency_1.b[k],ctx->cfg.dt);
    }
  }
}

/** Apply AB3 to every prognostic variable using two tendency-history buffers. */
static void update_state_ab3 (pawsim_context * ctx)
{
  uint k;

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    update_field_ab3(&ctx->state.u[k],&ctx->tendency.u[k],&ctx->tendency_1.u[k],
                     &ctx->tendency_2.u[k],ctx->cfg.dt);
    update_field_ab3(&ctx->state.v[k],&ctx->tendency.v[k],&ctx->tendency_1.v[k],
                     &ctx->tendency_2.v[k],ctx->cfg.dt);
    update_field_ab3(&ctx->state.h[k],&ctx->tendency.h[k],&ctx->tendency_1.h[k],
                     &ctx->tendency_2.h[k],ctx->cfg.dt);
    if (ctx->cfg.useTracer)
    {
      update_field_ab3(&ctx->state.b[k],&ctx->tendency.b[k],&ctx->tendency_1.b[k],
                       &ctx->tendency_2.b[k],ctx->cfg.dt);
    }
  }
}

/** Rotate derivative history after a successful explicit state update. */
static void shift_tendency_history (pawsim_context * ctx)
{
  /** Oldest history is overwritten; tendency always holds the newest derivative. */
  pawsim_state_copy_prognostic(&ctx->tendency_2,&ctx->tendency_1);
  pawsim_state_copy_prognostic(&ctx->tendency_1,&ctx->tendency);
}

/*
 * Advance one explicit time step, preserving AWSIM's lower-order AB startup
 * sequence before the selected scheme has enough tendency history.
 */
bool pawsim_integrator_step (pawsim_context * ctx, uint n)
{
  if (!pawsim_tendency_eval(ctx,ctx->t,&ctx->state,&ctx->tendency))
  {
    return false;
  }

  /*
   * AB2 and AB3 bootstrap with lower-order steps exactly as AWSIM does. This
   * mattered in the early AWSIM/PAWSIM comparisons: AB1 for the whole run
   * produced visibly different 25-day gyre states.
   */
  if (ctx->cfg.timeSteppingScheme == TIMESTEPPING_AB1)
  {
    update_state_ab1(ctx);
    shift_tendency_history(ctx);
  }
  else if (ctx->cfg.timeSteppingScheme == TIMESTEPPING_AB2)
  {
    if (n == 1)
    {
      update_state_ab1(ctx);
    }
    else
    {
      update_state_ab2(ctx);
    }
    shift_tendency_history(ctx);
  }
  else if (ctx->cfg.timeSteppingScheme == TIMESTEPPING_AB3)
  {
    if (n == 1)
    {
      update_state_ab1(ctx);
    }
    else if (n == 2)
    {
      update_state_ab2(ctx);
      shift_tendency_history(ctx);
    }
    else
    {
      update_state_ab3(ctx);
      shift_tendency_history(ctx);
    }
  }
  else
  {
    if (ctx->dom.rank == 0)
    {
      fprintf(stderr,"ERROR: PAWSIM only supports AB1, AB2 and AB3 time stepping so far\n");
    }
    return false;
  }

  ctx->t += ctx->cfg.dt;
  /** Refresh prognostic halos after every accepted step before pressure/update IO. */
  pawsim_state_exchange_prognostic_halos(&ctx->state,&ctx->dom);

  return true;
}
