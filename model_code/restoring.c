/**
 * restoring.c
 *
 * AWSIM-style restoring terms for PAWSIM.
 *
 * Thickness and interface-height restoring enter AWSIM through an induced
 * diapycnal velocity at layer interfaces. Direct u/v restoring is applied as a
 * local Newtonian acceleration and is kept last among momentum forcing terms.
 *
 */
#include "restoring.h"

#include <math.h>

/*
 * Compute diapycnal velocity from prescribed wDia records and/or thickness
 * restoring. This mirrors AWSIM's use of interface fluxes in both h and b.
 */
void pawsim_restoring_calc_wdia (pawsim_context * ctx)
{
  uint i,j,g;
  int k;

  if (ctx == NULL)
  {
    return;
  }

  for (k = 0; k < (int) ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_zero(&ctx->work.wdia[k]);
  }

  if (!ctx->cfg.useWDia)
  {
    return;
  }

  if (strlen(ctx->cfg.wDiaFile) > 0)
  {
    uint n0 = 0;
    uint n1 = 0;
    real w0 = 1;
    real w1 = 0;

    if ((ctx->cfg.wDiaNrecs > 1) && (ctx->cfg.wDiaPeriod > 0))
    {
      real nflt = (fmod(ctx->t,ctx->cfg.wDiaPeriod) / ctx->cfg.wDiaPeriod) * ctx->cfg.wDiaNrecs;
      uint nm1 = (uint) floor(nflt);
      uint np1 = (uint) ceil(nflt);

      if (np1 == nm1)
      {
        np1 = nm1 + 1;
      }
      n0 = nm1;
      n1 = np1 % ctx->cfg.wDiaNrecs;
      w0 = np1 - nflt;
      w1 = nflt - nm1;
    }

    for (k = 0; k < (int) ctx->cfg.Nlay+1; k ++)
    {
      uint idx0 = n0*(ctx->cfg.Nlay+1) + (uint) k;
      uint idx1 = n1*(ctx->cfg.Nlay+1) + (uint) k;

      g = ctx->dom.nghost;
      for (i = 0; i < ctx->dom.nx; i ++)
      {
        uint il = i+g;

        for (j = 0; j < ctx->dom.ny; j ++)
        {
          uint jl = j+g;

          ctx->work.wdia[k].a[il][jl] = w0*ctx->state.wdia_ff[idx0].a[il][jl]
                                      + w1*ctx->state.wdia_ff[idx1].a[il][jl];
        }
      }
    }
  }

  if (!ctx->cfg.useRelax)
  {
    for (k = 0; k < (int) ctx->cfg.Nlay+1; k ++)
    {
      pawsim_field2d_exchange_scalar_halo(&ctx->work.wdia[k],&ctx->dom);
    }
  }

  g = ctx->dom.nghost;
  if (ctx->cfg.useRelax)
  {
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;
        real wdia_hRelax = 0;

        /*
         * Work upward from the bottom. hTime is shared across layers, so its
         * thickness-restoring contribution is cumulative in the AWSIM stencil.
         */
        for (k = (int) ctx->cfg.Nlay-1; k >= 0; k --)
        {
          if (ctx->state.hTime.a[il][jl] > 0)
          {
            wdia_hRelax += (ctx->work.h_w[k].a[il][jl]
                         - ctx->state.hRelax[k].a[il][jl]) / ctx->state.hTime.a[il][jl];
            ctx->work.wdia[k].a[il][jl] += wdia_hRelax;
          }

          if (ctx->state.eTime[k].a[il][jl] > 0)
          {
            ctx->work.wdia[k].a[il][jl] += (ctx->work.eta_w[k].a[il][jl]
                                         - ctx->state.eRelax[k].a[il][jl])
                                        / ctx->state.eTime[k].a[il][jl];
          }
        }
      }
    }
  }

  for (k = 0; k < (int) ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_exchange_scalar_halo(&ctx->work.wdia[k],&ctx->dom);
  }

  /*
   * Momentum equations need interface velocities on u/v points. Averaging from
   * the adjacent cell centers matches AWSIM; halo values supply neighboring
   * rank data, while physical-wall halos encode the boundary condition.
   */
  for (k = 0; k < (int) ctx->cfg.Nlay+1; k ++)
  {
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;

        ctx->work.wdia_u[k].a[il][jl] = 0.5*(ctx->work.wdia[k].a[il][jl]
                                           + ctx->work.wdia[k].a[il-1][jl]);
        ctx->work.wdia_v[k].a[il][jl] = 0.5*(ctx->work.wdia[k].a[il][jl]
                                           + ctx->work.wdia[k].a[il][jl-1]);
      }
    }
    pawsim_field2d_exchange_u_halo(&ctx->work.wdia_u[k],&ctx->dom);
    pawsim_field2d_exchange_v_halo(&ctx->work.wdia_v[k],&ctx->dom);
  }
}

/*
 * Apply Newtonian momentum restoring toward prescribed time-dependent velocity
 * targets, with the same layer-thickness weighting used in AWSIM diagnostics.
 */
void pawsim_restoring_apply_momentum (pawsim_context * ctx)
{
  uint i,j,k,g;

  if ((ctx == NULL) || !ctx->cfg.useRelax)
  {
    return;
  }

  g = ctx->dom.nghost;
  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;

        if (ctx->state.uTime[k].a[il][jl] > 0)
        {
          real rhs = - (ctx->work.u_w[k].a[il][jl]
                     - ctx->state.uRelax[k].a[il][jl])
                    / ctx->state.uTime[k].a[il][jl];
          ctx->work.dt_u[k].a[il][jl] += rhs;
        }
        if (ctx->state.vTime[k].a[il][jl] > 0)
        {
          real rhs = - (ctx->work.v_w[k].a[il][jl]
                     - ctx->state.vRelax[k].a[il][jl])
                    / ctx->state.vTime[k].a[il][jl];
          ctx->work.dt_v[k].a[il][jl] += rhs;
        }
      }
    }

    pawsim_field2d_exchange_u_halo(&ctx->work.dt_u[k],&ctx->dom);
    pawsim_field2d_exchange_v_halo(&ctx->work.dt_v[k],&ctx->dom);
  }
}
