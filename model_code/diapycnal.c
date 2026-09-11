/**
 * diapycnal.c
 *
 * Diapycnal physics for PAWSIM.
 *
 * This module covers prescribed/restoring-induced vertical advection of
 * momentum, explicit diapycnal viscosity, tracer diapycnal diffusion, and the
 * AWSIM convective-adjustment enhancement keyed to interface reduced gravity.
 *
 */
#include "diapycnal.h"

/*
 * Compute reduced gravity at layer interfaces for convective enhancement of
 * diapycnal diffusion/viscosity. Active buoyancy modifies the static gg jump.
 */
static void calc_gprime (pawsim_context * ctx)
{
  uint i,j,k,g;

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    pawsim_field2d_zero(&ctx->work.gprime[k]);
  }

  if ((ctx->cfg.convDiff == 0) || (ctx->cfg.Nlay < 2))
  {
    return;
  }

  g = ctx->dom.nghost;
  for (k = 1; k < ctx->cfg.Nlay; k ++)
  {
    /*
     * gprime is the local reduced gravity at an interface. Without active
     * buoyancy it is just gg[k]; with active buoyancy, AWSIM adds the buoyancy
     * jump between adjacent layers.
     */
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;
        ctx->work.gprime[k].a[il][jl] = ctx->state.gg[k];
        if (ctx->cfg.useTracer && ctx->cfg.useBuoyancy)
        {
          ctx->work.gprime[k].a[il][jl] += ctx->work.b_w[k-1].a[il][jl]
                                        - ctx->work.b_w[k].a[il][jl];
        }
      }
    }
    pawsim_field2d_exchange_scalar_halo(&ctx->work.gprime[k],&ctx->dom);
  }
}

/*
 * Build explicit diapycnal viscous fluxes of u and v across layer interfaces.
 * Convectively unstable interfaces receive the AWSIM convDiff enhancement.
 */
static void calc_diapycnal_momentum_fluxes (pawsim_context * ctx)
{
  uint i,j,k,g;

  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_zero(&ctx->work.Fdia_u[k]);
    pawsim_field2d_zero(&ctx->work.Fdia_v[k]);
  }

  if (!ctx->cfg.useDiaVisc || (ctx->cfg.Nlay < 2))
  {
    return;
  }

  g = ctx->dom.nghost;
  for (k = 1; k < ctx->cfg.Nlay; k ++)
  {
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;
        real dz_u = 0.5*(ctx->work.h_west[k].a[il][jl]
                       + ctx->work.h_west[k-1].a[il][jl]);
        real dz_v = 0.5*(ctx->work.h_south[k].a[il][jl]
                       + ctx->work.h_south[k-1].a[il][jl]);
        real nu_u_eff = ctx->state.nu_u_dia[k].a[il][jl];
        real nu_v_eff = ctx->state.nu_v_dia[k].a[il][jl];

        if (ctx->cfg.convDiff != 0)
        {
          if ((ctx->work.gprime[k].a[il][jl] < 0)
           || (ctx->work.gprime[k].a[il-1][jl] < 0))
          {
            nu_u_eff += ctx->cfg.convDiff;
          }
          if ((ctx->work.gprime[k].a[il][jl] < 0)
           || (ctx->work.gprime[k].a[il][jl-1] < 0))
          {
            nu_v_eff += ctx->cfg.convDiff;
          }
        }

        if (dz_u != 0)
        {
          ctx->work.Fdia_u[k].a[il][jl] = nu_u_eff
                                        * (ctx->work.u_w[k-1].a[il][jl]
                                         - ctx->work.u_w[k].a[il][jl]) / dz_u;
        }
        if (dz_v != 0)
        {
          ctx->work.Fdia_v[k].a[il][jl] = nu_v_eff
                                        * (ctx->work.v_w[k-1].a[il][jl]
                                         - ctx->work.v_w[k].a[il][jl]) / dz_v;
        }
      }
    }
  }
}

/*
 * Build tracer diapycnal fluxes before tracer tendencies are evaluated. The
 * surface buoyancy flux, interior diffusion, and convective enhancement all
 * contribute to Fdia_b.
 */
void pawsim_diapycnal_prepare_tracer_fluxes (pawsim_context * ctx)
{
  uint i,j,k,g;

  if (ctx == NULL)
  {
    return;
  }

  calc_gprime(ctx);
  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_zero(&ctx->work.Fdia_b[k]);
  }

  if (!ctx->cfg.useTracer || !ctx->cfg.useDiaDiff)
  {
    return;
  }

  g = ctx->dom.nghost;
  for (i = 0; i < ctx->dom.nx; i ++)
  {
    uint il = i+g;

    for (j = 0; j < ctx->dom.ny; j ++)
    {
      uint jl = j+g;

      ctx->work.Fdia_b[0].a[il][jl] = ctx->state.Fsurf_b.a[il][jl];
      ctx->work.Fdia_b[ctx->cfg.Nlay].a[il][jl] = 0;
      for (k = 1; k < ctx->cfg.Nlay; k ++)
      {
        real db = ctx->work.b_w[k-1].a[il][jl] - ctx->work.b_w[k].a[il][jl];
        real dz = 0.5*(ctx->work.h_w[k].a[il][jl]+ctx->work.h_w[k-1].a[il][jl]);
        real kappa_eff = ctx->state.kappa_dia[k].a[il][jl];

        if (ctx->cfg.useBuoyancy)
        {
          db += ctx->state.gg[k];
        }
        if ((ctx->cfg.convDiff != 0) && (ctx->work.gprime[k].a[il][jl] < 0))
        {
          kappa_eff += ctx->cfg.convDiff;
        }
        if (dz != 0)
        {
          ctx->work.Fdia_b[k].a[il][jl] = kappa_eff * db / dz;
        }
      }
    }
  }

  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_exchange_scalar_halo(&ctx->work.Fdia_b[k],&ctx->dom);
  }
}

/*
 * Add vertical advection of momentum by wdia and explicit diapycnal viscosity
 * to the horizontal momentum tendencies, with matching AWSIM diagnostics.
 */
void pawsim_diapycnal_apply_momentum (pawsim_context * ctx)
{
  uint i,j,k,g;

  if (ctx == NULL)
  {
    return;
  }

  calc_gprime(ctx);
  calc_diapycnal_momentum_fluxes(ctx);

  if (!ctx->cfg.useWDia && !ctx->cfg.useDiaVisc)
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
        real hwest = ctx->work.h_west[k].a[il][jl];
        real hsouth = ctx->work.h_south[k].a[il][jl];

        if (ctx->cfg.useWDia)
        {
          real u_p = (k == 0) ? 0 : ctx->work.u_w[k-1].a[il][jl];
          real u_m = (k == ctx->cfg.Nlay-1) ? 0 : ctx->work.u_w[k+1].a[il][jl];
          real v_p = (k == 0) ? 0 : ctx->work.v_w[k-1].a[il][jl];
          real v_m = (k == ctx->cfg.Nlay-1) ? 0 : ctx->work.v_w[k+1].a[il][jl];
          real du_p = (ctx->work.wdia_u[k].a[il][jl] > 0) ? 0 : u_p-ctx->work.u_w[k].a[il][jl];
          real du_m = (ctx->work.wdia_u[k+1].a[il][jl] > 0) ? ctx->work.u_w[k].a[il][jl]-u_m : 0;
          real dv_p = (ctx->work.wdia_v[k].a[il][jl] > 0) ? 0 : v_p-ctx->work.v_w[k].a[il][jl];
          real dv_m = (ctx->work.wdia_v[k+1].a[il][jl] > 0) ? ctx->work.v_w[k].a[il][jl]-v_m : 0;

          if (hwest != 0)
          {
            real rhs = - (ctx->work.wdia_u[k].a[il][jl]*du_p
                        + ctx->work.wdia_u[k+1].a[il][jl]*du_m);
            ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          }
          if (hsouth != 0)
          {
            real rhs = - (ctx->work.wdia_v[k].a[il][jl]*dv_p
                        + ctx->work.wdia_v[k+1].a[il][jl]*dv_m);
            ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          }
        }

        if (ctx->cfg.useDiaVisc)
        {
          if (hwest != 0)
          {
            real rhs = ctx->work.Fdia_u[k].a[il][jl]
                     - ctx->work.Fdia_u[k+1].a[il][jl];
            ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          }
          if (hsouth != 0)
          {
            real rhs = ctx->work.Fdia_v[k].a[il][jl]
                     - ctx->work.Fdia_v[k+1].a[il][jl];
            ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          }
        }
      }
    }

    pawsim_field2d_exchange_u_halo(&ctx->work.dt_u[k],&ctx->dom);
    pawsim_field2d_exchange_v_halo(&ctx->work.dt_v[k],&ctx->dom);
  }
}
