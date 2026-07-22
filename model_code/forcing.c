/**
 * forcing.c
 *
 * PAWSIM optional forcing hooks.
 *
 * This module applies deterministic forcing terms after the core momentum and
 * thickness tendencies have been computed. Random forcing from AWSIM is not
 * ported yet; doing that well will need a parallel FFT/RNG design rather than
 * copying the serial FFTW state directly.
 *
 */
#include "forcing.h"

#include <math.h>

/** Add a term to an optional diagnostic budget array. */
static void diag_add (pawsim_field2d ** terms, uint term, uint k, uint i, uint j,
                      real value)
{
  if (terms != NULL)
  {
    terms[term][k].a[i][j] += value;
  }
}

/*
 * Select/interpolate time-dependent prescribed forcing records for time t.
 * PAWSIM follows AWSIM's file-driven forcing style rather than CLI flags.
 */
bool pawsim_forcing_update (pawsim_context * ctx, real t)
{
  uint i,j,g;
  uint n0 = 0;
  uint n1 = 0;
  real w0 = 1;
  real w1 = 0;

  if (ctx == NULL)
  {
    return false;
  }

  pawsim_field2d_zero(&ctx->work.taux_w);
  pawsim_field2d_zero(&ctx->work.tauy_w);

  if (!ctx->cfg.useWind)
  {
    return true;
  }

  if ((ctx->cfg.tauNrecs > 1) && (ctx->cfg.tauPeriod > 0))
  {
    /*
     * AWSIM wind files may contain a periodic sequence of records. Interpolate
     * in time so the tendency evaluation can be called at arbitrary AB times.
     */
    real nflt = (fmod(t,ctx->cfg.tauPeriod) / ctx->cfg.tauPeriod) * ctx->cfg.tauNrecs;
    uint nm1 = (uint) floor(nflt);
    uint np1 = (uint) ceil(nflt);

    if (np1 == nm1)
    {
      np1 = nm1 + 1;
    }
    n0 = nm1;
    n1 = np1 % ctx->cfg.tauNrecs;
    w0 = np1 - nflt;
    w1 = nflt - nm1;
  }

  g = ctx->dom.nghost;
  for (i = 0; i < ctx->dom.nx; i ++)
  {
    for (j = 0; j < ctx->dom.ny; j ++)
    {
      uint il = i+g;
      uint jl = j+g;

      ctx->work.taux_w.a[il][jl] = w0*ctx->state.taux[n0].a[il][jl]
                                 + w1*ctx->state.taux[n1].a[il][jl];
      ctx->work.tauy_w.a[il][jl] = w0*ctx->state.tauy[n0].a[il][jl]
                                 + w1*ctx->state.tauy[n1].a[il][jl];
    }
  }

  pawsim_field2d_exchange_scalar_halo(&ctx->work.taux_w,&ctx->dom);
  pawsim_field2d_exchange_scalar_halo(&ctx->work.tauy_w,&ctx->dom);
  return true;
}

/*
 * Compute layer-relative surface speeds used by linear/quadratic surface drag.
 * When windFeedback is enabled, the lid velocity is the moving reference frame.
 */
static void calc_surface_drag_speed (pawsim_context * ctx)
{
  uint i,j,k;
  real wfb_sw = ctx->cfg.windFeedback ? 0.3 : 0.0;

  /*
   * AWSIM evaluates quadratic surface drag from the lid-relative speed at the
   * surface. With hsml > 0 that speed is a thickness-weighted mixed-layer
   * average; otherwise it is simply the top-layer velocity.
   */
  for (i = 0; i < ctx->work.usq_surf.sx; i ++)
  {
    for (j = 0; j < ctx->work.usq_surf.sy; j ++)
    {
      real u_surf = 0;
      real v_surf = 0;
      real u_rel,v_rel;

      if (ctx->cfg.hsml > 0)
      {
        for (k = 0; k < ctx->cfg.Nlay; k ++)
        {
          u_surf += ctx->work.u_w[k].a[i][j] * ctx->work.hFsurf_west[k].a[i][j];
          v_surf += ctx->work.v_w[k].a[i][j] * ctx->work.hFsurf_south[k].a[i][j];
        }
      }
      else
      {
        u_surf = ctx->work.u_w[0].a[i][j];
        v_surf = ctx->work.v_w[0].a[i][j];
      }

      u_rel = (1.0-wfb_sw)*u_surf - ctx->state.uLid.a[i][j];
      v_rel = (1.0-wfb_sw)*v_surf - ctx->state.vLid.a[i][j];
      ctx->work.usq_surf.a[i][j] = u_rel * u_rel;
      ctx->work.vsq_surf.a[i][j] = v_rel * v_rel;
    }
  }

  for (i = 0; i < ctx->work.uabs_surf.sx-1; i ++)
  {
    for (j = 0; j < ctx->work.uabs_surf.sy-1; j ++)
    {
      ctx->work.uabs_surf.a[i][j] =
        sqrt(0.5*(ctx->work.usq_surf.a[i][j]+ctx->work.usq_surf.a[i+1][j])
           + 0.5*(ctx->work.vsq_surf.a[i][j]+ctx->work.vsq_surf.a[i][j+1]));
    }
  }
}

/*
 * Add wind stress, barotropic body forcing, and surface drag to momentum
 * tendencies, including AWSIM-compatible momentum/energy budget bookkeeping.
 */
void pawsim_forcing_apply (pawsim_context * ctx)
{
  uint i,j,k,g;
  bool useSurfDrag;
  bool useBotDrag;
  real wfb_sw;

  if (ctx == NULL)
  {
    return;
  }

  useSurfDrag = (ctx->cfg.linDragSurf > 0) || (ctx->cfg.quadDragSurf > 0);
  useBotDrag = (ctx->cfg.linDragCoeff > 0) || (ctx->cfg.quadDragCoeff > 0);
  wfb_sw = ctx->cfg.windFeedback ? 0.3 : 0.0;

  if (ctx->cfg.useWind || useSurfDrag)
  {
    /** Convert the surface mixed-layer mask/thickness to u/v face locations. */
    pawsim_work_calc_surface_forcing_thickness(&ctx->work,&ctx->cfg,&ctx->dom);
  }
  if (useBotDrag)
  {
    /** Convert the bottom boundary-layer mask/thickness to u/v face locations. */
    pawsim_work_calc_bottom_forcing_thickness(&ctx->work,&ctx->cfg,&ctx->dom);
  }

  if (ctx->cfg.quadDragSurf > 0)
  {
    calc_surface_drag_speed(ctx);
  }

  if (ctx->cfg.quadDragCoeff > 0)
  {
    /*
     * Quadratic drag uses the bottom-layer or bottom-boundary-layer velocity
     * magnitude. For hbbl > 0, the velocity is a thickness-weighted vertical
     * average over the part of each layer inside the bottom boundary layer.
     */
    for (i = 0; i < ctx->work.usq_bot.sx; i ++)
    {
      for (j = 0; j < ctx->work.usq_bot.sy; j ++)
      {
        real u_bot = 0;
        real v_bot = 0;

        if (ctx->cfg.hbbl > 0)
        {
          for (k = 0; k < ctx->cfg.Nlay; k ++)
          {
            u_bot += ctx->work.u_w[k].a[i][j] * ctx->work.hFbot_west[k].a[i][j];
            v_bot += ctx->work.v_w[k].a[i][j] * ctx->work.hFbot_south[k].a[i][j];
          }
        }
        else
        {
          u_bot = ctx->work.u_w[ctx->cfg.Nlay-1].a[i][j];
          v_bot = ctx->work.v_w[ctx->cfg.Nlay-1].a[i][j];
        }

        ctx->work.usq_bot.a[i][j] = u_bot * u_bot;
        ctx->work.vsq_bot.a[i][j] = v_bot * v_bot;
      }
    }

    for (i = 0; i < ctx->work.uabs_bot.sx-1; i ++)
    {
      for (j = 0; j < ctx->work.uabs_bot.sy-1; j ++)
      {
        ctx->work.uabs_bot.a[i][j] =
          sqrt(0.5*(ctx->work.usq_bot.a[i][j]+ctx->work.usq_bot.a[i+1][j])
             + 0.5*(ctx->work.vsq_bot.a[i][j]+ctx->work.vsq_bot.a[i][j+1]));
      }
    }
  }

  if (!ctx->cfg.useWind && !ctx->cfg.useFbaro && !useSurfDrag && !useBotDrag)
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

        /** Wind and drag accelerations are force divided by active face thickness. */
        if (ctx->cfg.useWind)
        {
          real rhs = ctx->work.taux_w.a[il][jl]
                   * ctx->work.hFsurf_west[k].a[il][jl];
          if (hwest != 0)
          {
            ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          }
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_WIND,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_WIND,k,il,jl,
                   rhs*ctx->work.u_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if (ctx->cfg.useWind)
        {
          real rhs = ctx->work.tauy_w.a[il][jl]
                   * ctx->work.hFsurf_south[k].a[il][jl];
          if (hsouth != 0)
          {
            ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          }
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_WIND,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_WIND,k,il,jl,
                   rhs*ctx->work.v_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if (ctx->cfg.useFbaro)
        {
          ctx->work.dt_u[k].a[il][jl] += ctx->state.Fbaro_x.a[il][jl];
          ctx->work.dt_v[k].a[il][jl] += ctx->state.Fbaro_y.a[il][jl];
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_FBARO,k,il,jl,
                   hwest*ctx->state.Fbaro_x.a[il][jl]*ctx->cfg.dt);
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_FBARO,k,il,jl,
                   hsouth*ctx->state.Fbaro_y.a[il][jl]*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_FBARO,k,il,jl,
                   (hwest*ctx->state.Fbaro_x.a[il][jl]*ctx->work.u_w[k].a[il][jl]
                  + hsouth*ctx->state.Fbaro_y.a[il][jl]*ctx->work.v_w[k].a[il][jl])
                   * ctx->cfg.dt);
        }
        if ((ctx->cfg.linDragCoeff > 0) && (hwest != 0))
        {
          real rhs = -ctx->cfg.linDragCoeff
                   * ctx->work.u_w[k].a[il][jl]
                   * ctx->work.hFbot_west[k].a[il][jl];
          ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_RDRAG,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_RDRAG,k,il,jl,
                   rhs*ctx->work.u_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.linDragCoeff > 0) && (hsouth != 0))
        {
          real rhs = -ctx->cfg.linDragCoeff
                   * ctx->work.v_w[k].a[il][jl]
                   * ctx->work.hFbot_south[k].a[il][jl];
          ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_RDRAG,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_RDRAG,k,il,jl,
                   rhs*ctx->work.v_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.linDragSurf > 0) && (hwest != 0))
        {
          real rhs = -ctx->cfg.linDragSurf
                   * (ctx->work.u_w[k].a[il][jl]-ctx->state.uLid.a[il][jl])
                   * ctx->work.hFsurf_west[k].a[il][jl];
          if (!ctx->cfg.oceanSurfDrag && ctx->cfg.useRL && (ctx->work.hhs_west.a[il][jl] >= 0))
          {
            rhs = 0;
          }
          ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_RSURF,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_RSURF,k,il,jl,
                   rhs*ctx->work.u_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.linDragSurf > 0) && (hsouth != 0))
        {
          real rhs = -ctx->cfg.linDragSurf
                   * (ctx->work.v_w[k].a[il][jl]-ctx->state.vLid.a[il][jl])
                   * ctx->work.hFsurf_south[k].a[il][jl];
          if (!ctx->cfg.oceanSurfDrag && ctx->cfg.useRL && (ctx->work.hhs_south.a[il][jl] >= 0))
          {
            rhs = 0;
          }
          ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_RSURF,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_RSURF,k,il,jl,
                   rhs*ctx->work.v_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.quadDragCoeff > 0) && (hwest != 0))
        {
          real uabs = 0.5*(ctx->work.uabs_bot.a[il][jl]+ctx->work.uabs_bot.a[il-1][jl]);

          ctx->work.dt_u[k].a[il][jl] -= ctx->cfg.quadDragCoeff * uabs
                                       * ctx->work.u_w[k].a[il][jl]
                                       * ctx->work.hFbot_west[k].a[il][jl]
                                       / hwest;
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_CDBOT,k,il,jl,
                   -ctx->cfg.quadDragCoeff*uabs*ctx->work.u_w[k].a[il][jl]
                   * ctx->work.hFbot_west[k].a[il][jl]*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_CDBOT,k,il,jl,
                   -ctx->cfg.quadDragCoeff*uabs*SQUARE(ctx->work.u_w[k].a[il][jl])
                   * ctx->work.hFbot_west[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.quadDragCoeff > 0) && (hsouth != 0))
        {
          real uabs = 0.5*(ctx->work.uabs_bot.a[il][jl]+ctx->work.uabs_bot.a[il][jl-1]);

          ctx->work.dt_v[k].a[il][jl] -= ctx->cfg.quadDragCoeff * uabs
                                       * ctx->work.v_w[k].a[il][jl]
                                       * ctx->work.hFbot_south[k].a[il][jl]
                                       / hsouth;
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_CDBOT,k,il,jl,
                   -ctx->cfg.quadDragCoeff*uabs*ctx->work.v_w[k].a[il][jl]
                   * ctx->work.hFbot_south[k].a[il][jl]*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_CDBOT,k,il,jl,
                   -ctx->cfg.quadDragCoeff*uabs*SQUARE(ctx->work.v_w[k].a[il][jl])
                   * ctx->work.hFbot_south[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.quadDragSurf > 0) && (hwest != 0))
        {
          real uabs = 0.5*(ctx->work.uabs_surf.a[il][jl]+ctx->work.uabs_surf.a[il-1][jl]);
          real rhs = -ctx->cfg.quadDragSurf * uabs
                   * ((1.0-wfb_sw)*ctx->work.u_w[k].a[il][jl]-ctx->state.uLid.a[il][jl])
                   * ctx->work.hFsurf_west[k].a[il][jl];
          if (!ctx->cfg.oceanSurfDrag && ctx->cfg.useRL && (ctx->work.hhs_west.a[il][jl] >= 0))
          {
            rhs = 0;
          }
          ctx->work.dt_u[k].a[il][jl] += rhs / hwest;
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_CDSURF,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_CDSURF,k,il,jl,
                   rhs*ctx->work.u_w[k].a[il][jl]*ctx->cfg.dt);
        }
        if ((ctx->cfg.quadDragSurf > 0) && (hsouth != 0))
        {
          real uabs = 0.5*(ctx->work.uabs_surf.a[il][jl]+ctx->work.uabs_surf.a[il][jl-1]);
          real rhs = -ctx->cfg.quadDragSurf * uabs
                   * ((1.0-wfb_sw)*ctx->work.v_w[k].a[il][jl]-ctx->state.vLid.a[il][jl])
                   * ctx->work.hFsurf_south[k].a[il][jl];
          if (!ctx->cfg.oceanSurfDrag && ctx->cfg.useRL && (ctx->work.hhs_south.a[il][jl] >= 0))
          {
            rhs = 0;
          }
          ctx->work.dt_v[k].a[il][jl] += rhs / hsouth;
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_CDSURF,k,il,jl,rhs*ctx->cfg.dt);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_CDSURF,k,il,jl,
                   rhs*ctx->work.v_w[k].a[il][jl]*ctx->cfg.dt);
        }
      }
    }

    pawsim_field2d_exchange_u_halo(&ctx->work.dt_u[k],&ctx->dom);
    pawsim_field2d_exchange_v_halo(&ctx->work.dt_v[k],&ctx->dom);
  }
}
