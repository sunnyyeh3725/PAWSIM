/**
 * diagnostics.c
 *
 * PAWSIM diagnostic and model-state output implementation.
 *
 * Prognostic outputs are intended to match AWSIM naming and layout. The
 * init_* diagnostics are development and regression aids for checking halo
 * fills, q-grid construction, pressure geometry, and tendency terms. They are
 * written only when the PAWSIM-only writeInitDiagnostics input flag is set.
 *
 */
#include "diagnostics.h"
#include "io.h"

/** Write one named cell-centered field into the PAWSIM/AWSIM output directory. */
static bool write_named_field (const char * outdir, const char * name,
                               const pawsim_field2d * field,
                               const pawsim_domain * dom)
{
  char outfile[MAX_PARAMETER_FILENAME_LENGTH];

  snprintf(outfile,sizeof(outfile),"%s/%s",outdir,name);
  return pawsim_write_global_field2d(outfile,field,dom);
}

/** Write one named q-grid field, including the wall-boundary q line. */
static bool write_named_qfield (const char * outdir, const char * name,
                                const pawsim_field2d * field,
                                const pawsim_domain * dom)
{
  char outfile[MAX_PARAMETER_FILENAME_LENGTH];

  snprintf(outfile,sizeof(outfile),"%s/%s",outdir,name);
  return pawsim_write_global_qfield2d(outfile,field,dom);
}

/** Write an AWSIM-style layer output such as U0_n=3.dat or PHIH1_n=2.dat. */
static bool write_layer_field (const char * outdir, const char * prefix, uint k, uint n,
                               const pawsim_field2d * field,
                               const pawsim_domain * dom)
{
  char name[MAX_PARAMETER_FILENAME_LENGTH];

  snprintf(name,sizeof(name),"%s%u_n=%u.dat",prefix,k,n);
  return write_named_field(outdir,name,field,dom);
}

/** Allocate a per-layer diagnostic array on the local MPI tile. */
static bool alloc_layer_fields (pawsim_field2d ** fields, uint n,
                                const pawsim_domain * dom)
{
  uint k;

  *fields = NULL;
  if (n == 0)
  {
    return true;
  }

  *fields = calloc(n,sizeof(pawsim_field2d));
  if (*fields == NULL)
  {
    return false;
  }

  for (k = 0; k < n; k ++)
  {
    if (!pawsim_field2d_alloc(&(*fields)[k],dom->nx,dom->ny,dom->nghost))
    {
      return false;
    }
  }

  return true;
}

/** Free a per-layer diagnostic array allocated by alloc_layer_fields(). */
static void free_layer_fields (pawsim_field2d * fields, uint n)
{
  uint k;

  if (fields == NULL)
  {
    return;
  }

  for (k = 0; k < n; k ++)
  {
    pawsim_field2d_free(&fields[k]);
  }
  free(fields);
}

/** Zero all fields in a per-layer diagnostic array. */
static void zero_layer_fields (pawsim_field2d * fields, uint n)
{
  uint k;

  if (fields == NULL)
  {
    return;
  }

  for (k = 0; k < n; k ++)
  {
    pawsim_field2d_zero(&fields[k]);
  }
}

/** Zero one full diagnostic-budget family, e.g. all U-momentum terms. */
static void zero_budget_terms (pawsim_field2d ** terms, uint nterms, uint nlay)
{
  uint m;

  if (terms == NULL)
  {
    return;
  }

  for (m = 0; m < nterms; m ++)
  {
    zero_layer_fields(terms[m],nlay);
  }
}

/** Scale only owned cells before writing an average over a completed window. */
static void scale_field_interior (pawsim_field2d * field, real scale)
{
  uint i,j,g;

  g = field->nghost;
  for (i = g; i < g+field->nx; i ++)
  {
    for (j = g; j < g+field->ny; j ++)
    {
      field->a[i][j] *= scale;
    }
  }
}

/*
 * Allocate online-average storage. Budget diagnostic storage is owned by
 * pawsim_work_alloc(), because tendency routines accumulate those terms there.
 */
bool pawsim_diagnostics_alloc (pawsim_context * ctx)
{
  if ((ctx == NULL) || (ctx->cfg.savefreqAvg <= 0))
  {
    return true;
  }

  return alloc_layer_fields(&ctx->avg_u,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_v,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_h,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_M,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_b,ctx->cfg.useTracer ? ctx->cfg.Nlay : 0,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_wdia,ctx->cfg.Nlay+1,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_hu,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_hv,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_huu,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_hvv,ctx->cfg.Nlay,&ctx->dom)
      && alloc_layer_fields(&ctx->avg_huv,ctx->cfg.Nlay,&ctx->dom)
      && pawsim_field2d_alloc(&ctx->avg_pi,ctx->dom.nx,ctx->dom.ny,ctx->dom.nghost);
}

/** Free online-average storage; safe for partially allocated contexts. */
void pawsim_diagnostics_free (pawsim_context * ctx)
{
  if (ctx == NULL)
  {
    return;
  }

  free_layer_fields(ctx->avg_u,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_v,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_h,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_M,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_b,ctx->cfg.useTracer ? ctx->cfg.Nlay : 0);
  free_layer_fields(ctx->avg_wdia,ctx->cfg.Nlay+1);
  free_layer_fields(ctx->avg_hu,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_hv,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_huu,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_hvv,ctx->cfg.Nlay);
  free_layer_fields(ctx->avg_huv,ctx->cfg.Nlay);
  pawsim_field2d_free(&ctx->avg_pi);
}

/** Reset all running average accumulators at the start of an averaging window. */
void pawsim_diagnostics_zero_averages (pawsim_context * ctx)
{
  if ((ctx == NULL) || (ctx->cfg.savefreqAvg <= 0))
  {
    return;
  }

  zero_layer_fields(ctx->avg_u,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_v,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_h,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_M,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_b,ctx->cfg.useTracer ? ctx->cfg.Nlay : 0);
  zero_layer_fields(ctx->avg_wdia,ctx->cfg.Nlay+1);
  zero_layer_fields(ctx->avg_hu,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_hv,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_huu,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_hvv,ctx->cfg.Nlay);
  zero_layer_fields(ctx->avg_huv,ctx->cfg.Nlay);
  pawsim_field2d_zero(&ctx->avg_pi);
}

/** Reset the tendency-budget accumulators after their corresponding write. */
void pawsim_diagnostics_zero_budgets (pawsim_context * ctx)
{
  if (ctx == NULL)
  {
    return;
  }

  zero_budget_terms(ctx->work.diag_umom,PAWSIM_UMOM_NTERMS,ctx->cfg.Nlay);
  zero_budget_terms(ctx->work.diag_vmom,PAWSIM_VMOM_NTERMS,ctx->cfg.Nlay);
  zero_budget_terms(ctx->work.diag_thic,PAWSIM_THIC_NTERMS,ctx->cfg.Nlay);
  zero_budget_terms(ctx->work.diag_energy,PAWSIM_ENERGY_NTERMS,ctx->cfg.Nlay);
  zero_budget_terms(ctx->work.diag_trac,PAWSIM_TRAC_NTERMS,ctx->cfg.Nlay);
}

/*
 * Refresh halos for average fields so products using neighboring points match
 * AWSIM's global-array calculations before output.
 */
static void exchange_average_halos (pawsim_context * ctx)
{
  uint k;

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_u[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_v[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_h[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_M[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_hu[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_hv[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_huu[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_hvv[k],&ctx->dom);
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_huv[k],&ctx->dom);
    if (ctx->cfg.useTracer)
    {
      pawsim_field2d_exchange_scalar_halo(&ctx->avg_b[k],&ctx->dom);
    }
  }
  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    pawsim_field2d_exchange_scalar_halo(&ctx->avg_wdia[k],&ctx->dom);
  }
  pawsim_field2d_exchange_scalar_halo(&ctx->avg_pi,&ctx->dom);
}

/*
 * Add the latest state to AWSIM-compatible online averages. AWSIM no longer
 * interpolates across save times here, because interpolation can break
 * conservation; PAWSIM follows that convention.
 */
bool pawsim_diagnostics_accumulate_averages (pawsim_context * ctx)
{
  uint i,j,k,g;
  real avg_fac;

  if ((ctx == NULL) || (ctx->cfg.savefreqAvg <= 0))
  {
    return true;
  }

  avg_fac = (ctx->t < ctx->t_next_avg) ? 1 : 1-(ctx->t-ctx->t_next_avg)/ctx->cfg.dt;
  g = ctx->dom.nghost;

  pawsim_work_load_state(&ctx->work,&ctx->state,&ctx->dom);
  if (!pawsim_work_calc_face_thickness_no_ghost(&ctx->work,&ctx->cfg,&ctx->dom))
  {
    return false;
  }
  pawsim_work_calc_eta(&ctx->work,&ctx->dom);
  pawsim_work_calc_bernoulli(&ctx->work,&ctx->state,&ctx->cfg,&ctx->dom);

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    for (i = g; i < g+ctx->dom.nx; i ++)
    {
      for (j = g; j < g+ctx->dom.ny; j ++)
      {
        uint im1 = i-1;
        uint jm1 = j-1;
        real dtfac = avg_fac * ctx->cfg.dt;
        real hq = 0.25*(ctx->state.h[k].a[i][j]+ctx->state.h[k].a[im1][j]
                      + ctx->state.h[k].a[i][jm1]+ctx->state.h[k].a[im1][jm1]);
        real uq = 0.5*(ctx->state.u[k].a[i][j]+ctx->state.u[k].a[i][jm1]);
        real vq = 0.5*(ctx->state.v[k].a[i][j]+ctx->state.v[k].a[im1][j]);

        ctx->avg_u[k].a[i][j] += dtfac * ctx->state.u[k].a[i][j];
        ctx->avg_v[k].a[i][j] += dtfac * ctx->state.v[k].a[i][j];
        ctx->avg_h[k].a[i][j] += dtfac * ctx->state.h[k].a[i][j];
        ctx->avg_M[k].a[i][j] += dtfac * (ctx->work.MM_B[k].a[i][j]
                                      + (ctx->cfg.useRL ? ctx->pi.a[i][j] : 0));
        ctx->avg_wdia[k].a[i][j] += dtfac * ctx->work.wdia[k].a[i][j];
        ctx->avg_hu[k].a[i][j] += dtfac * ctx->work.h_west[k].a[i][j] * ctx->state.u[k].a[i][j];
        ctx->avg_hv[k].a[i][j] += dtfac * ctx->work.h_south[k].a[i][j] * ctx->state.v[k].a[i][j];
        ctx->avg_huu[k].a[i][j] += dtfac * ctx->work.h_west[k].a[i][j]
                                  * ctx->state.u[k].a[i][j] * ctx->state.u[k].a[i][j];
        ctx->avg_hvv[k].a[i][j] += dtfac * ctx->work.h_south[k].a[i][j]
                                  * ctx->state.v[k].a[i][j] * ctx->state.v[k].a[i][j];
        ctx->avg_huv[k].a[i][j] += dtfac * hq * uq * vq;
        if (ctx->cfg.useTracer)
        {
          ctx->avg_b[k].a[i][j] += dtfac * ctx->state.b[k].a[i][j];
        }
      }
    }
  }

  for (i = g; i < g+ctx->dom.nx; i ++)
  {
    for (j = g; j < g+ctx->dom.ny; j ++)
    {
      real dtfac = avg_fac * ctx->cfg.dt;
      ctx->avg_wdia[ctx->cfg.Nlay].a[i][j] += dtfac * ctx->work.wdia[ctx->cfg.Nlay].a[i][j];
      ctx->avg_pi.a[i][j] += dtfac * ctx->pi.a[i][j];
    }
  }

  exchange_average_halos(ctx);
  return true;
}

/** Write one completed online-average record using AWSIM diagnostic names. */
static bool write_average_state (pawsim_context * ctx, uint n)
{
  uint k;
  char name[MAX_PARAMETER_FILENAME_LENGTH];

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    if (!write_layer_field(ctx->outdir,OUTN_U_AVG,k,n,&ctx->avg_u[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_V_AVG,k,n,&ctx->avg_v[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_H_AVG,k,n,&ctx->avg_h[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_M_AVG,k,n,&ctx->avg_M[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_W_AVG,k,n,&ctx->avg_wdia[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_HU_AVG,k,n,&ctx->avg_hu[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_HV_AVG,k,n,&ctx->avg_hv[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_HUU_AVG,k,n,&ctx->avg_huu[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_HVV_AVG,k,n,&ctx->avg_hvv[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,OUTN_HUV_AVG,k,n,&ctx->avg_huv[k],&ctx->dom))
    {
      return false;
    }

    if (ctx->cfg.useTracer
     && !write_layer_field(ctx->outdir,OUTN_B_AVG,k,n,&ctx->avg_b[k],&ctx->dom))
    {
      return false;
    }
  }

  if (!write_layer_field(ctx->outdir,OUTN_W_AVG,ctx->cfg.Nlay,n,&ctx->avg_wdia[ctx->cfg.Nlay],&ctx->dom))
  {
    return false;
  }

  if (ctx->cfg.useRL)
  {
    snprintf(name,sizeof(name),"%s_n=%u.dat",OUTN_PI_AVG,n);
    if (!write_named_field(ctx->outdir,name,&ctx->avg_pi,&ctx->dom))
    {
      return false;
    }
  }

  return true;
}

/*
 * If the model has reached the end of an averaging window, normalize the
 * accumulated sums by the window length, write them, and reset for the next
 * window. The counters avoid accumulating floating-point time drift.
 */
bool pawsim_diagnostics_write_averages_if_due (pawsim_context * ctx)
{
  uint k;

  if ((ctx == NULL) || (ctx->cfg.savefreqAvg <= 0) || (ctx->t < ctx->t_next_avg))
  {
    return true;
  }

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    scale_field_interior(&ctx->avg_u[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_v[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_h[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_M[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_hu[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_hv[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_huu[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_hvv[k],1/ctx->cfg.savefreqAvg);
    scale_field_interior(&ctx->avg_huv[k],1/ctx->cfg.savefreqAvg);
    if (ctx->cfg.useTracer)
    {
      scale_field_interior(&ctx->avg_b[k],1/ctx->cfg.savefreqAvg);
    }
  }
  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    scale_field_interior(&ctx->avg_wdia[k],1/ctx->cfg.savefreqAvg);
  }
  scale_field_interior(&ctx->avg_pi,1/ctx->cfg.savefreqAvg);
  exchange_average_halos(ctx);

  if (!write_average_state(ctx,ctx->n_avg))
  {
    return false;
  }

  pawsim_diagnostics_zero_averages(ctx);

  ctx->n_avg ++;
  ctx->t_next_avg = ctx->n_avg * ctx->cfg.savefreqAvg;
  return true;
}

/** Lightweight accessor used where energy/enstrophy formulas read halo values. */
static real local_or_halo (const pawsim_field2d * field, uint i, uint j)
{
  return field->a[i][j];
}

/*
 * Write total kinetic energy, potential energy, and layer potential enstrophy.
 * Like AWSIM, PAWSIM writes the latest model state at the diagnostic time and
 * does not interpolate because interpolation would spoil conservation checks.
 */
bool pawsim_diagnostics_write_EZ_if_due (pawsim_context * ctx)
{
  uint i,j,k,g;
  char outfile[MAX_PARAMETER_FILENAME_LENGTH];

  if ((ctx == NULL) || (ctx->cfg.savefreqEZ <= 0))
  {
    return true;
  }

  while (ctx->t >= ctx->t_next_EZ)
  {
    double local_KE = 0;
    double local_PE = 0;
    double global_KE = 0;
    double global_PE = 0;
    double * local_Z = NULL;
    double * global_Z = NULL;
    FILE * file = NULL;

    local_Z = calloc(ctx->cfg.Nlay,sizeof(double));
    global_Z = calloc(ctx->cfg.Nlay,sizeof(double));
    if ((local_Z == NULL) || (global_Z == NULL))
    {
      free(local_Z);
      free(global_Z);
      return false;
    }

    pawsim_work_calc_eta(&ctx->work,&ctx->dom);
    pawsim_work_calc_pv(&ctx->work,&ctx->state,&ctx->dom,ctx->dx,ctx->dy);

    g = ctx->dom.nghost;
    for (k = 0; k < ctx->cfg.Nlay; k ++)
    {
      for (i = g; i < g+ctx->dom.nx; i ++)
      {
        uint gi = ctx->dom.i0 + (i-g);
        uint ip1 = i+1;
        uint im1 = i-1;

        for (j = g; j < g+ctx->dom.ny; j ++)
        {
          uint gj = ctx->dom.j0 + (j-g);
          uint jp1 = j+1;
          uint jm1 = j-1;
          real h_q;
          real zeta;
          real pv;
          real KEdens = 0;
          real gtot;

          if (ctx->cfg.useWallNS && (gj == 0))
          {
            h_q = 0.5*(local_or_halo(&ctx->state.h[k],im1,j)
                     + local_or_halo(&ctx->state.h[k],i,j));
            zeta = 0;
          }
          else if (ctx->cfg.useWallEW && (gi == 0))
          {
            h_q = 0.5*(local_or_halo(&ctx->state.h[k],i,jm1)
                     + local_or_halo(&ctx->state.h[k],i,j));
            zeta = 0;
          }
          else
          {
            h_q = 0.25*(local_or_halo(&ctx->state.h[k],i,j)
                      + local_or_halo(&ctx->state.h[k],im1,jm1)
                      + local_or_halo(&ctx->state.h[k],im1,j)
                      + local_or_halo(&ctx->state.h[k],i,jm1));
            zeta = (local_or_halo(&ctx->state.u[k],i,jm1)
                  - local_or_halo(&ctx->state.u[k],i,j)) / ctx->dy
                 + (local_or_halo(&ctx->state.v[k],i,j)
                  - local_or_halo(&ctx->state.v[k],im1,j)) / ctx->dx;
          }
          if (h_q != 0)
          {
            pv = (2*local_or_halo(&ctx->state.omega_z,i,j) + zeta) / h_q;
            local_Z[k] += ctx->dx*ctx->dy * 0.5 * h_q * SQUARE(pv);
          }

          switch (ctx->cfg.thicknessScheme)
          {
            case THICKNESS_AL81:
            case THICKNESS_UP3:
            case THICKNESS_KT00:
            {
              KEdens = 0.125 * ctx->dx*ctx->dy
                     * ((local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],ip1,j))
                        * SQUARE(local_or_halo(&ctx->state.u[k],ip1,j))
                      + (local_or_halo(&ctx->state.h[k],im1,j)+local_or_halo(&ctx->state.h[k],i,j))
                        * SQUARE(local_or_halo(&ctx->state.u[k],i,j))
                      + (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],i,jp1))
                        * SQUARE(local_or_halo(&ctx->state.v[k],i,jp1))
                      + (local_or_halo(&ctx->state.h[k],i,jm1)+local_or_halo(&ctx->state.h[k],i,j))
                        * SQUARE(local_or_halo(&ctx->state.v[k],i,j)));
              break;
            }
            case THICKNESS_HK83:
            {
              KEdens = 0;
              if (ctx->cfg.useWallNS && (gj == 0))
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],ip1,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,jp1)))
                        + (local_or_halo(&ctx->state.h[k],im1,j)+local_or_halo(&ctx->state.h[k],i,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.u[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],i,jp1)));
              }
              else if (ctx->cfg.useWallNS && (gj == ctx->cfg.Ny-1))
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],ip1,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,jm1)))
                        + (local_or_halo(&ctx->state.h[k],im1,j)+local_or_halo(&ctx->state.h[k],i,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.u[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],i,jm1)));
              }
              else
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],ip1,j))
                        * (SQUARE(local_or_halo(&ctx->state.u[k],ip1,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],ip1,jm1)))
                        + (local_or_halo(&ctx->state.h[k],im1,j)+local_or_halo(&ctx->state.h[k],i,j))
                        * (SQUARE(local_or_halo(&ctx->state.u[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],i,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.u[k],i,jm1)));
              }

              if (ctx->cfg.useWallEW && (gi == 0))
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],i,jp1))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.v[k],i,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],ip1,jp1)))
                        + (local_or_halo(&ctx->state.h[k],i,jm1)+local_or_halo(&ctx->state.h[k],i,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.v[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],ip1,j)));
              }
              else if (ctx->cfg.useWallEW && (gi == ctx->cfg.Nx-1))
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],i,jp1))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.v[k],i,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],im1,jp1)))
                        + (local_or_halo(&ctx->state.h[k],i,jm1)+local_or_halo(&ctx->state.h[k],i,j))
                        * (1.25*SQUARE(local_or_halo(&ctx->state.v[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],im1,j)));
              }
              else
              {
                KEdens += (local_or_halo(&ctx->state.h[k],i,j)+local_or_halo(&ctx->state.h[k],i,jp1))
                        * (SQUARE(local_or_halo(&ctx->state.v[k],i,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],ip1,jp1))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],im1,jp1)))
                        + (local_or_halo(&ctx->state.h[k],i,jm1)+local_or_halo(&ctx->state.h[k],i,j))
                        * (SQUARE(local_or_halo(&ctx->state.v[k],i,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],ip1,j))
                         + 0.25*SQUARE(local_or_halo(&ctx->state.v[k],im1,j)));
              }
              KEdens *= ctx->dx*ctx->dy/12.0;
              break;
            }
            default:
            {
              KEdens = 0;
              break;
            }
          }

          gtot = (ctx->cfg.useTracer && ctx->cfg.useBuoyancy)
               ? ctx->state.geff[k] - ctx->state.b[k].a[i][j]
               : ctx->state.geff[k];
          local_KE += KEdens;
          local_PE += ctx->dx*ctx->dy * gtot * ctx->state.h[k].a[i][j]
                    * (ctx->work.eta_w[k].a[i][j] - 0.5*ctx->state.h[k].a[i][j]);
          local_PE += ctx->dx*ctx->dy * gtot * POW4(ctx->cfg.h0)
                    / SQUARE(ctx->state.h[k].a[i][j]) / 6;
        }
      }
    }

#ifdef PAWSIM_USE_MPI
    MPI_Reduce(&local_KE,&global_KE,1,MPI_DOUBLE,MPI_SUM,0,ctx->dom.comm);
    MPI_Reduce(&local_PE,&global_PE,1,MPI_DOUBLE,MPI_SUM,0,ctx->dom.comm);
    MPI_Reduce(local_Z,global_Z,(int) ctx->cfg.Nlay,MPI_DOUBLE,MPI_SUM,0,ctx->dom.comm);
#else
    global_KE = local_KE;
    global_PE = local_PE;
    memcpy(global_Z,local_Z,ctx->cfg.Nlay*sizeof(double));
#endif

    if (ctx->dom.rank == 0)
    {
      snprintf(outfile,sizeof(outfile),"%s/%s",ctx->outdir,EZFILE);
      file = fopen(outfile,(ctx->cfg.restart || (ctx->n_EZ > 1)) ? "a" : "w");
      if (file == NULL)
      {
        free(local_Z);
        free(global_Z);
        return false;
      }

      fprintf(file,EXP_FMT,ctx->t_next_EZ);
      fprintf(file," %.50e %.50e %.50e ",global_KE,global_PE,global_KE+global_PE);
      for (k = 0; k < ctx->cfg.Nlay; k ++)
      {
        fprintf(file,"%.50e ",global_Z[k]);
      }
      fprintf(file,"\n");
      fclose(file);
    }

    free(local_Z);
    free(global_Z);
    ctx->n_EZ ++;
    ctx->t_next_EZ = ctx->n_EZ * ctx->cfg.savefreqEZ;
  }

  return true;
}

/** Normalize and halo-refresh every term in one budget family. */
static void scale_budget_terms (pawsim_field2d ** terms, uint nterms, uint nlay,
                                real scale, const pawsim_domain * dom)
{
  uint m,k;

  if (terms == NULL)
  {
    return;
  }

  for (m = 0; m < nterms; m ++)
  {
    for (k = 0; k < nlay; k ++)
    {
      scale_field_interior(&terms[m][k],scale);
      pawsim_field2d_exchange_scalar_halo(&terms[m][k],dom);
    }
  }
}

/** Write all layers and terms in one budget family using AWSIM output names. */
static bool write_budget_terms (pawsim_context * ctx, pawsim_field2d ** terms,
                                const char ** names, uint nterms, uint n)
{
  uint m,k;

  if (terms == NULL)
  {
    return true;
  }

  for (m = 0; m < nterms; m ++)
  {
    for (k = 0; k < ctx->cfg.Nlay; k ++)
    {
      if (!write_layer_field(ctx->outdir,names[m],k,n,&terms[m][k],&ctx->dom))
      {
        return false;
      }
    }
  }

  return true;
}

/*
 * Write momentum, thickness, energy, and tracer budget averages whenever their
 * independent AWSIM save clocks are due. Each family may have a different
 * averaging period, so each keeps its own previous-step counter.
 */
bool pawsim_diagnostics_write_budgets_if_due (pawsim_context * ctx, uint n)
{
  static const char * umom_names[PAWSIM_UMOM_NTERMS] =
  {
    OUTN_UMOM_Q,OUTN_UMOM_GRADM,OUTN_UMOM_GRADKE,OUTN_UMOM_DHDT,
    OUTN_UMOM_A2,OUTN_UMOM_A4,OUTN_UMOM_RDRAG,OUTN_UMOM_RSURF,
    OUTN_UMOM_CDBOT,OUTN_UMOM_CDSURF,OUTN_UMOM_WIND,OUTN_UMOM_BUOY,
    OUTN_UMOM_RELAX,OUTN_UMOM_WDIA,OUTN_UMOM_FBARO,OUTN_UMOM_DIAVISC
  };
  static const char * vmom_names[PAWSIM_VMOM_NTERMS] =
  {
    OUTN_VMOM_Q,OUTN_VMOM_GRADM,OUTN_VMOM_GRADKE,OUTN_VMOM_DHDT,
    OUTN_VMOM_A2,OUTN_VMOM_A4,OUTN_VMOM_RDRAG,OUTN_VMOM_RSURF,
    OUTN_VMOM_CDBOT,OUTN_VMOM_CDSURF,OUTN_VMOM_WIND,OUTN_VMOM_BUOY,
    OUTN_VMOM_RELAX,OUTN_VMOM_WDIA,OUTN_VMOM_FBARO,OUTN_VMOM_DIAVISC
  };
  static const char * thic_names[PAWSIM_THIC_NTERMS] =
  {
    OUTN_THIC_ADV,OUTN_THIC_RELAX
  };
  static const char * energy_names[PAWSIM_ENERGY_NTERMS] =
  {
    OUTN_ENERGY_ADV,OUTN_ENERGY_GRADM,OUTN_ENERGY_WIND,
    OUTN_ENERGY_RDRAG,OUTN_ENERGY_RSURF,OUTN_ENERGY_CDBOT,
    OUTN_ENERGY_CDSURF,OUTN_ENERGY_A2,OUTN_ENERGY_A4,
    OUTN_ENERGY_WDIAPE,OUTN_ENERGY_WDIAKE,OUTN_ENERGY_FBARO,
    OUTN_ENERGY_BUOY,OUTN_ENERGY_RELAX,OUTN_ENERGY_DIAVISC
  };
  static const char * trac_names[PAWSIM_TRAC_NTERMS] =
  {
    OUTN_TRAC_ADV,OUTN_TRAC_WDIA,OUTN_TRAC_K2,OUTN_TRAC_K4,
    OUTN_TRAC_DIADIFF,OUTN_TRAC_RELAX
  };

  if (ctx == NULL)
  {
    return false;
  }

  if ((ctx->cfg.savefreqUMom > 0) && (ctx->t >= ctx->t_next_avg_hu))
  {
    real len = (n-ctx->n_prev_avg_hu)*ctx->cfg.dt;
    scale_budget_terms(ctx->work.diag_umom,PAWSIM_UMOM_NTERMS,ctx->cfg.Nlay,1/len,&ctx->dom);
    if (!write_budget_terms(ctx,ctx->work.diag_umom,umom_names,PAWSIM_UMOM_NTERMS,ctx->n_avg_hu))
    {
      return false;
    }
    zero_budget_terms(ctx->work.diag_umom,PAWSIM_UMOM_NTERMS,ctx->cfg.Nlay);
    ctx->n_prev_avg_hu = n;
    ctx->n_avg_hu ++;
    ctx->t_next_avg_hu = ctx->n_avg_hu * ctx->cfg.savefreqUMom;
  }

  if ((ctx->cfg.savefreqVMom > 0) && (ctx->t >= ctx->t_next_avg_hv))
  {
    real len = (n-ctx->n_prev_avg_hv)*ctx->cfg.dt;
    scale_budget_terms(ctx->work.diag_vmom,PAWSIM_VMOM_NTERMS,ctx->cfg.Nlay,1/len,&ctx->dom);
    if (!write_budget_terms(ctx,ctx->work.diag_vmom,vmom_names,PAWSIM_VMOM_NTERMS,ctx->n_avg_hv))
    {
      return false;
    }
    zero_budget_terms(ctx->work.diag_vmom,PAWSIM_VMOM_NTERMS,ctx->cfg.Nlay);
    ctx->n_prev_avg_hv = n;
    ctx->n_avg_hv ++;
    ctx->t_next_avg_hv = ctx->n_avg_hv * ctx->cfg.savefreqVMom;
  }

  if ((ctx->cfg.savefreqThic > 0) && (ctx->t >= ctx->t_next_avg_h))
  {
    real len = (n-ctx->n_prev_avg_h)*ctx->cfg.dt;
    scale_budget_terms(ctx->work.diag_thic,PAWSIM_THIC_NTERMS,ctx->cfg.Nlay,1/len,&ctx->dom);
    if (!write_budget_terms(ctx,ctx->work.diag_thic,thic_names,PAWSIM_THIC_NTERMS,ctx->n_avg_h))
    {
      return false;
    }
    zero_budget_terms(ctx->work.diag_thic,PAWSIM_THIC_NTERMS,ctx->cfg.Nlay);
    ctx->n_prev_avg_h = n;
    ctx->n_avg_h ++;
    ctx->t_next_avg_h = ctx->n_avg_h * ctx->cfg.savefreqThic;
  }

  if ((ctx->cfg.savefreqEnergy > 0) && (ctx->t >= ctx->t_next_avg_e))
  {
    real len = (n-ctx->n_prev_avg_e)*ctx->cfg.dt;
    scale_budget_terms(ctx->work.diag_energy,PAWSIM_ENERGY_NTERMS,ctx->cfg.Nlay,1/len,&ctx->dom);
    if (!write_budget_terms(ctx,ctx->work.diag_energy,energy_names,PAWSIM_ENERGY_NTERMS,ctx->n_avg_e))
    {
      return false;
    }
    zero_budget_terms(ctx->work.diag_energy,PAWSIM_ENERGY_NTERMS,ctx->cfg.Nlay);
    ctx->n_prev_avg_e = n;
    ctx->n_avg_e ++;
    ctx->t_next_avg_e = ctx->n_avg_e * ctx->cfg.savefreqEnergy;
  }

  if (ctx->cfg.useTracer && (ctx->cfg.savefreqTracer > 0) && (ctx->t >= ctx->t_next_avg_b))
  {
    real len = (n-ctx->n_prev_avg_b)*ctx->cfg.dt;
    scale_budget_terms(ctx->work.diag_trac,PAWSIM_TRAC_NTERMS,ctx->cfg.Nlay,1/len,&ctx->dom);
    if (!write_budget_terms(ctx,ctx->work.diag_trac,trac_names,PAWSIM_TRAC_NTERMS,ctx->n_avg_b))
    {
      return false;
    }
    zero_budget_terms(ctx->work.diag_trac,PAWSIM_TRAC_NTERMS,ctx->cfg.Nlay);
    ctx->n_prev_avg_b = n;
    ctx->n_avg_b ++;
    ctx->t_next_avg_b = ctx->n_avg_b * ctx->cfg.savefreqTracer;
  }

  return true;
}

/**
 * pawsim_diagnostics_write_model_state
 *
 * Mirrors AWSIM's prognostic-output convention for PAWSIM fields.
 *
 */
bool pawsim_diagnostics_write_model_state (pawsim_context * ctx, uint n)
{
  uint k;
  char name[MAX_PARAMETER_FILENAME_LENGTH];

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    if (!write_layer_field(ctx->outdir,"U",k,n,&ctx->state.u[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"V",k,n,&ctx->state.v[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"H",k,n,&ctx->state.h[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"W",k,n,&ctx->work.wdia[k],&ctx->dom))
    {
      return false;
    }

    if (ctx->cfg.useTracer
     && !write_layer_field(ctx->outdir,"B",k,n,&ctx->state.b[k],&ctx->dom))
    {
      return false;
    }
  }

  if (!write_layer_field(ctx->outdir,"W",ctx->cfg.Nlay,n,&ctx->work.wdia[ctx->cfg.Nlay],&ctx->dom))
  {
    return false;
  }

  if (ctx->cfg.useRL)
  {
    snprintf(name,sizeof(name),"P_n=%u.dat",n);
    return write_named_field(ctx->outdir,name,&ctx->pi,&ctx->dom);
  }

  return true;
}

/*
 * Write opt-in initialization/work-array diagnostics for MPI regression and
 * porting checks. These are intentionally broader than AWSIM's output set.
 */
bool pawsim_diagnostics_write_initialization (pawsim_context * ctx)
{
  uint k;

  /*
   * These files are not part of AWSIM's normal output set. They are retained as
   * permanent debug/regression hooks because many PAWSIM errors show up first
   * in derived geometry or q-grid quantities before they are obvious in U/V/H.
   */
  if (!write_named_field(ctx->outdir,"init_hhs.dat",&ctx->state.hhs,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_hhb.dat",&ctx->state.hhb,&ctx->dom)
   || !write_named_qfield(ctx->outdir,"init_omegaz.dat",&ctx->state.omega_z,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_hhs_west.dat",&ctx->work.hhs_west,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_hhs_south.dat",&ctx->work.hhs_south,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_hhb_west.dat",&ctx->work.hhb_west,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_hhb_south.dat",&ctx->work.hhb_south,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_Hc.dat",&ctx->work.Hc,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_Hw.dat",&ctx->work.Hw,&ctx->dom)
   || !write_named_field(ctx->outdir,"init_Hs.dat",&ctx->work.Hs,&ctx->dom))
  {
    return false;
  }

  for (k = 0; k < ctx->cfg.Nlay+1; k ++)
  {
    if (!write_layer_field(ctx->outdir,"init_eta",k,0,&ctx->work.eta_w[k],&ctx->dom))
    {
      return false;
    }
  }

  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    if (!write_layer_field(ctx->outdir,"init_hwest",k,0,&ctx->work.h_west[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_hsouth",k,0,&ctx->work.h_south[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_dudt",k,0,&ctx->work.dt_u[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_dvdt",k,0,&ctx->work.dt_v[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_dhdt",k,0,&ctx->work.dt_h[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_zeta",k,0,&ctx->work.zeta[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_hhq",k,0,&ctx->work.hh_q[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_qq",k,0,&ctx->work.qq[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_pp",k,0,&ctx->work.pp[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_KE",k,0,&ctx->work.KE_B[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_MM",k,0,&ctx->work.MM_B[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_alpha",k,0,&ctx->work.alpha[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_beta",k,0,&ctx->work.beta[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_gamma",k,0,&ctx->work.gamma[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_delta",k,0,&ctx->work.delta[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_epsilon",k,0,&ctx->work.epsilon[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_phi",k,0,&ctx->work.phi[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_lambda",k,0,&ctx->work.lambda[k],&ctx->dom)
     || !write_layer_field(ctx->outdir,"init_mu",k,0,&ctx->work.mu[k],&ctx->dom))
    {
      return false;
    }
  }

  return true;
}

/*
 * Write startup output: always the AWSIM-compatible n=0 model state, plus the
 * verbose PAWSIM init diagnostics only when explicitly requested.
 */
bool pawsim_diagnostics_write_initial_outputs (pawsim_context * ctx)
{
  /*
   * The AWSIM-compatible n=0 model state is always written.  The much more
   * verbose init_* work-array dump is opt-in so production-like PAWSIM runs do
   * not accumulate development files by default.
   */
  if (ctx->cfg.writeInitDiagnostics
   && !pawsim_diagnostics_write_initialization(ctx))
  {
    fprintf(stderr,"Unable to write PAWSIM initialization diagnostics\n");
    return false;
  }

  if (!pawsim_diagnostics_write_model_state(ctx,ctx->n_saves))
  {
    fprintf(stderr,"Unable to write model initial state\n");
    return false;
  }

  ctx->n_saves ++;
  return true;
}
