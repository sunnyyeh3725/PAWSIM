/**
 * work.c
 *
 * PAWSIM work-array allocation and state-to-work halo preparation.
 *
 * This file contains the ported AWSIM stencil machinery: face thicknesses,
 * interface heights, q-grid potential vorticity, Bernoulli functions,
 * viscosity, momentum tendencies, thickness tendencies, and forcing masks.
 * Most arrays are stored with the same staggered-grid interpretation as AWSIM:
 * h-like quantities at cell centers, u at west/east faces, v at south/north
 * faces, and q/PV quantities at cell corners.
 *
 */
#include "work.h"

#include <math.h>

/** First global index for a rank tile, matching domain.c's decomposition. */
static uint block_start_rank (uint N, uint coord, uint dim)
{
  return (N*coord) / dim;
}

/** Local tile size for the rank-coordinate/block-decomposition pair. */
static uint block_size_rank (uint N, uint coord, uint dim)
{
  return block_start_rank(N,coord+1,dim) - block_start_rank(N,coord,dim);
}

/** Allocate one local field per model layer. */
static bool alloc_layer_fields (pawsim_field2d ** fields, uint Nlay,
                                const pawsim_domain * dom)
{
  uint k;

  *fields = calloc(Nlay,sizeof(pawsim_field2d));
  if (*fields == NULL)
  {
    return false;
  }

  for (k = 0; k < Nlay; k ++)
  {
    if (!pawsim_field2d_alloc(&(*fields)[k],dom->nx,dom->ny,dom->nghost))
    {
      return false;
    }
  }

  return true;
}

/** Free a per-layer field array. */
static void free_layer_fields (pawsim_field2d * fields, uint Nlay)
{
  uint k;

  if (fields == NULL)
  {
    return;
  }

  for (k = 0; k < Nlay; k ++)
  {
    pawsim_field2d_free(&fields[k]);
  }
  free(fields);
}

/** Allocate diagnostic-budget fields indexed as terms[term][layer]. */
static bool alloc_term_fields (pawsim_field2d *** terms, uint nterms, uint nlay,
                               bool (*term_enabled)(uint),
                               const pawsim_domain * dom)
{
  uint m;

  *terms = NULL;
  if ((nterms == 0) || (nlay == 0))
  {
    return true;
  }

  *terms = calloc(nterms,sizeof(pawsim_field2d *));
  if (*terms == NULL)
  {
    return false;
  }

  for (m = 0; m < nterms; m ++)
  {
    if ((term_enabled != NULL) && !term_enabled(m))
    {
      continue;
    }
    if (!alloc_layer_fields(&(*terms)[m],nlay,dom))
    {
      return false;
    }
  }

  return true;
}

static bool umom_term_enabled (uint term)
{
  return (term != PAWSIM_UMOM_BUOY)
      && (term != PAWSIM_UMOM_RELAX)
      && (term != PAWSIM_UMOM_WDIA)
      && (term != PAWSIM_UMOM_FBARO)
      && (term != PAWSIM_UMOM_DIAVISC);
}

static bool vmom_term_enabled (uint term)
{
  return (term != PAWSIM_VMOM_BUOY)
      && (term != PAWSIM_VMOM_RELAX)
      && (term != PAWSIM_VMOM_WDIA)
      && (term != PAWSIM_VMOM_FBARO)
      && (term != PAWSIM_VMOM_DIAVISC);
}

static bool thic_term_enabled (uint term)
{
  return term != PAWSIM_THIC_RELAX;
}

static bool energy_term_enabled (uint term)
{
  return (term != PAWSIM_ENERGY_A2)
      && (term != PAWSIM_ENERGY_WDIAPE)
      && (term != PAWSIM_ENERGY_WDIAKE)
      && (term != PAWSIM_ENERGY_FBARO)
      && (term != PAWSIM_ENERGY_BUOY)
      && (term != PAWSIM_ENERGY_RELAX)
      && (term != PAWSIM_ENERGY_DIAVISC);
}

/** Free diagnostic-budget fields allocated by alloc_term_fields(). */
static void free_term_fields (pawsim_field2d ** terms, uint nterms, uint nlay)
{
  uint m;

  if (terms == NULL)
  {
    return;
  }

  for (m = 0; m < nterms; m ++)
  {
    free_layer_fields(terms[m],nlay);
  }
  free(terms);
}

/** Zero every field in one diagnostic-budget family. */
static void zero_term_fields (pawsim_field2d ** terms, uint nterms, uint nlay)
{
  uint m,k;

  if (terms == NULL)
  {
    return;
  }

  for (m = 0; m < nterms; m ++)
  {
    if (terms[m] == NULL)
    {
      continue;
    }
    for (k = 0; k < nlay; k ++)
    {
      pawsim_field2d_zero(&terms[m][k]);
    }
  }
}

/** Add a term to an optional diagnostic budget array. */
static void diag_add (pawsim_field2d ** terms, uint term, uint k, uint i, uint j,
                      real value)
{
  if ((terms != NULL) && (terms[term] != NULL))
  {
    terms[term][k].a[i][j] += value;
  }
}

/** Copy owned cells between equal-sized local fields, leaving halos untouched. */
static void copy_interior (pawsim_field2d * dst, const pawsim_field2d * src)
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

/** Compute and halo-refresh h*u and h*v for one layer. */
static void calc_layer_mass_fluxes (pawsim_work * work, uint k,
                                    const pawsim_domain * dom)
{
  uint i,j;

  pawsim_field2d_zero(&work->huu[k]);
  pawsim_field2d_zero(&work->hvv[k]);

  /*
   * Store h*u and h*v per layer. Earlier versions rebuilt these twice; keeping
   * huu/hvv layer-indexed costs memory but avoids redundant work and makes the
   * thickness tendency use the same mass fluxes as the momentum RHS.
   */
  for (i = 1; i < work->h_w[k].sx-1; i ++)
  {
    for (j = 1; j < work->h_w[k].sy-1; j ++)
    {
      work->huu[k].a[i][j] = work->u_w[k].a[i][j] * work->h_west[k].a[i][j];
      work->hvv[k].a[i][j] = work->v_w[k].a[i][j] * work->h_south[k].a[i][j];
    }
  }

  pawsim_field2d_exchange_internal_halo(&work->huu[k],dom);
  pawsim_field2d_exchange_internal_halo(&work->hvv[k],dom);
}

/** Build active-tracer buoyancy-gradient work arrays for one layer. */
static void calc_layer_active_buoyancy_work (pawsim_work * work, uint k,
                                             real dx, real dy)
{
  uint i,j;

  pawsim_field2d_zero(&work->hz);
  pawsim_field2d_zero(&work->db_dx);
  pawsim_field2d_zero(&work->db_dy);

  /*
   * AWSIM treats the prognostic tracer as a layer buoyancy anomaly when
   * useBuoyancy is enabled. hz is h times the layer mid-depth and multiplies
   * the lateral buoyancy gradient in the momentum equation.
   */
  for (i = 1; i < work->b_w[k].sx-1; i ++)
  {
    for (j = 1; j < work->b_w[k].sy-1; j ++)
    {
      work->db_dx.a[i][j] = (work->b_w[k].a[i][j]-work->b_w[k].a[i-1][j]) / dx;
      work->db_dy.a[i][j] = (work->b_w[k].a[i][j]-work->b_w[k].a[i][j-1]) / dy;
      work->hz.a[i][j] = work->h_w[k].a[i][j]
                        * 0.5*(work->eta_w[k].a[i][j]+work->eta_w[k+1].a[i][j]);
    }
  }
}

/*
 * Allocate all derived fields used by AWSIM's tendency and diagnostic
 * calculations. Keeping these in one work object avoids hidden globals and
 * makes rank-local ownership explicit.
 */
bool pawsim_work_alloc (pawsim_work * work, const pawsim_config * cfg,
                        const pawsim_domain * dom)
{
  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return false;
  }

  memset(work,0,sizeof(*work));
  work->Nlay = cfg->Nlay;

  if (!alloc_layer_fields(&work->u_w,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->v_w,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->h_w,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->b_w,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->dt_u,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->dt_v,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->dt_h,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->dt_b,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->h_west,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->h_south,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hFsurf_west,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hFsurf_south,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hFbot_west,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hFbot_south,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hh_q,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->zeta,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->qq,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->alpha,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->beta,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->gamma,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->delta,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->epsilon,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->phi,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->lambda,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->mu,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->pp,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->KE_B,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->MM_B,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->eta_w,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->wdia,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->wdia_u,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->wdia_v,cfg->Nlay+1,dom)
   || !pawsim_field2d_alloc(&work->hhs_w,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hhb_w,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hhs_west,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hhs_south,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hhb_west,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hhb_south,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->Hc,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->Hw,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->Hs,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->taux_w,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->tauy_w,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->usq_surf,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->vsq_surf,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->uabs_surf,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->usq_bot,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->vsq_bot,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->uabs_bot,dom->nx,dom->ny,dom->nghost)
   || !alloc_layer_fields(&work->Fdia_u,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->Fdia_v,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->Fdia_b,cfg->Nlay+1,dom)
   || !alloc_layer_fields(&work->gprime,cfg->Nlay,dom)
   || !pawsim_field2d_alloc(&work->DD_T,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->DD_S,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->DD4_T,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->DD4_S,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->A2_h,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->A2_q,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->A4sqrt_h,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->A4sqrt_q,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->UU4,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->VV4,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->BB4,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hz,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->d2hx,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->d2hy,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->d2bx,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->d2by,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->dh_dx,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->dh_dy,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->db_dx,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->db_dy,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->udb_dx,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->vdb_dy,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hub,dom->nx,dom->ny,dom->nghost)
   || !pawsim_field2d_alloc(&work->hvb,dom->nx,dom->ny,dom->nghost)
   || !alloc_layer_fields(&work->huu,cfg->Nlay,dom)
   || !alloc_layer_fields(&work->hvv,cfg->Nlay,dom)
   || ((cfg->savefreqUMom > 0)
    && !alloc_term_fields(&work->diag_umom,PAWSIM_UMOM_NTERMS,cfg->Nlay,
                          umom_term_enabled,dom))
   || ((cfg->savefreqVMom > 0)
    && !alloc_term_fields(&work->diag_vmom,PAWSIM_VMOM_NTERMS,cfg->Nlay,
                          vmom_term_enabled,dom))
   || ((cfg->savefreqThic > 0)
    && !alloc_term_fields(&work->diag_thic,PAWSIM_THIC_NTERMS,cfg->Nlay,
                          thic_term_enabled,dom))
   || ((cfg->savefreqEnergy > 0)
    && !alloc_term_fields(&work->diag_energy,PAWSIM_ENERGY_NTERMS,cfg->Nlay,
                          energy_term_enabled,dom))
   || ((cfg->savefreqTracer > 0)
    && !alloc_term_fields(&work->diag_trac,PAWSIM_TRAC_NTERMS,cfg->Nlay,
                          NULL,dom)))
  {
    pawsim_work_free(work);
    return false;
  }

  return true;
}

/** Free every work and diagnostic-budget array; safe after partial allocation. */
void pawsim_work_free (pawsim_work * work)
{
  if (work == NULL)
  {
    return;
  }

  free_layer_fields(work->u_w,work->Nlay);
  free_layer_fields(work->v_w,work->Nlay);
  free_layer_fields(work->h_w,work->Nlay);
  free_layer_fields(work->b_w,work->Nlay);
  free_layer_fields(work->dt_u,work->Nlay);
  free_layer_fields(work->dt_v,work->Nlay);
  free_layer_fields(work->dt_h,work->Nlay);
  free_layer_fields(work->dt_b,work->Nlay);
  free_layer_fields(work->h_west,work->Nlay);
  free_layer_fields(work->h_south,work->Nlay);
  free_layer_fields(work->hFsurf_west,work->Nlay);
  free_layer_fields(work->hFsurf_south,work->Nlay);
  free_layer_fields(work->hFbot_west,work->Nlay);
  free_layer_fields(work->hFbot_south,work->Nlay);
  free_layer_fields(work->hh_q,work->Nlay);
  free_layer_fields(work->zeta,work->Nlay);
  free_layer_fields(work->qq,work->Nlay);
  free_layer_fields(work->alpha,work->Nlay);
  free_layer_fields(work->beta,work->Nlay);
  free_layer_fields(work->gamma,work->Nlay);
  free_layer_fields(work->delta,work->Nlay);
  free_layer_fields(work->epsilon,work->Nlay);
  free_layer_fields(work->phi,work->Nlay);
  free_layer_fields(work->lambda,work->Nlay);
  free_layer_fields(work->mu,work->Nlay);
  free_layer_fields(work->pp,work->Nlay);
  free_layer_fields(work->KE_B,work->Nlay);
  free_layer_fields(work->MM_B,work->Nlay);
  free_layer_fields(work->eta_w,work->Nlay+1);
  free_layer_fields(work->wdia,work->Nlay+1);
  free_layer_fields(work->wdia_u,work->Nlay+1);
  free_layer_fields(work->wdia_v,work->Nlay+1);
  pawsim_field2d_free(&work->hhs_w);
  pawsim_field2d_free(&work->hhb_w);
  pawsim_field2d_free(&work->hhs_west);
  pawsim_field2d_free(&work->hhs_south);
  pawsim_field2d_free(&work->hhb_west);
  pawsim_field2d_free(&work->hhb_south);
  pawsim_field2d_free(&work->Hc);
  pawsim_field2d_free(&work->Hw);
  pawsim_field2d_free(&work->Hs);
  pawsim_field2d_free(&work->taux_w);
  pawsim_field2d_free(&work->tauy_w);
  pawsim_field2d_free(&work->usq_surf);
  pawsim_field2d_free(&work->vsq_surf);
  pawsim_field2d_free(&work->uabs_surf);
  pawsim_field2d_free(&work->usq_bot);
  pawsim_field2d_free(&work->vsq_bot);
  pawsim_field2d_free(&work->uabs_bot);
  free_layer_fields(work->Fdia_u,work->Nlay+1);
  free_layer_fields(work->Fdia_v,work->Nlay+1);
  free_layer_fields(work->Fdia_b,work->Nlay+1);
  free_layer_fields(work->gprime,work->Nlay);
  pawsim_field2d_free(&work->DD_T);
  pawsim_field2d_free(&work->DD_S);
  pawsim_field2d_free(&work->DD4_T);
  pawsim_field2d_free(&work->DD4_S);
  pawsim_field2d_free(&work->A2_h);
  pawsim_field2d_free(&work->A2_q);
  pawsim_field2d_free(&work->A4sqrt_h);
  pawsim_field2d_free(&work->A4sqrt_q);
  pawsim_field2d_free(&work->UU4);
  pawsim_field2d_free(&work->VV4);
  pawsim_field2d_free(&work->BB4);
  pawsim_field2d_free(&work->hz);
  pawsim_field2d_free(&work->d2hx);
  pawsim_field2d_free(&work->d2hy);
  pawsim_field2d_free(&work->d2bx);
  pawsim_field2d_free(&work->d2by);
  pawsim_field2d_free(&work->dh_dx);
  pawsim_field2d_free(&work->dh_dy);
  pawsim_field2d_free(&work->db_dx);
  pawsim_field2d_free(&work->db_dy);
  pawsim_field2d_free(&work->udb_dx);
  pawsim_field2d_free(&work->vdb_dy);
  pawsim_field2d_free(&work->hub);
  pawsim_field2d_free(&work->hvb);
  free_layer_fields(work->huu,work->Nlay);
  free_layer_fields(work->hvv,work->Nlay);
  free_term_fields(work->diag_umom,PAWSIM_UMOM_NTERMS,work->Nlay);
  free_term_fields(work->diag_vmom,PAWSIM_VMOM_NTERMS,work->Nlay);
  free_term_fields(work->diag_thic,PAWSIM_THIC_NTERMS,work->Nlay);
  free_term_fields(work->diag_energy,PAWSIM_ENERGY_NTERMS,work->Nlay);
  free_term_fields(work->diag_trac,PAWSIM_TRAC_NTERMS,work->Nlay);
  memset(work,0,sizeof(*work));
}

/** Zero all work arrays and currently allocated diagnostic accumulators. */
void pawsim_work_zero (pawsim_work * work)
{
  uint k;

  if (work == NULL)
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->u_w[k]);
    pawsim_field2d_zero(&work->v_w[k]);
    pawsim_field2d_zero(&work->h_w[k]);
    pawsim_field2d_zero(&work->b_w[k]);
    pawsim_field2d_zero(&work->dt_u[k]);
    pawsim_field2d_zero(&work->dt_v[k]);
    pawsim_field2d_zero(&work->dt_h[k]);
    pawsim_field2d_zero(&work->dt_b[k]);
    pawsim_field2d_zero(&work->h_west[k]);
    pawsim_field2d_zero(&work->h_south[k]);
    pawsim_field2d_zero(&work->hFsurf_west[k]);
    pawsim_field2d_zero(&work->hFsurf_south[k]);
    pawsim_field2d_zero(&work->hFbot_west[k]);
    pawsim_field2d_zero(&work->hFbot_south[k]);
    pawsim_field2d_zero(&work->hh_q[k]);
    pawsim_field2d_zero(&work->zeta[k]);
    pawsim_field2d_zero(&work->qq[k]);
    pawsim_field2d_zero(&work->alpha[k]);
    pawsim_field2d_zero(&work->beta[k]);
    pawsim_field2d_zero(&work->gamma[k]);
    pawsim_field2d_zero(&work->delta[k]);
    pawsim_field2d_zero(&work->epsilon[k]);
    pawsim_field2d_zero(&work->phi[k]);
    pawsim_field2d_zero(&work->lambda[k]);
    pawsim_field2d_zero(&work->mu[k]);
    pawsim_field2d_zero(&work->pp[k]);
    pawsim_field2d_zero(&work->KE_B[k]);
    pawsim_field2d_zero(&work->MM_B[k]);
  }
  for (k = 0; k < work->Nlay+1; k ++)
  {
    pawsim_field2d_zero(&work->eta_w[k]);
    pawsim_field2d_zero(&work->wdia[k]);
    pawsim_field2d_zero(&work->wdia_u[k]);
    pawsim_field2d_zero(&work->wdia_v[k]);
  }
  pawsim_field2d_zero(&work->hhs_w);
  pawsim_field2d_zero(&work->hhb_w);
  pawsim_field2d_zero(&work->hhs_west);
  pawsim_field2d_zero(&work->hhs_south);
  pawsim_field2d_zero(&work->hhb_west);
  pawsim_field2d_zero(&work->hhb_south);
  pawsim_field2d_zero(&work->Hc);
  pawsim_field2d_zero(&work->Hw);
  pawsim_field2d_zero(&work->Hs);
  pawsim_field2d_zero(&work->taux_w);
  pawsim_field2d_zero(&work->tauy_w);
  pawsim_field2d_zero(&work->usq_surf);
  pawsim_field2d_zero(&work->vsq_surf);
  pawsim_field2d_zero(&work->uabs_surf);
  pawsim_field2d_zero(&work->usq_bot);
  pawsim_field2d_zero(&work->vsq_bot);
  pawsim_field2d_zero(&work->uabs_bot);
  for (k = 0; k < work->Nlay+1; k ++)
  {
    pawsim_field2d_zero(&work->Fdia_u[k]);
    pawsim_field2d_zero(&work->Fdia_v[k]);
    pawsim_field2d_zero(&work->Fdia_b[k]);
  }
  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->gprime[k]);
  }
  pawsim_field2d_zero(&work->DD_T);
  pawsim_field2d_zero(&work->DD_S);
  pawsim_field2d_zero(&work->DD4_T);
  pawsim_field2d_zero(&work->DD4_S);
  pawsim_field2d_zero(&work->A2_h);
  pawsim_field2d_zero(&work->A2_q);
  pawsim_field2d_zero(&work->A4sqrt_h);
  pawsim_field2d_zero(&work->A4sqrt_q);
  pawsim_field2d_zero(&work->UU4);
  pawsim_field2d_zero(&work->VV4);
  pawsim_field2d_zero(&work->BB4);
  pawsim_field2d_zero(&work->hz);
  pawsim_field2d_zero(&work->d2hx);
  pawsim_field2d_zero(&work->d2hy);
  pawsim_field2d_zero(&work->d2bx);
  pawsim_field2d_zero(&work->d2by);
  pawsim_field2d_zero(&work->dh_dx);
  pawsim_field2d_zero(&work->dh_dy);
  pawsim_field2d_zero(&work->db_dx);
  pawsim_field2d_zero(&work->db_dy);
  pawsim_field2d_zero(&work->udb_dx);
  pawsim_field2d_zero(&work->vdb_dy);
  pawsim_field2d_zero(&work->hub);
  pawsim_field2d_zero(&work->hvb);
  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->huu[k]);
    pawsim_field2d_zero(&work->hvv[k]);
  }
  zero_term_fields(work->diag_umom,PAWSIM_UMOM_NTERMS,work->Nlay);
  zero_term_fields(work->diag_vmom,PAWSIM_VMOM_NTERMS,work->Nlay);
  zero_term_fields(work->diag_thic,PAWSIM_THIC_NTERMS,work->Nlay);
  zero_term_fields(work->diag_energy,PAWSIM_ENERGY_NTERMS,work->Nlay);
  zero_term_fields(work->diag_trac,PAWSIM_TRAC_NTERMS,work->Nlay);
}

/*
 * Copy prognostic state into work arrays and immediately refresh halos. This
 * is the MPI analogue of AWSIM copying variables into ghosted work arrays.
 */
void pawsim_work_load_state (pawsim_work * work, const pawsim_state * state,
                             const pawsim_domain * dom)
{
  uint k;

  if ((work == NULL) || (state == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    copy_interior(&work->u_w[k],&state->u[k]);
    copy_interior(&work->v_w[k],&state->v[k]);
    copy_interior(&work->h_w[k],&state->h[k]);
    copy_interior(&work->b_w[k],&state->b[k]);
    pawsim_field2d_exchange_u_halo(&work->u_w[k],dom);
    pawsim_field2d_exchange_v_halo(&work->v_w[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->h_w[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->b_w[k],dom);
  }
}

/*
 * Calculate layer thickness on western and southern C-grid faces. With no
 * velocity input this is the centered static/topographic interpolation; with
 * velocity input it selects AL81, HK83, UP3, or KT00 as AWSIM does.
 */
static void calc_face_thickness_layer (pawsim_field2d * hh,
                                       pawsim_field2d * h_west,
                                       pawsim_field2d * h_south,
                                       pawsim_field2d * uu,
                                       pawsim_field2d * vv,
                                       pawsim_work * work,
                                       const pawsim_config * cfg)
{
  uint i,j;
  uint imin = 1;
  uint jmin = 1;
  uint imax = hh->sx - 1;
  uint jmax = hh->sy - 1;
  uint scheme = cfg->thicknessScheme;
  real hh_xm,hh_xp,hh_ym,hh_yp;
  bool vel_flag = (uu != NULL) && (vv != NULL);

  /*
   * Without velocities, use centered interpolation. Static topography and the
   * rigid-lid column thickness use this branch.
   */
  if (!vel_flag)
  {
    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        h_west->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i-1][j]);
        h_south->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i][j-1]);
      }
    }
    return;
  }

  /** AL81 uses arithmetic face averages. */
  if (scheme == THICKNESS_AL81)
  {
    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        h_west->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i-1][j]);
        h_south->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i][j-1]);
      }
    }
  }

  /** HK83 uses the wider weighted average from AWSIM. */
  if (scheme == THICKNESS_HK83)
  {
    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        h_south->a[i][j] = (1.0/3) * ( hh->a[i][j]+hh->a[i][j-1]
                         + 0.25 * (hh->a[i+1][j]+hh->a[i-1][j]
                                  + hh->a[i+1][j-1]+hh->a[i-1][j-1]) );
        h_west->a[i][j] = (1.0/3) * ( hh->a[i][j]+hh->a[i-1][j]
                        + 0.25 * (hh->a[i][j+1]+hh->a[i-1][j+1]
                                 + hh->a[i][j-1]+hh->a[i-1][j-1]) );
      }
    }
  }

  /** UP3 applies the third-order upwind correction selected by face velocity. */
  if (vel_flag && (scheme == THICKNESS_UP3))
  {
    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        work->d2hx.a[i][j] = hh->a[i+1][j] - 2*hh->a[i][j] + hh->a[i-1][j];
        work->d2hy.a[i][j] = hh->a[i][j+1] - 2*hh->a[i][j] + hh->a[i][j-1];
      }
    }

    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        h_south->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i][j-1]);
        h_south->a[i][j] -= (1/6.0) * ((vv->a[i][j] > 0) ? work->d2hy.a[i][j-1] : work->d2hy.a[i][j]);
        h_west->a[i][j] = 0.5 * (hh->a[i][j]+hh->a[i-1][j]);
        h_west->a[i][j] -= (1/6.0) * ((uu->a[i][j] > 0) ? work->d2hx.a[i-1][j] : work->d2hx.a[i][j]);
      }
    }
  }

  /** KT00 reconstructs limited left/right interface states using minmod. */
  if (vel_flag && (scheme == THICKNESS_KT00))
  {
    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        work->dh_dx.a[i][j] = minmod(cfg->KT00_sigma * (hh->a[i+1][j]-hh->a[i][j]),
                                     0.5 * (hh->a[i+1][j]-hh->a[i-1][j]),
                                     cfg->KT00_sigma * (hh->a[i][j]-hh->a[i-1][j]));
        work->dh_dy.a[i][j] = minmod(cfg->KT00_sigma * (hh->a[i][j+1]-hh->a[i][j]),
                                     0.5 * (hh->a[i][j+1]-hh->a[i][j-1]),
                                     cfg->KT00_sigma * (hh->a[i][j]-hh->a[i][j-1]));
      }
    }

    for (i = imin; i < imax; i ++)
    {
      for (j = jmin; j < jmax; j ++)
      {
        hh_xm = hh->a[i-1][j] + 0.5*work->dh_dx.a[i-1][j];
        hh_xp = hh->a[i][j] - 0.5*work->dh_dx.a[i][j];
        h_west->a[i][j] = (uu->a[i][j] > 0) ? hh_xm : hh_xp;

        hh_ym = hh->a[i][j-1] + 0.5*work->dh_dy.a[i][j-1];
        hh_yp = hh->a[i][j] + 0.5*work->dh_dy.a[i][j];
        h_south->a[i][j] = (vv->a[i][j] > 0) ? hh_ym : hh_yp;
      }
    }
  }
}

/*
 * Gather one local layer into a flat rank-0 global array. This is used only for
 * face-thickness calculations that need exact AWSIM wall-stencil behavior.
 */
static bool gather_global_layer (real * global,
                                 const pawsim_field2d * field,
                                 const pawsim_domain * dom,
                                 int tag_base)
{
#ifdef PAWSIM_USE_MPI
  if (dom->rank == 0)
  {
    int src;
    uint i,j,g;

    g = field->nghost;
    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        global[(dom->i0+i)*dom->Ny + (dom->j0+j)] = field->a[i+g][j+g];
      }
    }

    for (src = 1; src < dom->size; src ++)
    {
      uint meta[4];
      real * buf = NULL;

      MPI_Recv(meta,4,MPI_UNSIGNED,src,tag_base,dom->comm,MPI_STATUS_IGNORE);
      buf = malloc((size_t) meta[2]*meta[3]*sizeof(real));
      if (buf == NULL)
      {
        return false;
      }
      MPI_Recv(buf,(int) (meta[2]*meta[3]),MPI_DOUBLE,src,tag_base+1,
               dom->comm,MPI_STATUS_IGNORE);
      for (i = 0; i < meta[2]; i ++)
      {
        for (j = 0; j < meta[3]; j ++)
        {
          global[(meta[0]+i)*dom->Ny + (meta[1]+j)] = buf[i*meta[3]+j];
        }
      }
      free(buf);
    }
  }
  else
  {
    uint i,j,g;
    uint meta[4];
    real * buf = malloc((size_t) field->nx*field->ny*sizeof(real));

    if (buf == NULL)
    {
      return false;
    }
    meta[0] = dom->i0;
    meta[1] = dom->j0;
    meta[2] = field->nx;
    meta[3] = field->ny;
    g = field->nghost;
    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        buf[i*field->ny+j] = field->a[i+g][j+g];
      }
    }
    MPI_Send(meta,4,MPI_UNSIGNED,0,tag_base,dom->comm);
    MPI_Send(buf,(int) (field->nx*field->ny),MPI_DOUBLE,0,tag_base+1,dom->comm);
    free(buf);
  }
  return true;
#else
  uint i,j,g;
  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      global[(dom->i0+i)*dom->Ny + (dom->j0+j)] = field->a[i+g][j+g];
    }
  }
  (void) tag_base;
  return true;
#endif
}

/** Scatter a flat rank-0 global layer back to rank-local owned cells. */
static bool scatter_global_layer (pawsim_field2d * field,
                                  const real * global,
                                  const pawsim_domain * dom,
                                  int tag_base)
{
#ifdef PAWSIM_USE_MPI
  if (dom->rank == 0)
  {
    int rank;
    uint i,j,g;

    g = field->nghost;
    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        field->a[i+g][j+g] = global[(dom->i0+i)*dom->Ny + (dom->j0+j)];
      }
    }

    for (rank = 1; rank < dom->size; rank ++)
    {
      int coords[2];
      uint i0,j0,nx,ny;
      real * buf = NULL;

      MPI_Cart_coords(dom->comm,rank,2,coords);
      i0 = block_start_rank(dom->Nx,(uint) coords[0],(uint) dom->dims[0]);
      j0 = block_start_rank(dom->Ny,(uint) coords[1],(uint) dom->dims[1]);
      nx = block_size_rank(dom->Nx,(uint) coords[0],(uint) dom->dims[0]);
      ny = block_size_rank(dom->Ny,(uint) coords[1],(uint) dom->dims[1]);
      buf = malloc((size_t) nx*ny*sizeof(real));
      if (buf == NULL)
      {
        return false;
      }
      for (i = 0; i < nx; i ++)
      {
        for (j = 0; j < ny; j ++)
        {
          buf[i*ny+j] = global[(i0+i)*dom->Ny + (j0+j)];
        }
      }
      MPI_Send(buf,(int) (nx*ny),MPI_DOUBLE,rank,tag_base,dom->comm);
      free(buf);
    }
  }
  else
  {
    uint i,j,g,p = 0;
    real * buf = malloc((size_t) field->nx*field->ny*sizeof(real));

    if (buf == NULL)
    {
      return false;
    }
    MPI_Recv(buf,(int) (field->nx*field->ny),MPI_DOUBLE,0,tag_base,
             dom->comm,MPI_STATUS_IGNORE);
    g = field->nghost;
    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        field->a[i+g][j+g] = buf[p++];
      }
    }
    free(buf);
  }
  return true;
#else
  uint i,j,g;
  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      field->a[i+g][j+g] = global[(dom->i0+i)*dom->Ny + (dom->j0+j)];
    }
  }
  (void) tag_base;
  return true;
#endif
}

/*
 * Serial AWSIM-compatible face-thickness calculation on a global field. This
 * avoids ambiguous ghost construction at MPI physical-wall corners for the
 * pressure solve and diagnostics that must use identical mass fluxes.
 */
static void calc_face_thickness_global_no_ghost (const real * hh,
                                                 const real * uu,
                                                 const real * vv,
                                                 real * h_west,
                                                 real * h_south,
                                                 const pawsim_config * cfg)
{
  uint i,j,Nx,Ny;
  uint scheme = cfg->thicknessScheme;
  bool vel_flag = (uu != NULL) && (vv != NULL);

  Nx = cfg->Nx;
  Ny = cfg->Ny;

#define G(arr,ii,jj) ((arr)[((ii)%Nx)*Ny + ((jj)%Ny)])
#define S(arr,ii,jj,val) ((arr)[(ii)*Ny + (jj)] = (val))

  if ((scheme == THICKNESS_AL81) || !vel_flag)
  {
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        S(h_west,i,j,0.5*(G(hh,i,j)+G(hh,im1,j)));
        S(h_south,i,j,0.5*(G(hh,i,j)+G(hh,i,jm1)));
      }
    }
  }

  if (scheme == THICKNESS_HK83)
  {
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      uint ip1 = (i+1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        uint jp1 = (j+1) % Ny;

        if (cfg->useWallEW && (i == 0))
        {
          S(h_south,i,j,(1.0/3) * (1.25*(G(hh,i,j)+G(hh,i,jm1))
                         + 0.25*(G(hh,ip1,j)+G(hh,ip1,jm1))));
        }
        else if (cfg->useWallEW && (i == Nx-1))
        {
          S(h_south,i,j,(1.0/3) * (1.25*(G(hh,i,j)+G(hh,i,jm1))
                         + 0.25*(G(hh,im1,j)+G(hh,im1,jm1))));
        }
        else
        {
          S(h_south,i,j,(1.0/3) * (G(hh,i,j)+G(hh,i,jm1)
                         + 0.25*(G(hh,ip1,j)+G(hh,im1,j)
                                + G(hh,ip1,jm1)+G(hh,im1,jm1))));
        }

        if (cfg->useWallNS && (j == 0))
        {
          S(h_west,i,j,(1.0/3) * (1.25*(G(hh,i,j)+G(hh,im1,j))
                        + 0.25*(G(hh,i,jp1)+G(hh,im1,jp1))));
        }
        else if (cfg->useWallNS && (j == Ny-1))
        {
          S(h_west,i,j,(1.0/3) * (1.25*(G(hh,i,j)+G(hh,im1,j))
                        + 0.25*(G(hh,i,jm1)+G(hh,im1,jm1))));
        }
        else
        {
          S(h_west,i,j,(1.0/3) * (G(hh,i,j)+G(hh,im1,j)
                        + 0.25*(G(hh,i,jp1)+G(hh,im1,jp1)
                               + G(hh,i,jm1)+G(hh,im1,jm1))));
        }
      }
    }
  }

  if (vel_flag && (scheme == THICKNESS_UP3))
  {
    real * d2hx = calloc((size_t) Nx*Ny,sizeof(real));
    real * d2hy = calloc((size_t) Nx*Ny,sizeof(real));

    if ((d2hx == NULL) || (d2hy == NULL))
    {
      free(d2hx);
      free(d2hy);
      return;
    }
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      uint ip1 = (i+1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        uint jp1 = (j+1) % Ny;
        S(d2hx,i,j,G(hh,ip1,j)-2*G(hh,i,j)+G(hh,im1,j));
        S(d2hy,i,j,G(hh,i,jp1)-2*G(hh,i,j)+G(hh,i,jm1));
      }
    }
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        S(h_south,i,j,0.5*(G(hh,i,j)+G(hh,i,jm1))
                     - (1/6.0) * ((G(vv,i,j) > 0) ? G(d2hy,i,jm1) : G(d2hy,i,j)));
        S(h_west,i,j,0.5*(G(hh,i,j)+G(hh,im1,j))
                    - (1/6.0) * ((G(uu,i,j) > 0) ? G(d2hx,im1,j) : G(d2hx,i,j)));
      }
    }
    free(d2hx);
    free(d2hy);
  }

  if (vel_flag && (scheme == THICKNESS_KT00))
  {
    real * dh_dx = calloc((size_t) Nx*Ny,sizeof(real));
    real * dh_dy = calloc((size_t) Nx*Ny,sizeof(real));

    if ((dh_dx == NULL) || (dh_dy == NULL))
    {
      free(dh_dx);
      free(dh_dy);
      return;
    }
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      uint ip1 = (i+1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        uint jp1 = (j+1) % Ny;
        S(dh_dx,i,j,minmod(cfg->KT00_sigma*(G(hh,ip1,j)-G(hh,i,j)),
                           0.5*(G(hh,ip1,j)-G(hh,im1,j)),
                           cfg->KT00_sigma*(G(hh,i,j)-G(hh,im1,j))));
        S(dh_dy,i,j,minmod(cfg->KT00_sigma*(G(hh,i,jp1)-G(hh,i,j)),
                           0.5*(G(hh,i,jp1)-G(hh,i,jm1)),
                           cfg->KT00_sigma*(G(hh,i,j)-G(hh,i,jm1))));
      }
    }
    for (i = 0; i < Nx; i ++)
    {
      uint im1 = (i+Nx-1) % Nx;
      for (j = 0; j < Ny; j ++)
      {
        uint jm1 = (j+Ny-1) % Ny;
        real hh_xm = G(hh,im1,j) + 0.5*G(dh_dx,im1,j);
        real hh_xp = G(hh,i,j) - 0.5*G(dh_dx,i,j);
        real hh_ym = G(hh,i,jm1) + 0.5*G(dh_dy,i,jm1);
        real hh_yp = G(hh,i,j) + 0.5*G(dh_dy,i,j);
        S(h_west,i,j,(G(uu,i,j) > 0) ? hh_xm : hh_xp);
        S(h_south,i,j,(G(vv,i,j) > 0) ? hh_ym : hh_yp);
      }
    }
    free(dh_dx);
    free(dh_dy);
  }

#undef G
#undef S
}

/*
 * Compute h_west/h_south by gathering full layers, applying the serial stencil,
 * and scattering back. This is slower than local stencils but removes boundary
 * ambiguity in pressure and diagnostic comparisons.
 */
bool pawsim_work_calc_face_thickness_no_ghost (pawsim_work * work,
                                               const pawsim_config * cfg,
                                               const pawsim_domain * dom)
{
  uint k;
  bool ok = true;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return false;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    real * global_h = NULL;
    real * global_u = NULL;
    real * global_v = NULL;
    real * global_hw = NULL;
    real * global_hs = NULL;
    size_t n = (size_t) dom->Nx * dom->Ny;

    pawsim_field2d_zero(&work->h_west[k]);
    pawsim_field2d_zero(&work->h_south[k]);

    if (dom->rank == 0)
    {
      global_h = malloc(n*sizeof(real));
      global_u = malloc(n*sizeof(real));
      global_v = malloc(n*sizeof(real));
      global_hw = malloc(n*sizeof(real));
      global_hs = malloc(n*sizeof(real));
      ok = (global_h != NULL) && (global_u != NULL) && (global_v != NULL)
        && (global_hw != NULL) && (global_hs != NULL);
    }
#ifdef PAWSIM_USE_MPI
    {
      int ok_int = ok ? 1 : 0;
      MPI_Bcast(&ok_int,1,MPI_INT,0,dom->comm);
      ok = (ok_int != 0);
    }
#endif
    if (!ok)
    {
      free(global_h);
      free(global_u);
      free(global_v);
      free(global_hw);
      free(global_hs);
      return false;
    }

    ok = gather_global_layer(global_h,&work->h_w[k],dom,6000+(int) (10*k))
      && gather_global_layer(global_u,&work->u_w[k],dom,6002+(int) (10*k))
      && gather_global_layer(global_v,&work->v_w[k],dom,6004+(int) (10*k));

    if (ok && (dom->rank == 0))
    {
      calc_face_thickness_global_no_ghost(global_h,global_u,global_v,
                                          global_hw,global_hs,cfg);
    }

    ok = ok
      && scatter_global_layer(&work->h_west[k],global_hw,dom,6006+(int) (10*k))
      && scatter_global_layer(&work->h_south[k],global_hs,dom,6008+(int) (10*k));

    free(global_h);
    free(global_u);
    free(global_v);
    free(global_hw);
    free(global_hs);

    if (!ok)
    {
      return false;
    }

    pawsim_field2d_exchange_scalar_halo(&work->h_west[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->h_south[k],dom);
  }

  return true;
}

/*
 * Build static topography and total-column-thickness work arrays on centers
 * and faces. AWSIM uses these repeatedly in pressure, forcing masks, and eta.
 */
void pawsim_work_init_static_geometry (pawsim_work * work, const pawsim_state * state,
                                       const pawsim_config * cfg,
                                       const pawsim_domain * dom)
{
  uint i,j,g;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  /*
   * Static surface/bottom geometry is copied into work arrays so the same face
   * interpolation and halo logic can be used for static and prognostic fields.
   */
  copy_interior(&work->hhs_w,&state->hhs);
  copy_interior(&work->hhb_w,&state->hhb);
  pawsim_field2d_exchange_scalar_halo(&work->hhs_w,dom);
  pawsim_field2d_exchange_scalar_halo(&work->hhb_w,dom);

  calc_face_thickness_layer(&work->hhs_w,&work->hhs_west,&work->hhs_south,
                            NULL,NULL,work,cfg);
  calc_face_thickness_layer(&work->hhb_w,&work->hhb_west,&work->hhb_south,
                            NULL,NULL,work,cfg);
  pawsim_field2d_exchange_scalar_halo(&work->hhs_west,dom);
  pawsim_field2d_exchange_scalar_halo(&work->hhs_south,dom);
  pawsim_field2d_exchange_scalar_halo(&work->hhb_west,dom);
  pawsim_field2d_exchange_scalar_halo(&work->hhb_south,dom);

  /** Hc, Hw and Hs are the total water-column thickness used by rigid-lid pressure. */
  g = work->Hc.nghost;
  for (i = 0; i < work->Hc.nx; i ++)
  {
    for (j = 0; j < work->Hc.ny; j ++)
    {
      work->Hc.a[i+g][j+g] = state->hhs.a[i+g][j+g] - state->hhb.a[i+g][j+g];
    }
  }
  pawsim_field2d_exchange_scalar_halo(&work->Hc,dom);

  calc_face_thickness_layer(&work->Hc,&work->Hw,&work->Hs,NULL,NULL,work,cfg);
  pawsim_field2d_exchange_scalar_halo(&work->Hw,dom);
  pawsim_field2d_exchange_scalar_halo(&work->Hs,dom);
}

/*
 * Compute layer-interface heights eta from bottom topography and layer
 * thicknesses. This is the PAWSIM equivalent of AWSIM's eta work array build.
 */
void pawsim_work_calc_eta (pawsim_work * work, const pawsim_domain * dom)
{
  int k;
  uint i,j;

  if ((work == NULL) || (dom == NULL))
  {
    return;
  }

  /** Interface heights are built upward from bottom topography. */
  for (k = (int) work->Nlay; k >= 0; k --)
  {
    for (i = 0; i < work->hhb_w.sx; i ++)
    {
      for (j = 0; j < work->hhb_w.sy; j ++)
      {
        if ((uint) k == work->Nlay)
        {
          work->eta_w[k].a[i][j] = work->hhb_w.a[i][j];
        }
        else
        {
          work->eta_w[k].a[i][j] = work->eta_w[k+1].a[i][j] + work->h_w[k].a[i][j];
        }
      }
    }

    pawsim_field2d_exchange_scalar_halo(&work->eta_w[k],dom);
  }
}

/** Compute velocity-dependent layer thicknesses on u and v faces. */
void pawsim_work_calc_face_thickness (pawsim_work * work, const pawsim_config * cfg,
                                      const pawsim_domain * dom)
{
  uint k;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  /** Prognostic layer thicknesses may use velocity-dependent advection schemes. */
  for (k = 0; k < work->Nlay; k ++)
  {
    calc_face_thickness_layer(&work->h_w[k],&work->h_west[k],&work->h_south[k],
                              &work->u_w[k],&work->v_w[k],work,cfg);
    pawsim_field2d_exchange_scalar_halo(&work->h_west[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->h_south[k],dom);
  }
}

/*
 * Compute relative vorticity, q-grid layer thickness, and potential vorticity.
 * Wall q values are computed from ghosted u/v/h rather than filled afterward,
 * preserving AWSIM's no viscous momentum flux property at solid walls.
 */
void pawsim_work_calc_pv (pawsim_work * work, const pawsim_state * state,
                          const pawsim_domain * dom, real dx, real dy)
{
  uint k,i,j,g;

  if ((work == NULL) || (state == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    uint imax = work->h_w[k].sx - 2;
    uint jmax = work->h_w[k].sy - 2;

    pawsim_field2d_zero(&work->zeta[k]);
    pawsim_field2d_zero(&work->hh_q[k]);
    pawsim_field2d_zero(&work->qq[k]);

    /*
     * Compute q-grid quantities over the full AWSIM ghosted work-array stencil.
     * The momentum PV terms only need locally defined q points, but the
     * biharmonic viscosity operator uses hh_q on a wider boundary stencil.
     * Building zeta/hh_q/qq everywhere the ghosted primitive variables support
     * the centered stencil keeps those A4 boundary terms aligned with AWSIM.
     */
    for (i = 1; i <= imax; i ++)
    {
      for (j = 1; j <= jmax; j ++)
      {
        work->zeta[k].a[i][j] = (work->u_w[k].a[i][j-1]-work->u_w[k].a[i][j]) / dy
                              + (work->v_w[k].a[i][j]-work->v_w[k].a[i-1][j]) / dx;

        work->hh_q[k].a[i][j] = 0.25 * (work->h_w[k].a[i][j]
                                      + work->h_w[k].a[i-1][j-1]
                                      + work->h_w[k].a[i-1][j]
                                      + work->h_w[k].a[i][j-1]);

        work->qq[k].a[i][j] = (2*state->omega_z.a[i][j] + work->zeta[k].a[i][j])
                            / work->hh_q[k].a[i][j];

      }
    }

    g = work->qq[k].nghost;
    (void) g;

    /*
     * These q-grid quantities inherit their wall behavior from the ghosted
     * primitive variables above.  Physical wall filling after this point would
     * overwrite directly-computed boundary values.
     */
    pawsim_field2d_exchange_internal_halo(&work->zeta[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->hh_q[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->qq[k],dom);

  }
}

/*
 * Compute the q interpolation coefficients used by the vector-invariant
 * momentum advection schemes: AL81/HK83, TW81, and Sadourny 1975 enstrophy.
 */
void pawsim_work_calc_pv_coefficients (pawsim_work * work, const pawsim_config * cfg,
                                       const pawsim_domain * dom)
{
  uint k,i,j;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->alpha[k]);
    pawsim_field2d_zero(&work->beta[k]);
    pawsim_field2d_zero(&work->gamma[k]);
    pawsim_field2d_zero(&work->delta[k]);
    pawsim_field2d_zero(&work->epsilon[k]);
    pawsim_field2d_zero(&work->phi[k]);
    pawsim_field2d_zero(&work->lambda[k]);
    pawsim_field2d_zero(&work->mu[k]);

    for (i = 1; i < work->qq[k].sx-1; i ++)
    {
      for (j = 1; j < work->qq[k].sy-1; j ++)
      {
        /*
         * These coefficients distribute q to the u/v momentum points. AL81 and
         * HK83 share one set; TW81 uses the wider Takano-Wurtele stencil with
         * additional lambda/mu terms.
         */
        if ((cfg->momentumScheme == MOMENTUM_AL81)
         || (cfg->momentumScheme == MOMENTUM_HK83))
        {
          work->alpha[k].a[i][j] = (2*work->qq[k].a[i+1][j+1]
                                  + work->qq[k].a[i][j+1]
                                  + 2*work->qq[k].a[i][j]
                                  + work->qq[k].a[i+1][j]) / 24;
          work->beta[k].a[i][j] = (work->qq[k].a[i][j+1]
                                 + 2*work->qq[k].a[i-1][j+1]
                                 + work->qq[k].a[i-1][j]
                                 + 2*work->qq[k].a[i][j]) / 24;
          work->gamma[k].a[i][j] = (2*work->qq[k].a[i][j+1]
                                  + work->qq[k].a[i-1][j+1]
                                  + 2*work->qq[k].a[i-1][j]
                                  + work->qq[k].a[i][j]) / 24;
          work->delta[k].a[i][j] = (work->qq[k].a[i+1][j+1]
                                  + 2*work->qq[k].a[i][j+1]
                                  + work->qq[k].a[i][j]
                                  + 2*work->qq[k].a[i+1][j]) / 24;
          work->epsilon[k].a[i][j] = (work->qq[k].a[i+1][j+1]
                                    + work->qq[k].a[i][j+1]
                                    - work->qq[k].a[i][j]
                                    - work->qq[k].a[i+1][j]) / 24;
          work->phi[k].a[i][j] = (-work->qq[k].a[i+1][j+1]
                                + work->qq[k].a[i][j+1]
                                + work->qq[k].a[i][j]
                                - work->qq[k].a[i+1][j]) / 24;
        }
        else if ((cfg->momentumScheme == MOMENTUM_TW81) && (j < work->qq[k].sy-2))
        {
          work->alpha[k].a[i][j] = (2*work->qq[k].a[i+1][j+1]
                                  + 3*work->qq[k].a[i][j+1]
                                  + 2*work->qq[k].a[i][j]
                                  + work->qq[k].a[i+1][j]
                                  - work->qq[k].a[i][j+2]
                                  - work->qq[k].a[i-1][j+1]) / 24;
          work->beta[k].a[i][j] = (3*work->qq[k].a[i][j+1]
                                 + 2*work->qq[k].a[i-1][j+1]
                                 + work->qq[k].a[i-1][j]
                                 + 2*work->qq[k].a[i][j]
                                 - work->qq[k].a[i+1][j+1]
                                 - work->qq[k].a[i][j+2]) / 24;
          work->gamma[k].a[i][j] = (2*work->qq[k].a[i][j+1]
                                  + work->qq[k].a[i-1][j+1]
                                  + 2*work->qq[k].a[i-1][j]
                                  + 3*work->qq[k].a[i][j]
                                  - work->qq[k].a[i][j-1]
                                  - work->qq[k].a[i+1][j]) / 24;
          work->delta[k].a[i][j] = (work->qq[k].a[i+1][j+1]
                                  + 2*work->qq[k].a[i][j+1]
                                  + 3*work->qq[k].a[i][j]
                                  + 2*work->qq[k].a[i+1][j]
                                  - work->qq[k].a[i-1][j]
                                  - work->qq[k].a[i][j-1]) / 24;
          work->epsilon[k].a[i][j] = (work->qq[k].a[i+1][j+1]
                                    + work->qq[k].a[i][j+1]
                                    - work->qq[k].a[i][j]
                                    - work->qq[k].a[i+1][j]) / 24;
          work->phi[k].a[i][j] = (-work->qq[k].a[i+1][j+1]
                                + work->qq[k].a[i][j+1]
                                + work->qq[k].a[i][j]
                                - work->qq[k].a[i+1][j]) / 24;
          work->lambda[k].a[i][j] = (work->qq[k].a[i+1][j]
                                   - work->qq[k].a[i-1][j]) / 24;
          work->mu[k].a[i][j] = (work->qq[k].a[i][j-1]
                               - work->qq[k].a[i][j+1]) / 24;
        }
      }
    }

    pawsim_field2d_exchange_internal_halo(&work->alpha[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->beta[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->gamma[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->delta[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->epsilon[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->phi[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->lambda[k],dom);
    pawsim_field2d_exchange_internal_halo(&work->mu[k],dom);

  }
}

/*
 * Compute Bernoulli/Montgomery pressure terms and kinetic-energy gradients.
 * For rigid-lid runs the barotropic pressure is added later by pressure.c.
 */
void pawsim_work_calc_bernoulli (pawsim_work * work, const pawsim_state * state,
                                 const pawsim_config * cfg,
                                 const pawsim_domain * dom)
{
  uint k,i,j;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->pp[k]);
    pawsim_field2d_zero(&work->KE_B[k]);
    pawsim_field2d_zero(&work->MM_B[k]);

    for (i = 1; i < work->h_w[k].sx-1; i ++)
    {
      for (j = 1; j < work->h_w[k].sy-1; j ++)
      {
        /*
         * Under a rigid lid the barotropic pressure pi is solved separately, so
         * the layer Bernoulli pressure starts at zero in the top layer.
         */
        if (k == 0)
        {
          work->pp[k].a[i][j] = cfg->useRL ? 0 : state->gg[0] * work->eta_w[0].a[i][j];
          if (!cfg->useRL && cfg->useTracer && cfg->useBuoyancy)
          {
            work->pp[k].a[i][j] -= work->b_w[k].a[i][j] * work->eta_w[0].a[i][j];
          }
        }
        else
        {
          work->pp[k].a[i][j] = work->pp[k-1].a[i][j]
                              + state->gg[k] * work->eta_w[k].a[i][j];
          if (cfg->useTracer && cfg->useBuoyancy)
          {
            work->pp[k].a[i][j] += work->eta_w[k].a[i][j]
                                 * (work->b_w[k-1].a[i][j]-work->b_w[k].a[i][j]);
          }
        }

        /** Kinetic energy in the Bernoulli function follows the momentum scheme. */
        switch (cfg->momentumScheme)
        {
          case MOMENTUM_HK83:
          {
            work->KE_B[k].a[i][j] =
              ( SQUARE(work->u_w[k].a[i+1][j]) + SQUARE(work->u_w[k].a[i][j])
              + 0.25 * ( SQUARE(work->u_w[k].a[i+1][j+1]) + SQUARE(work->u_w[k].a[i][j+1])
                       + SQUARE(work->u_w[k].a[i+1][j-1]) + SQUARE(work->u_w[k].a[i][j-1]) )
              + SQUARE(work->v_w[k].a[i][j+1]) + SQUARE(work->v_w[k].a[i][j])
              + 0.25 * ( SQUARE(work->v_w[k].a[i+1][j+1]) + SQUARE(work->v_w[k].a[i+1][j])
                       + SQUARE(work->v_w[k].a[i-1][j+1]) + SQUARE(work->v_w[k].a[i-1][j]) )
              ) / 6;
            break;
          }
          case MOMENTUM_TW81:
          {
            if ((i < work->h_w[k].sx-2) && (j < work->h_w[k].sy-2))
            {
              work->KE_B[k].a[i][j] =
                0.25 * ( (2.0/3.0)*SQUARE(work->u_w[k].a[i][j])
                       + (1.0/3.0)*SQUARE(0.5*(work->u_w[k].a[i-1][j]+work->u_w[k].a[i+1][j]))
                       + (2.0/3.0)*SQUARE(work->u_w[k].a[i+1][j])
                       + (1.0/3.0)*SQUARE(0.5*(work->u_w[k].a[i][j]+work->u_w[k].a[i+2][j]))
                       + (2.0/3.0)*SQUARE(work->v_w[k].a[i][j])
                       + (1.0/3.0)*SQUARE(0.5*(work->v_w[k].a[i][j-1]+work->v_w[k].a[i][j+1]))
                       + (2.0/3.0)*SQUARE(work->v_w[k].a[i][j+1])
                       + (1.0/3.0)*SQUARE(0.5*(work->v_w[k].a[i][j]+work->v_w[k].a[i][j+2])) );
            }
            break;
          }
          case MOMENTUM_AL81:
          case MOMENTUM_S75e:
          {
            work->KE_B[k].a[i][j] =
                ( SQUARE(work->u_w[k].a[i+1][j]) + SQUARE(work->u_w[k].a[i][j]) ) / 4
              + ( SQUARE(work->v_w[k].a[i][j+1]) + SQUARE(work->v_w[k].a[i][j]) ) / 4;
            break;
          }
          default:
          {
            break;
          }
        }

        work->MM_B[k].a[i][j] = work->pp[k].a[i][j];
        if (cfg->h0 != 0)
        {
          work->MM_B[k].a[i][j] -= state->geff[k] * POW4(cfg->h0)
                                 / POW3(work->h_w[k].a[i][j]) / 3;
        }
      }
    }

    pawsim_field2d_exchange_scalar_halo(&work->pp[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->KE_B[k],dom);
    pawsim_field2d_exchange_scalar_halo(&work->MM_B[k],dom);
  }
}

/*
 * Assemble the deterministic horizontal momentum tendency from Montgomery
 * pressure gradients, KE gradients, PV advection, viscosity, and active
 * buoyancy terms. External forcing/restoring/diapycnal terms are added in
 * their own modules after this core AWSIM stencil.
 */
void pawsim_work_calc_momentum_tendency (pawsim_work * work,
                                         const pawsim_config * cfg,
                                         const pawsim_domain * dom,
                                         real dx, real dy)
{
  uint k,i,j,g;
  real rhs_u,rhs_v;
  bool useA2,useA4,useSmag;
  real A2smag_fac,A4smag_fac,A4sqrt_const;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  /*
   * Viscosity options match AWSIM: constant Laplacian/biharmonic pieces can be
   * augmented by Smagorinsky coefficients evaluated on h and q locations.
   */
  useA2 = (cfg->A2 > 0) || (cfg->A2smag > 0);
  useA4 = (cfg->A4 > 0) || (cfg->A4smag > 0);
  useSmag = (cfg->A2smag > 0) || (cfg->A4smag > 0);
  A2smag_fac = SQUARE(cfg->A2smag*fmax(dx,dy)/_PI);
  A4smag_fac = SQUARE(cfg->A4smag/_PI) * POW4(fmax(dx,dy)) / 8;
  A4sqrt_const = sqrt(cfg->A4);

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->dt_u[k]);
    pawsim_field2d_zero(&work->dt_v[k]);

    /*
     * The AL81/HK83/TW81/S75 momentum terms use layer-local mass fluxes.
     * AWSIM builds these in the same layer loop as q before evaluating the
     * momentum RHS; PAWSIM keeps them as shared work arrays, so refresh them
     * for each layer immediately before that layer's RHS is computed.
     */
    calc_layer_mass_fluxes(work,k,dom);
    if (cfg->useTracer && cfg->useBuoyancy)
    {
      calc_layer_active_buoyancy_work(work,k,dx,dy);
    }

    if (useA2 || useA4)
    {
      pawsim_field2d_zero(&work->DD_T);
      pawsim_field2d_zero(&work->DD_S);
      pawsim_field2d_zero(&work->A2_h);
      pawsim_field2d_zero(&work->A2_q);
      pawsim_field2d_zero(&work->A4sqrt_h);
      pawsim_field2d_zero(&work->A4sqrt_q);
      pawsim_field2d_zero(&work->UU4);
      pawsim_field2d_zero(&work->VV4);
      pawsim_field2d_zero(&work->DD4_T);
      pawsim_field2d_zero(&work->DD4_S);

      for (i = 1; i < work->DD_T.sx-1; i ++)
      {
        for (j = 1; j < work->DD_T.sy-1; j ++)
        {
          work->DD_T.a[i][j] = (work->u_w[k].a[i+1][j]-work->u_w[k].a[i][j])/dx
                             - (work->v_w[k].a[i][j+1]-work->v_w[k].a[i][j])/dy;
          work->DD_S.a[i][j] = (work->u_w[k].a[i][j]-work->u_w[k].a[i][j-1])/dy
                             + (work->v_w[k].a[i][j]-work->v_w[k].a[i-1][j])/dx;
        }
      }

      if (useSmag)
      {
        /*
         * DD_T lives naturally at cell centers and DD_S at q points. The h/q
         * viscosity coefficients therefore use slightly different local
         * averages, following the AWSIM discretization.
         */
        for (i = 1; i < work->DD_T.sx-1; i ++)
        {
          for (j = 1; j < work->DD_T.sy-1; j ++)
          {
            real DD_q = sqrt(SQUARE(0.25*(work->DD_T.a[i][j]+work->DD_T.a[i-1][j]
                                      + work->DD_T.a[i][j-1]+work->DD_T.a[i-1][j-1]))
                           + SQUARE(work->DD_S.a[i][j]));
            real DD_h = sqrt(SQUARE(work->DD_T.a[i][j])
                           + SQUARE(0.25*(work->DD_S.a[i][j]+work->DD_S.a[i+1][j]
                                      + work->DD_S.a[i][j+1]+work->DD_S.a[i+1][j+1])));
            work->A2_q.a[i][j] = fmax(cfg->A2,A2smag_fac*DD_q);
            work->A2_h.a[i][j] = fmax(cfg->A2,A2smag_fac*DD_h);
            work->A4sqrt_q.a[i][j] = sqrt(fmax(cfg->A4,A4smag_fac*DD_q));
            work->A4sqrt_h.a[i][j] = sqrt(fmax(cfg->A4,A4smag_fac*DD_h));
          }
        }
      }
      else
      {
        for (i = 1; i < work->DD_T.sx-1; i ++)
        {
          for (j = 1; j < work->DD_T.sy-1; j ++)
          {
            if (useA2)
            {
              work->A2_q.a[i][j] = cfg->A2;
              work->A2_h.a[i][j] = cfg->A2;
            }
            if (useA4)
            {
              work->A4sqrt_q.a[i][j] = A4sqrt_const;
              work->A4sqrt_h.a[i][j] = A4sqrt_const;
            }
          }
        }
      }

      if (useA4)
      {
        /** Biharmonic viscosity is implemented as divergence of A4^1/2 h D4. */
        for (i = 1; i < work->UU4.sx-1; i ++)
        {
          for (j = 1; j < work->UU4.sy-1; j ++)
          {
            work->UU4.a[i][j] = (
                (work->A4sqrt_h.a[i][j]*work->h_w[k].a[i][j]*work->DD_T.a[i][j]
               - work->A4sqrt_h.a[i-1][j]*work->h_w[k].a[i-1][j]*work->DD_T.a[i-1][j]) / dx
              + (work->A4sqrt_q.a[i][j+1]*work->hh_q[k].a[i][j+1]*work->DD_S.a[i][j+1]
               - work->A4sqrt_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD_S.a[i][j]) / dy
              ) / work->h_west[k].a[i][j];
            work->VV4.a[i][j] = (
                (work->A4sqrt_q.a[i+1][j]*work->hh_q[k].a[i+1][j]*work->DD_S.a[i+1][j]
               - work->A4sqrt_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD_S.a[i][j]) / dx
              - (work->A4sqrt_h.a[i][j]*work->h_w[k].a[i][j]*work->DD_T.a[i][j]
               - work->A4sqrt_h.a[i][j-1]*work->h_w[k].a[i][j-1]*work->DD_T.a[i][j-1]) / dy
              ) / work->h_south[k].a[i][j];
          }
        }

        for (i = 1; i < work->DD4_T.sx-1; i ++)
        {
          for (j = 1; j < work->DD4_T.sy-1; j ++)
          {
            work->DD4_T.a[i][j] = (work->UU4.a[i+1][j]-work->UU4.a[i][j])/dx
                                - (work->VV4.a[i][j+1]-work->VV4.a[i][j])/dy;
            work->DD4_S.a[i][j] = (work->UU4.a[i][j]-work->UU4.a[i][j-1])/dy
                                + (work->VV4.a[i][j]-work->VV4.a[i-1][j])/dx;
          }
        }
      }
    }

    g = work->dt_u[k].nghost;
    for (i = g; i < g+work->dt_u[k].nx; i ++)
    {
      for (j = g; j < g+work->dt_u[k].ny; j ++)
      {
        real hwest = work->h_west[k].a[i][j];
        real hsouth = work->h_south[k].a[i][j];
        real rhs_u_gradM,rhs_u_gradKE,rhs_u_q,rhs_u_A2,rhs_u_A4,rhs_u_buoy;
        real rhs_v_gradM,rhs_v_gradKE,rhs_v_q,rhs_v_A2,rhs_v_A4,rhs_v_buoy;
        uint gi = dom->i0 + (i-g);
        uint gj = dom->j0 + (j-g);
        bool active_u = !(cfg->useWallEW && (gi == 0));
        bool active_v = !(cfg->useWallNS && (gj == 0));

        /** Pressure/Bernoulli gradient plus scheme-specific vector-invariant PV fluxes. */
        rhs_u_gradM = - (work->MM_B[k].a[i][j]-work->MM_B[k].a[i-1][j]) / dx;
        rhs_u_gradKE = - (work->KE_B[k].a[i][j]-work->KE_B[k].a[i-1][j]) / dx;
        rhs_u_q = 0;
        rhs_u_A2 = 0;
        rhs_u_A4 = 0;
        rhs_u_buoy = 0;

        switch (cfg->momentumScheme)
        {
          case MOMENTUM_AL81:
          case MOMENTUM_HK83:
          {
            rhs_u_q += work->alpha[k].a[i][j]*work->hvv[k].a[i][j+1]
                    + work->beta[k].a[i][j]*work->hvv[k].a[i-1][j+1]
                    + work->gamma[k].a[i][j]*work->hvv[k].a[i-1][j]
                    + work->delta[k].a[i][j]*work->hvv[k].a[i][j]
                    - work->epsilon[k].a[i][j]*work->huu[k].a[i+1][j]
                    + work->epsilon[k].a[i-1][j]*work->huu[k].a[i-1][j];
            break;
          }
          case MOMENTUM_TW81:
          {
            rhs_u_q += work->alpha[k].a[i][j]*work->hvv[k].a[i][j+1]
                    + work->beta[k].a[i][j]*work->hvv[k].a[i-1][j+1]
                    + work->gamma[k].a[i][j]*work->hvv[k].a[i-1][j]
                    + work->delta[k].a[i][j]*work->hvv[k].a[i][j]
                    - work->epsilon[k].a[i][j]*work->huu[k].a[i+1][j]
                    + work->epsilon[k].a[i-1][j]*work->huu[k].a[i-1][j]
                    - work->lambda[k].a[i][j+1]*work->huu[k].a[i][j]
                    + work->lambda[k].a[i][j]*work->huu[k].a[i][j-1];
            break;
          }
          case MOMENTUM_S75e:
          {
            rhs_u_q += 0.25 * work->qq[k].a[i][j] * (work->hvv[k].a[i][j] + work->hvv[k].a[i-1][j])
                    + 0.25 * work->qq[k].a[i][j+1] * (work->hvv[k].a[i][j+1] + work->hvv[k].a[i-1][j+1]);
            break;
          }
          default:
          {
            break;
          }
        }

        if (useA2)
        {
          rhs_u_A2 = (
              (work->A2_h.a[i][j]*work->h_w[k].a[i][j]*work->DD_T.a[i][j]
             - work->A2_h.a[i-1][j]*work->h_w[k].a[i-1][j]*work->DD_T.a[i-1][j]) / dx
            + (work->A2_q.a[i][j+1]*work->hh_q[k].a[i][j+1]*work->DD_S.a[i][j+1]
             - work->A2_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD_S.a[i][j]) / dy
            );
        }

        if (useA4)
        {
          rhs_u_A4 = -(
              (work->A4sqrt_h.a[i][j]*work->h_w[k].a[i][j]*work->DD4_T.a[i][j]
             - work->A4sqrt_h.a[i-1][j]*work->h_w[k].a[i-1][j]*work->DD4_T.a[i-1][j]) / dx
            + (work->A4sqrt_q.a[i][j+1]*work->hh_q[k].a[i][j+1]*work->DD4_S.a[i][j+1]
             - work->A4sqrt_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD4_S.a[i][j]) / dy
            );
        }

        if (cfg->useTracer && cfg->useBuoyancy && (hwest != 0))
        {
          rhs_u_buoy = -0.5*(work->hz.a[i][j]+work->hz.a[i-1][j])
                     * work->db_dx.a[i][j];
        }

        rhs_u = rhs_u_gradM + rhs_u_gradKE + rhs_u_q
              + ((hwest != 0) ? (rhs_u_A2+rhs_u_A4+rhs_u_buoy)/hwest : 0);
        diag_add(work->diag_umom,PAWSIM_UMOM_GRADM,k,i,j,hwest*rhs_u_gradM*cfg->dt);
        diag_add(work->diag_umom,PAWSIM_UMOM_GRADKE,k,i,j,hwest*rhs_u_gradKE*cfg->dt);
        diag_add(work->diag_umom,PAWSIM_UMOM_Q,k,i,j,hwest*rhs_u_q*cfg->dt);
        diag_add(work->diag_umom,PAWSIM_UMOM_A2,k,i,j,active_u ? rhs_u_A2*cfg->dt : 0);
        diag_add(work->diag_umom,PAWSIM_UMOM_A4,k,i,j,active_u ? rhs_u_A4*cfg->dt : 0);
        diag_add(work->diag_energy,PAWSIM_ENERGY_GRADM,k,i,j,hwest*rhs_u_gradM*work->u_w[k].a[i][j]*cfg->dt);
        diag_add(work->diag_energy,PAWSIM_ENERGY_ADV,k,i,j,hwest*(rhs_u_gradKE+rhs_u_q)*work->u_w[k].a[i][j]*cfg->dt);
        diag_add(work->diag_energy,PAWSIM_ENERGY_A4,k,i,j,rhs_u_A4*work->u_w[k].a[i][j]*cfg->dt);

        rhs_v_gradM = - (work->MM_B[k].a[i][j]-work->MM_B[k].a[i][j-1]) / dy;
        rhs_v_gradKE = - (work->KE_B[k].a[i][j]-work->KE_B[k].a[i][j-1]) / dy;
        rhs_v_q = 0;
        rhs_v_A2 = 0;
        rhs_v_A4 = 0;
        rhs_v_buoy = 0;

        switch (cfg->momentumScheme)
        {
          case MOMENTUM_AL81:
          case MOMENTUM_HK83:
          {
            rhs_v_q += work->phi[k].a[i][j-1]*work->hvv[k].a[i][j-1]
                    - work->phi[k].a[i][j]*work->hvv[k].a[i][j+1]
                    - work->gamma[k].a[i+1][j]*work->huu[k].a[i+1][j]
                    - work->delta[k].a[i][j]*work->huu[k].a[i][j]
                    - work->alpha[k].a[i][j-1]*work->huu[k].a[i][j-1]
                    - work->beta[k].a[i+1][j-1]*work->huu[k].a[i+1][j-1];
            break;
          }
          case MOMENTUM_TW81:
          {
            rhs_v_q += work->phi[k].a[i][j-1]*work->hvv[k].a[i][j-1]
                    - work->phi[k].a[i][j]*work->hvv[k].a[i][j+1]
                    - work->gamma[k].a[i+1][j]*work->huu[k].a[i+1][j]
                    - work->delta[k].a[i][j]*work->huu[k].a[i][j]
                    - work->alpha[k].a[i][j-1]*work->huu[k].a[i][j-1]
                    - work->beta[k].a[i+1][j-1]*work->huu[k].a[i+1][j-1]
                    - work->mu[k].a[i+1][j]*work->hvv[k].a[i][j]
                    + work->mu[k].a[i][j]*work->hvv[k].a[i-1][j];
            break;
          }
          case MOMENTUM_S75e:
          {
            rhs_v_q -= 0.25 * work->qq[k].a[i][j] * (work->huu[k].a[i][j] + work->huu[k].a[i][j-1])
                    + 0.25 * work->qq[k].a[i+1][j] * (work->huu[k].a[i+1][j] + work->huu[k].a[i+1][j-1]);
            break;
          }
          default:
          {
            break;
          }
        }

        if (useA2)
        {
          rhs_v_A2 = (
              (work->A2_q.a[i+1][j]*work->hh_q[k].a[i+1][j]*work->DD_S.a[i+1][j]
             - work->A2_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD_S.a[i][j]) / dx
            - (work->A2_h.a[i][j]*work->h_w[k].a[i][j]*work->DD_T.a[i][j]
             - work->A2_h.a[i][j-1]*work->h_w[k].a[i][j-1]*work->DD_T.a[i][j-1]) / dy
            );
        }

        if (useA4)
        {
          rhs_v_A4 = -(
              (work->A4sqrt_q.a[i+1][j]*work->hh_q[k].a[i+1][j]*work->DD4_S.a[i+1][j]
             - work->A4sqrt_q.a[i][j]*work->hh_q[k].a[i][j]*work->DD4_S.a[i][j]) / dx
            - (work->A4sqrt_h.a[i][j]*work->h_w[k].a[i][j]*work->DD4_T.a[i][j]
             - work->A4sqrt_h.a[i][j-1]*work->h_w[k].a[i][j-1]*work->DD4_T.a[i][j-1]) / dy
            );
        }

        if (cfg->useTracer && cfg->useBuoyancy && (hsouth != 0))
        {
          rhs_v_buoy = -0.5*(work->hz.a[i][j]+work->hz.a[i][j-1])
                     * work->db_dy.a[i][j];
        }

        rhs_v = rhs_v_gradM + rhs_v_gradKE + rhs_v_q
              + ((hsouth != 0) ? (rhs_v_A2+rhs_v_A4+rhs_v_buoy)/hsouth : 0);
        diag_add(work->diag_vmom,PAWSIM_VMOM_GRADM,k,i,j,hsouth*rhs_v_gradM*cfg->dt);
        diag_add(work->diag_vmom,PAWSIM_VMOM_GRADKE,k,i,j,hsouth*rhs_v_gradKE*cfg->dt);
        diag_add(work->diag_vmom,PAWSIM_VMOM_Q,k,i,j,hsouth*rhs_v_q*cfg->dt);
        diag_add(work->diag_vmom,PAWSIM_VMOM_A2,k,i,j,active_v ? rhs_v_A2*cfg->dt : 0);
        diag_add(work->diag_vmom,PAWSIM_VMOM_A4,k,i,j,active_v ? rhs_v_A4*cfg->dt : 0);
        diag_add(work->diag_energy,PAWSIM_ENERGY_GRADM,k,i,j,hsouth*rhs_v_gradM*work->v_w[k].a[i][j]*cfg->dt);
        diag_add(work->diag_energy,PAWSIM_ENERGY_ADV,k,i,j,hsouth*(rhs_v_gradKE+rhs_v_q)*work->v_w[k].a[i][j]*cfg->dt);
        diag_add(work->diag_energy,PAWSIM_ENERGY_A4,k,i,j,rhs_v_A4*work->v_w[k].a[i][j]*cfg->dt);

        work->dt_u[k].a[i][j] = rhs_u;
        work->dt_v[k].a[i][j] = rhs_v;
      }
    }

    pawsim_field2d_exchange_u_halo(&work->dt_u[k],dom);
    pawsim_field2d_exchange_v_halo(&work->dt_v[k],dom);
  }
}

/*
 * Compute layer-thickness tendencies from mass-flux divergence and diapycnal
 * interface velocity. Momentum and tracer diagnostics depend on this same
 * dh/dt split, so the diagnostic bookkeeping lives here too.
 */
void pawsim_work_calc_thickness_tendency (pawsim_work * work,
                                          const pawsim_state * state,
                                          const pawsim_config * cfg,
                                          const pawsim_domain * dom, real dx, real dy)
{
  uint k,i,j,g;
  real rhs_adv,rhs_relax;

  if ((work == NULL) || (state == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->dt_h[k]);

    /** Reuse the layer mass fluxes refreshed by the preceding momentum step. */
    g = work->dt_h[k].nghost;
    for (i = g; i < g+work->dt_h[k].nx; i ++)
    {
      for (j = g; j < g+work->dt_h[k].ny; j ++)
      {
        rhs_adv = (work->huu[k].a[i][j]-work->huu[k].a[i+1][j]) / dx
                + (work->hvv[k].a[i][j]-work->hvv[k].a[i][j+1]) / dy;
        rhs_relax = - (work->wdia[k].a[i][j] - work->wdia[k+1].a[i][j]);
        work->dt_h[k].a[i][j] = rhs_adv + rhs_relax;
        diag_add(work->diag_thic,PAWSIM_THIC_ADV,k,i,j,rhs_adv*cfg->dt);
        diag_add(work->diag_energy,PAWSIM_ENERGY_ADV,k,i,j,
                 0.5*(SQUARE(work->u_w[k].a[i][j])+SQUARE(work->v_w[k].a[i][j]))
                 * rhs_adv*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_ADV,k,i,j,work->b_w[k].a[i][j]*rhs_adv*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_WDIA,k,i,j,work->b_w[k].a[i][j]*rhs_relax*cfg->dt);
      }
    }

    pawsim_field2d_exchange_scalar_halo(&work->dt_h[k],dom);
    if (work->diag_thic != NULL)
    {
      if (work->diag_thic[PAWSIM_THIC_ADV] != NULL)
      {
        pawsim_field2d_exchange_scalar_halo(&work->diag_thic[PAWSIM_THIC_ADV][k],dom);
      }
    }

    for (i = g; i < g+work->dt_h[k].nx; i ++)
    {
      for (j = g; j < g+work->dt_h[k].ny; j ++)
      {
        real adv_c = (work->huu[k].a[i][j]-work->huu[k].a[i+1][j]) / dx
                   + (work->hvv[k].a[i][j]-work->hvv[k].a[i][j+1]) / dy;
        real adv_w = (work->huu[k].a[i-1][j]-work->huu[k].a[i][j]) / dx
                   + (work->hvv[k].a[i-1][j]-work->hvv[k].a[i-1][j+1]) / dy;
        real adv_s = (work->huu[k].a[i][j-1]-work->huu[k].a[i+1][j-1]) / dx
                   + (work->hvv[k].a[i][j-1]-work->hvv[k].a[i][j]) / dy;
        real adv_u = 0.5*work->u_w[k].a[i][j]*(adv_c+adv_w);
        real adv_v = 0.5*work->v_w[k].a[i][j]*(adv_c+adv_s);
        diag_add(work->diag_umom,PAWSIM_UMOM_DHDT,k,i,j,adv_u*cfg->dt);
        diag_add(work->diag_vmom,PAWSIM_VMOM_DHDT,k,i,j,adv_v*cfg->dt);
      }
    }
  }
}

/*
 * Compute active/passive tracer tendencies: horizontal advection, vertical
 * advection, diapycnal diffusion, horizontal K2/K4 diffusion, and restoring.
 */
void pawsim_work_calc_tracer_tendency (pawsim_work * work, const pawsim_state * state,
                                       const pawsim_config * cfg,
                                       const pawsim_domain * dom, real dx, real dy)
{
  uint k,i,j,g;
  real dxsq,dysq;

  if ((work == NULL) || (state == NULL) || (cfg == NULL) || (dom == NULL)
   || !cfg->useTracer)
  {
    return;
  }

  dxsq = dx*dx;
  dysq = dy*dy;

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->dt_b[k]);
    pawsim_field2d_zero(&work->hub);
    pawsim_field2d_zero(&work->hvb);

    for (i = 1; i < work->b_w[k].sx-1; i ++)
    {
      for (j = 1; j < work->b_w[k].sy-1; j ++)
      {
        if ((cfg->tracerScheme == TRACER_AL81) || (cfg->tracerScheme == TRACER_UP3))
        {
          work->db_dx.a[i][j] = (work->b_w[k].a[i][j]-work->b_w[k].a[i-1][j]) / dx;
          work->db_dy.a[i][j] = (work->b_w[k].a[i][j]-work->b_w[k].a[i][j-1]) / dy;
          work->udb_dx.a[i][j] = work->u_w[k].a[i][j] * work->db_dx.a[i][j];
          work->vdb_dy.a[i][j] = work->v_w[k].a[i][j] * work->db_dy.a[i][j];
        }

        if (cfg->K4 > 0)
        {
          work->BB4.a[i][j] =
            ((work->h_west[k].a[i+1][j]*(work->b_w[k].a[i+1][j]-work->b_w[k].a[i][j])
            - work->h_west[k].a[i][j]*(work->b_w[k].a[i][j]-work->b_w[k].a[i-1][j])) / dxsq
           + (work->h_south[k].a[i][j+1]*(work->b_w[k].a[i][j+1]-work->b_w[k].a[i][j])
            - work->h_south[k].a[i][j]*(work->b_w[k].a[i][j]-work->b_w[k].a[i][j-1])) / dysq)
            / work->h_w[k].a[i][j];
        }
      }
    }

    if (cfg->tracerScheme == TRACER_UP3)
    {
      for (i = 1; i < work->b_w[k].sx-1; i ++)
      {
        for (j = 1; j < work->b_w[k].sy-1; j ++)
        {
          work->d2bx.a[i][j] = work->b_w[k].a[i+1][j] - 2*work->b_w[k].a[i][j]
                             + work->b_w[k].a[i-1][j];
          work->d2by.a[i][j] = work->b_w[k].a[i][j+1] - 2*work->b_w[k].a[i][j]
                             + work->b_w[k].a[i][j-1];
        }
      }

      for (i = 1; i < work->b_w[k].sx-1; i ++)
      {
        for (j = 1; j < work->b_w[k].sy-1; j ++)
        {
          work->hvb.a[i][j] = 0.5*(work->b_w[k].a[i][j]+work->b_w[k].a[i][j-1])
                            - (1/6.0) * ((work->v_w[k].a[i][j] > 0)
                                          ? work->d2by.a[i][j-1] : work->d2by.a[i][j]);
          work->hvb.a[i][j] *= work->hvv[k].a[i][j];
          work->hub.a[i][j] = 0.5*(work->b_w[k].a[i][j]+work->b_w[k].a[i-1][j])
                            - (1/6.0) * ((work->u_w[k].a[i][j] > 0)
                                          ? work->d2bx.a[i-1][j] : work->d2bx.a[i][j]);
          work->hub.a[i][j] *= work->huu[k].a[i][j];
        }
      }
    }
    else if (cfg->tracerScheme == TRACER_KT00)
    {
      for (i = 1; i < work->b_w[k].sx-1; i ++)
      {
        for (j = 1; j < work->b_w[k].sy-1; j ++)
        {
          work->db_dx.a[i][j] = minmod(cfg->KT00_sigma*(work->b_w[k].a[i+1][j]-work->b_w[k].a[i][j]),
                                       0.5*(work->b_w[k].a[i+1][j]-work->b_w[k].a[i-1][j]),
                                       cfg->KT00_sigma*(work->b_w[k].a[i][j]-work->b_w[k].a[i-1][j])) / dx;
          work->db_dy.a[i][j] = minmod(cfg->KT00_sigma*(work->b_w[k].a[i][j+1]-work->b_w[k].a[i][j]),
                                       0.5*(work->b_w[k].a[i][j+1]-work->b_w[k].a[i][j-1]),
                                       cfg->KT00_sigma*(work->b_w[k].a[i][j]-work->b_w[k].a[i][j-1])) / dy;
        }
      }

      for (i = 1; i < work->b_w[k].sx-1; i ++)
      {
        for (j = 1; j < work->b_w[k].sy-1; j ++)
        {
          real b_xm = work->b_w[k].a[i-1][j] + 0.5*work->db_dx.a[i-1][j]*dx;
          real b_xp = work->b_w[k].a[i][j] - 0.5*work->db_dx.a[i][j]*dx;
          real b_ym = work->b_w[k].a[i][j-1] + 0.5*work->db_dy.a[i][j-1]*dy;
          real b_yp = work->b_w[k].a[i][j] + 0.5*work->db_dy.a[i][j]*dy;

          work->hub.a[i][j] = ((work->u_w[k].a[i][j] > 0) ? b_xm : b_xp)
                            * work->huu[k].a[i][j];
          work->hvb.a[i][j] = ((work->v_w[k].a[i][j] > 0) ? b_ym : b_yp)
                            * work->hvv[k].a[i][j];
        }
      }
    }

    g = work->dt_b[k].nghost;
    for (i = g; i < g+work->dt_b[k].nx; i ++)
    {
      for (j = g; j < g+work->dt_b[k].ny; j ++)
      {
        real rhs_b = 0;
        real rhs_adv = 0;
        real rhs_wdia = 0;
        real rhs_diadiff = 0;
        real rhs_K2 = 0;
        real rhs_K4 = 0;
        real rhs_relax = 0;
        real h = work->h_w[k].a[i][j];

        if (cfg->tracerScheme == TRACER_AL81)
        {
          rhs_adv = -0.5*(work->udb_dx.a[i+1][j] + work->udb_dx.a[i][j]
                        + work->vdb_dy.a[i][j+1] + work->vdb_dy.a[i][j]);
        }
        else if (h != 0)
        {
          rhs_adv = -((work->hub.a[i+1][j]-work->hub.a[i][j]) / dx
                    + (work->hvb.a[i][j+1]-work->hvb.a[i][j]) / dy) / h
                  + work->b_w[k].a[i][j]
                  * ((work->huu[k].a[i+1][j]-work->huu[k].a[i][j]) / dx
                   + (work->hvv[k].a[i][j+1]-work->hvv[k].a[i][j]) / dy) / h;
        }
        rhs_b += rhs_adv;

        if (cfg->useWDia && (h != 0))
        {
          real b_p = (k == 0) ? 0 : work->b_w[k-1].a[i][j];
          real b_m = (k == work->Nlay-1) ? 0 : work->b_w[k+1].a[i][j];
          real db_p = (work->wdia[k].a[i][j] > 0) ? 0 : b_p-work->b_w[k].a[i][j];
          real db_m = (work->wdia[k+1].a[i][j] > 0) ? work->b_w[k].a[i][j]-b_m : 0;
          if (cfg->useBuoyancy)
          {
            db_p += (work->wdia[k].a[i][j] > 0) ? 0 : state->gg[k];
            db_m += ((k+1 < work->Nlay) && (work->wdia[k+1].a[i][j] > 0))
                  ? state->gg[k+1] : 0;
          }
          rhs_wdia = - (work->wdia[k].a[i][j]*db_p + work->wdia[k+1].a[i][j]*db_m);
          rhs_b += rhs_wdia / h;
        }

        if (cfg->useDiaDiff && (h != 0))
        {
          rhs_diadiff = work->Fdia_b[k].a[i][j]-work->Fdia_b[k+1].a[i][j];
          rhs_b += rhs_diadiff / h;
        }

        if ((cfg->K2 > 0) && (h != 0))
        {
          rhs_K2 = cfg->K2 *
            ((work->h_west[k].a[i+1][j]*(work->b_w[k].a[i+1][j]-work->b_w[k].a[i][j])
            - work->h_west[k].a[i][j]*(work->b_w[k].a[i][j]-work->b_w[k].a[i-1][j])) / dxsq
           + (work->h_south[k].a[i][j+1]*(work->b_w[k].a[i][j+1]-work->b_w[k].a[i][j])
            - work->h_south[k].a[i][j]*(work->b_w[k].a[i][j]-work->b_w[k].a[i][j-1])) / dysq);
          rhs_b += rhs_K2 / h;
        }

        if ((cfg->K4 > 0) && (h != 0))
        {
          rhs_K4 = -cfg->K4 *
            ((work->h_west[k].a[i+1][j]*(work->BB4.a[i+1][j]-work->BB4.a[i][j])
            - work->h_west[k].a[i][j]*(work->BB4.a[i][j]-work->BB4.a[i-1][j])) / dxsq
           + (work->h_south[k].a[i][j+1]*(work->BB4.a[i][j+1]-work->BB4.a[i][j])
            - work->h_south[k].a[i][j]*(work->BB4.a[i][j]-work->BB4.a[i][j-1])) / dysq);
          rhs_b += rhs_K4 / h;
        }

        if (state->bTime[k].a[i][j] > 0)
        {
          rhs_relax = - (work->b_w[k].a[i][j] - state->bRelax[k].a[i][j])
                    / state->bTime[k].a[i][j];
          rhs_b += rhs_relax;
        }

        diag_add(work->diag_trac,PAWSIM_TRAC_ADV,k,i,j,h*rhs_adv*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_WDIA,k,i,j,rhs_wdia*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_DIADIFF,k,i,j,rhs_diadiff*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_K2,k,i,j,rhs_K2*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_K4,k,i,j,rhs_K4*cfg->dt);
        diag_add(work->diag_trac,PAWSIM_TRAC_RELAX,k,i,j,h*rhs_relax*cfg->dt);

        work->dt_b[k].a[i][j] = rhs_b;
      }
    }

    pawsim_field2d_exchange_scalar_halo(&work->dt_b[k],dom);
  }
}

/*
 * Calculate the surface-forcing layer fractions hFsurf_west/south from the
 * mixed-layer depth hsml. These weights distribute wind stress and surface drag
 * through the upper water column.
 */
void pawsim_work_calc_surface_forcing_thickness (pawsim_work * work,
                                                 const pawsim_config * cfg,
                                                 const pawsim_domain * dom)
{
  uint i,j,k;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->hFsurf_west[k]);
    pawsim_field2d_zero(&work->hFsurf_south[k]);
  }

  /*
   * hFsurf_* is the fraction of the applied surface stress assigned to each
   * layer at u/v faces. With hsml <= 0, AWSIM applies all stress to the top
   * layer.
   */
  if (cfg->hsml <= 0)
  {
    if (work->Nlay > 0)
    {
      for (i = 0; i < work->hFsurf_west[0].sx; i ++)
      {
        for (j = 0; j < work->hFsurf_west[0].sy; j ++)
        {
          work->hFsurf_west[0].a[i][j] = 1;
          work->hFsurf_south[0].a[i][j] = 1;
        }
      }
    }
  }
  else
  {
    for (i = 1; i < work->hFsurf_west[0].sx-1; i ++)
    {
      for (j = 1; j < work->hFsurf_west[0].sy-1; j ++)
      {
        real z_sml_west = work->hhs_west.a[i][j] - cfg->hsml;
        real z_sml_south = work->hhs_south.a[i][j] - cfg->hsml;
        real eta_west_upper = work->hhs_west.a[i][j];
        real eta_south_upper = work->hhs_south.a[i][j];
        real hF_west_sum = 0;
        real hF_south_sum = 0;

        for (k = 0; k < work->Nlay; k ++)
        {
          real eta_west_lower = eta_west_upper - work->h_west[k].a[i][j];
          real eta_south_lower = eta_south_upper - work->h_south[k].a[i][j];

          if (hF_west_sum < 1.0)
          {
            real heff_west;

            z_sml_west -= fmin(cfg->hmin_surf,work->h_west[k].a[i][j]);
            heff_west = fmax(eta_west_upper,z_sml_west)
                       - fmax(eta_west_lower,z_sml_west);
            heff_west = fmax(heff_west-cfg->hmin_surf,0);
            work->hFsurf_west[k].a[i][j] = heff_west / cfg->hsml;
            hF_west_sum += work->hFsurf_west[k].a[i][j];
          }

          if (hF_south_sum < 1.0)
          {
            real heff_south;

            z_sml_south -= fmin(cfg->hmin_surf,work->h_south[k].a[i][j]);
            heff_south = fmax(eta_south_upper,z_sml_south)
                        - fmax(eta_south_lower,z_sml_south);
            heff_south = fmax(heff_south-cfg->hmin_surf,0);
            work->hFsurf_south[k].a[i][j] = heff_south / cfg->hsml;
            hF_south_sum += work->hFsurf_south[k].a[i][j];
          }

          eta_west_upper = eta_west_lower;
          eta_south_upper = eta_south_lower;
        }
      }
    }
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_exchange_u_halo(&work->hFsurf_west[k],dom);
    pawsim_field2d_exchange_v_halo(&work->hFsurf_south[k],dom);
  }
}

/*
 * Calculate the bottom-forcing layer fractions hFbot_west/south from hbbl.
 * These weights distribute linear and quadratic bottom drag upward from the
 * sea floor.
 */
void pawsim_work_calc_bottom_forcing_thickness (pawsim_work * work,
                                                const pawsim_config * cfg,
                                                const pawsim_domain * dom)
{
  uint i,j,k;

  if ((work == NULL) || (cfg == NULL) || (dom == NULL))
  {
    return;
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_zero(&work->hFbot_west[k]);
    pawsim_field2d_zero(&work->hFbot_south[k]);
  }

  /*
   * hFbot_* is the fraction of bottom drag assigned to each layer at u/v faces.
   * With hbbl <= 0, AWSIM applies all drag to the bottom layer.
   */
  if (cfg->hbbl <= 0)
  {
    if (work->Nlay > 0)
    {
      uint kb = work->Nlay - 1;

      for (i = 0; i < work->hFbot_west[kb].sx; i ++)
      {
        for (j = 0; j < work->hFbot_west[kb].sy; j ++)
        {
          work->hFbot_west[kb].a[i][j] = 1;
          work->hFbot_south[kb].a[i][j] = 1;
        }
      }
    }
  }
  else
  {
    for (i = 1; i < work->hFbot_west[0].sx-1; i ++)
    {
      for (j = 1; j < work->hFbot_west[0].sy-1; j ++)
      {
        real z_bbl_west = work->hhb_west.a[i][j] + cfg->hbbl;
        real z_bbl_south = work->hhb_south.a[i][j] + cfg->hbbl;
        real eta_west_lower = work->hhb_west.a[i][j];
        real eta_south_lower = work->hhb_south.a[i][j];
        real hF_west_sum = 0;
        real hF_south_sum = 0;

        for (k = work->Nlay; k > 0; k --)
        {
          uint kk = k - 1;
          real eta_west_upper = eta_west_lower + work->h_west[kk].a[i][j];
          real eta_south_upper = eta_south_lower + work->h_south[kk].a[i][j];

          if (hF_west_sum < 1.0)
          {
            real heff_west;

            z_bbl_west += fmin(cfg->hmin_bot,work->h_west[kk].a[i][j]);
            heff_west = fmin(eta_west_upper,z_bbl_west)
                       - fmin(eta_west_lower,z_bbl_west);
            heff_west = fmax(heff_west-cfg->hmin_bot,0);
            work->hFbot_west[kk].a[i][j] = heff_west / cfg->hbbl;
            hF_west_sum += work->hFbot_west[kk].a[i][j];
          }

          if (hF_south_sum < 1.0)
          {
            real heff_south;

            z_bbl_south += fmin(cfg->hmin_bot,work->h_south[kk].a[i][j]);
            heff_south = fmin(eta_south_upper,z_bbl_south)
                        - fmin(eta_south_lower,z_bbl_south);
            heff_south = fmax(heff_south-cfg->hmin_bot,0);
            work->hFbot_south[kk].a[i][j] = heff_south / cfg->hbbl;
            hF_south_sum += work->hFbot_south[kk].a[i][j];
          }

          eta_west_lower = eta_west_upper;
          eta_south_lower = eta_south_upper;
        }
      }
    }
  }

  for (k = 0; k < work->Nlay; k ++)
  {
    pawsim_field2d_exchange_u_halo(&work->hFbot_west[k],dom);
    pawsim_field2d_exchange_v_halo(&work->hFbot_south[k],dom);
  }
}
