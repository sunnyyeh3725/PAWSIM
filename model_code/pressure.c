/**
 * pressure.c
 *
 * PAWSIM rigid-lid pressure correction.
 *
 * The rigid-lid correction solves a variable-coefficient elliptic equation
 * for the barotropic pressure pi, then removes the barotropic divergence from
 * every layer by subtracting grad(pi) from the velocities. The pressure is
 * only defined up to an arbitrary constant, so all solvers subtract its global
 * mean before returning.
 *
 * Three solver paths are kept here:
 *
 *  - distributed red-black SOR, mainly as a simple reference implementation;
 *  - gathered multigrid, which reuses the AWSIM serial multigrid code on rank
 *    0 and scatters the result back to the MPI decomposition;
 *  - distributed multigrid, which smooths/restricts on distributed fine levels
 *    and uses a gathered coarse tail when the decomposition stops being a
 *    clean 2:1 hierarchy.
 *
 * The gathered coarse tail is a deliberate compromise. It lets arbitrary grid
 * sizes such as 250 x 250 run through the distributed-MG entry point without
 * reverting the whole solve to gathered AWSIM MG, but it is not expected to
 * scale as well as a fully distributed coarse-grid solve.
 *
 */
#include "pressure.h"

/** Add a term to an optional diagnostic budget array. */
static void diag_add (pawsim_field2d ** terms, uint term, uint k, uint i, uint j,
                      real value)
{
  if (terms != NULL)
  {
    terms[term][k].a[i][j] += value;
  }
}
#include "multigrid.h"

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

/** Compute the global mean of a decomposed cell-centered field. */
static real field_mean (const pawsim_field2d * field, const pawsim_domain * dom)
{
  uint i,j,g;
  real local_sum = 0;
  real global_sum = 0;

  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      local_sum += field->a[i+g][j+g];
    }
  }

  MPI_Allreduce(&local_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,dom->comm);
  return global_sum / ((real) dom->Nx * (real) dom->Ny);
}

/** Remove the null-space component from a distributed pressure-like field. */
static void subtract_field_mean (pawsim_field2d * field, const pawsim_domain * dom)
{
  uint i,j,g;
  real mean = field_mean(field,dom);

  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      field->a[i+g][j+g] -= mean;
    }
  }
}

/** Pack the rank-local owned region into a rank-0 global matrix. */
static void copy_local_to_global_field (real ** global,
                                        const pawsim_field2d * field,
                                        const pawsim_domain * dom)
{
  uint i,j,g;

  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      global[dom->i0+i][dom->j0+j] = field->a[i+g][j+g];
    }
  }
}

/** Copy a rank's owned rectangle out of a rank-0 global matrix. */
static void copy_global_to_local_field (pawsim_field2d * field,
                                        real ** global,
                                        const pawsim_domain * dom)
{
  uint i,j,g;

  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      field->a[i+g][j+g] = global[dom->i0+i][dom->j0+j];
    }
  }
}

/*
 * Gather a decomposed scalar field onto rank 0.
 *
 * This intentionally uses simple point-to-point messages with explicit
 * rectangle metadata. The pressure solves are still changing quickly, and this
 * keeps gathered-reference paths easy to reason about before replacing them
 * with MPI derived datatypes or collectives.
 */
static bool gather_pressure_field (real ** global,
                                   const pawsim_field2d * field,
                                   const pawsim_domain * dom,
                                   int tag_base)
{
#ifdef PAWSIM_USE_MPI
  if (dom->rank == 0)
  {
    int src;

    copy_local_to_global_field(global,field,dom);
    for (src = 1; src < dom->size; src ++)
    {
      uint i,j;
      uint meta[4];
      real * buf = NULL;

      MPI_Recv(meta,4,MPI_UNSIGNED,src,tag_base,dom->comm,MPI_STATUS_IGNORE);
      buf = malloc((size_t) meta[2]*meta[3]*sizeof(real));
      if (buf == NULL)
      {
        return false;
      }
      MPI_Recv(buf,(int) (meta[2]*meta[3]),MPI_DOUBLE,src,tag_base+1,dom->comm,MPI_STATUS_IGNORE);
      for (i = 0; i < meta[2]; i ++)
      {
        for (j = 0; j < meta[3]; j ++)
        {
          global[meta[0]+i][meta[1]+j] = buf[i*meta[3]+j];
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
  copy_local_to_global_field(global,field,dom);
  (void) tag_base;
  return true;
#endif
}

/** Scatter a rank-0 global matrix back onto the current Cartesian layout. */
static bool scatter_pressure_field (pawsim_field2d * field,
                                    real ** global,
                                    const pawsim_domain * dom,
                                    int tag_base)
{
#ifdef PAWSIM_USE_MPI
  if (dom->rank == 0)
  {
    int rank;

    copy_global_to_local_field(field,global,dom);
    for (rank = 1; rank < dom->size; rank ++)
    {
      int coords[2];
      uint i0,j0,nx,ny;
      real * buf = NULL;
      uint i,j;

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
          buf[i*ny+j] = global[i0+i][j0+j];
        }
      }
      MPI_Send(buf,(int) (nx*ny),MPI_DOUBLE,rank,tag_base,dom->comm);
      free(buf);
    }
  }
  else
  {
    real * buf = malloc((size_t) field->nx*field->ny*sizeof(real));
    uint i,j,g,p = 0;

    if (buf == NULL)
    {
      return false;
    }
    MPI_Recv(buf,(int) (field->nx*field->ny),MPI_DOUBLE,0,tag_base,dom->comm,MPI_STATUS_IGNORE);
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
  copy_global_to_local_field(field,global,dom);
  (void) tag_base;
  return true;
#endif
}

/*
 * Build face thicknesses for a serial pressure operator.
 *
 * Wall boundary conditions are imposed by zeroing the corresponding boundary
 * face coefficient outside this helper; periodic cases keep the modular
 * indexing used here.
 */
/** Build global west/south pressure coefficients from cell-centered Hc. */
static void compute_global_face_thickness (uint Nx, uint Ny, real ** Hc,
                                           real ** Hw, real ** Hs)
{
  uint i,j;

  for (i = 0; i < Nx; i ++)
  {
    uint im1 = (i+Nx-1) % Nx;

    for (j = 0; j < Ny; j ++)
    {
      uint jm1 = (j+Ny-1) % Ny;

      Hw[i][j] = 0.5*(Hc[i][j] + Hc[im1][j]);
      Hs[i][j] = 0.5*(Hc[i][j] + Hc[i][jm1]);
    }
  }
}

/** West/east coefficient for the serial variable-coefficient pressure stencil. */
/** Global-matrix version of the x-face pressure-operator coefficient. */
static real global_west_operator (uint Nx, real dx, real ** Hw, uint i, uint j,
                                  bool useWallEW)
{
  if (useWallEW && (i == 0 || i == Nx))
  {
    return 0;
  }

  return Hw[i % Nx][j] / SQUARE(dx);
}

/** South/north coefficient for the serial variable-coefficient pressure stencil. */
/** Global-matrix version of the y-face pressure-operator coefficient. */
static real global_south_operator (uint Ny, real dy, real ** Hs, uint i, uint j,
                                   bool useWallNS)
{
  if (useWallNS && (j == 0 || j == Ny))
  {
    return 0;
  }

  return Hs[i][j % Ny] / SQUARE(dy);
}

/** Mean removal for rank-0 matrices used by gathered pressure paths. */
/** Remove the constant pressure null mode from a rank-0 global matrix. */
static void subtract_global_matrix_mean (uint Nx, uint Ny, real ** field)
{
  uint i,j;
  real mean = 0;

  for (i = 0; i < Nx; i ++)
  {
    for (j = 0; j < Ny; j ++)
    {
      mean += field[i][j];
    }
  }
  mean /= ((real) Nx * (real) Ny);

  for (i = 0; i < Nx; i ++)
  {
    for (j = 0; j < Ny; j ++)
    {
      field[i][j] -= mean;
    }
  }
}

/*
 * Small serial SOR solver used only inside the distributed-MG coarse tail.
 *
 * The main gathered-MG path still calls AWSIM's multigrid implementation. The
 * tail solver lives here instead so the distributed hierarchy can solve a
 * coarsened residual without mutating the static state in multigrid.c.
 */
/*
 * Serial red-black SOR on rank-0 global matrices. This is used by gathered
 * coarse-grid tails where a small, robust exact-ish solve is preferable to
 * further distributed coarsening.
 */
static uint solve_global_pressure_sor (uint Nx, uint Ny, real dx, real dy,
                                       real ** pi, real ** rhs, real ** Hc,
                                       bool useWallEW, bool useWallNS,
                                       real omega, real tol, uint maxiters)
{
  real ** Hw = matalloc(Nx,Ny);
  real ** Hs = matalloc(Nx,Ny);
  uint iters = 0;
  real max_update = tol + 1;

  if ((Hw == NULL) || (Hs == NULL))
  {
    if (Hw != NULL) matfree(Hw);
    if (Hs != NULL) matfree(Hs);
    return maxiters;
  }

  compute_global_face_thickness(Nx,Ny,Hc,Hw,Hs);
  if (useWallEW)
  {
    uint j;
    for (j = 0; j < Ny; j ++) Hw[0][j] = 0;
  }
  if (useWallNS)
  {
    uint i;
    for (i = 0; i < Nx; i ++) Hs[i][0] = 0;
  }

  while ((max_update > tol) && (iters < maxiters))
  {
    uint i,j,color;

    max_update = 0;
    for (color = 0; color < 2; color ++)
    {
      for (i = 0; i < Nx; i ++)
      {
        uint im1 = (i+Nx-1) % Nx;
        uint ip1 = (i+1) % Nx;

        for (j = 0; j < Ny; j ++)
        {
          uint jm1 = (j+Ny-1) % Ny;
          uint jp1 = (j+1) % Ny;
          real Ow = global_west_operator(Nx,dx,Hw,i,j,useWallEW);
          real Oe = global_west_operator(Nx,dx,Hw,i+1,j,useWallEW);
          real Os = global_south_operator(Ny,dy,Hs,i,j,useWallNS);
          real On = global_south_operator(Ny,dy,Hs,i,j+1,useWallNS);
          real Osum = Ow + Oe + Os + On;
          real old,newval;

          if (((i+j) & 1U) != color)
          {
            continue;
          }
          if (Osum == 0)
          {
            continue;
          }

          old = pi[i][j];
          newval = (1-omega)*old
                 + omega*(Oe*pi[ip1][j] + Ow*pi[im1][j]
                         + On*pi[i][jp1] + Os*pi[i][jm1] - rhs[i][j]) / Osum;
          pi[i][j] = newval;
          max_update = fmax(max_update,fabs(newval-old));
        }
      }
    }
    iters ++;
  }

  subtract_global_matrix_mean(Nx,Ny,pi);
  matfree(Hw);
  matfree(Hs);
  return iters;
}

/** Rank-0 2x2 cell average, used after gathering an awkward distributed level. */
/** Restrict a rank-0 even-sized global field by 2x2 averaging. */
static void restrict_global_power2 (uint Nx_f, uint Ny_f, real ** fine,
                                    real ** coarse)
{
  uint i,j;

  for (i = 0; i < Nx_f/2; i ++)
  {
    for (j = 0; j < Ny_f/2; j ++)
    {
      coarse[i][j] = 0.25*(fine[2*i][2*j] + fine[2*i+1][2*j]
                          + fine[2*i][2*j+1] + fine[2*i+1][2*j+1]);
    }
  }
}

/*
 * Enforce the rigid-lid column thickness constraint before the pressure solve.
 *
 * AWSIM keeps the layer sum equal to the fixed water-column thickness. PAWSIM
 * applies the same correction locally, then refreshes prognostic halos so the
 * pressure RHS sees consistent face thicknesses.
 */
void pawsim_pressure_correct_thickness (pawsim_context * ctx)
{
  uint i,j,k,g;

  if (ctx == NULL)
  {
    return;
  }

  g = ctx->dom.nghost;
  for (i = 0; i < ctx->dom.nx; i ++)
  {
    for (j = 0; j < ctx->dom.ny; j ++)
    {
      real h_tot = 0;
      real h_err;

      for (k = 0; k < ctx->cfg.Nlay; k ++)
      {
        h_tot += ctx->state.h[k].a[i+g][j+g];
      }

      if (h_tot == 0)
      {
        continue;
      }

      h_err = h_tot - ctx->work.Hc.a[i+g][j+g];
      for (k = 0; k < ctx->cfg.Nlay; k ++)
      {
        ctx->state.h[k].a[i+g][j+g] *= (1 - h_err / h_tot);
      }
    }
  }

  pawsim_state_exchange_prognostic_halos(&ctx->state,&ctx->dom);
}

/*
 * Form div(h u) / dt for the pressure equation.
 *
 * The subsequent solve finds pi such that subtracting dt*grad(pi) from every
 * layer velocity cancels this barotropic divergence. The RHS mean is removed
 * because the closed/periodic elliptic operator has a constant null space.
 */
static bool build_pressure_rhs (pawsim_context * ctx)
{
  uint i,j,k,g;

  pawsim_field2d_zero(&ctx->pi_rhs);
  pawsim_work_load_state(&ctx->work,&ctx->state,&ctx->dom);
  if (!pawsim_work_calc_face_thickness_no_ghost(&ctx->work,&ctx->cfg,&ctx->dom))
  {
    return false;
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
        real hu_w = ctx->state.u[k].a[il][jl] * ctx->work.h_west[k].a[il][jl];
        real hu_e = ctx->state.u[k].a[il+1][jl] * ctx->work.h_west[k].a[il+1][jl];
        real hv_s = ctx->state.v[k].a[il][jl] * ctx->work.h_south[k].a[il][jl];
        real hv_n = ctx->state.v[k].a[il][jl+1] * ctx->work.h_south[k].a[il][jl+1];

        ctx->pi_rhs.a[il][jl] += (hu_e - hu_w) / (ctx->dx * ctx->cfg.dt)
                               + (hv_n - hv_s) / (ctx->dy * ctx->cfg.dt);
      }
    }
  }

  pawsim_field2d_exchange_scalar_halo(&ctx->pi_rhs,&ctx->dom);
  subtract_field_mean(&ctx->pi_rhs,&ctx->dom);
  pawsim_field2d_exchange_scalar_halo(&ctx->pi_rhs,&ctx->dom);
  return true;
}

/** Local-domain x-face pressure-operator coefficient for the main solve. */
static real west_operator (const pawsim_context * ctx, uint i, uint j)
{
  uint gi = ctx->dom.i0 + i - ctx->dom.nghost;

  /*
   * H_w/dx^2 is the west/east coefficient in the variable-depth pressure
   * operator. Physical walls remove the face from the stencil.
   */
  if (ctx->cfg.useWallEW && ((gi == 0) || (gi == ctx->dom.Nx)))
  {
    return 0;
  }

  return ctx->work.Hw.a[i][j] / SQUARE(ctx->dx);
}

/** Local-domain y-face pressure-operator coefficient for the main solve. */
static real south_operator (const pawsim_context * ctx, uint i, uint j)
{
  uint gj = ctx->dom.j0 + j - ctx->dom.nghost;

  /** Same operator coefficient as west_operator(), but on south/north faces. */
  if (ctx->cfg.useWallNS && ((gj == 0) || (gj == ctx->dom.Ny)))
  {
    return 0;
  }

  return ctx->work.Hs.a[i][j] / SQUARE(ctx->dy);
}

/** Convert local max/sumsq residual pieces into global max and RMS norms. */
static void set_norms_from_local_sums (real local_max, real local_sumsq,
                                       const pawsim_domain * dom,
                                       real * global_max, real * global_rms)
{
  real global_sumsq = 0;

  MPI_Allreduce(&local_max,global_max,1,MPI_DOUBLE,MPI_MAX,dom->comm);
  MPI_Allreduce(&local_sumsq,&global_sumsq,1,MPI_DOUBLE,MPI_SUM,dom->comm);
  *global_rms = sqrt(global_sumsq / ((real) dom->Nx * (real) dom->Ny));
}

/** Diagnose the residual rhs - L(pi) of the pressure equation. */
static void compute_pressure_equation_residual (pawsim_context * ctx)
{
  uint i,j,g;
  real local_max = 0;
  real local_sumsq = 0;

  /*
   * Diagnostic only: evaluate rhs - L(pi) before the velocity update. This
   * measures the elliptic solve quality independently of the subsequent
   * pressure-gradient correction.
   */
  pawsim_field2d_exchange_scalar_halo(&ctx->pi,&ctx->dom);

  g = ctx->dom.nghost;
  for (i = 0; i < ctx->dom.nx; i ++)
  {
    uint il = i+g;

    for (j = 0; j < ctx->dom.ny; j ++)
    {
      uint jl = j+g;
      real Ow = west_operator(ctx,il,jl);
      real Oe = west_operator(ctx,il+1,jl);
      real Os = south_operator(ctx,il,jl);
      real On = south_operator(ctx,il,jl+1);
      real pc = ctx->pi.a[il][jl];
      real Lpi = Oe*(ctx->pi.a[il+1][jl]-pc) + Ow*(ctx->pi.a[il-1][jl]-pc)
               + On*(ctx->pi.a[il][jl+1]-pc) + Os*(ctx->pi.a[il][jl-1]-pc);
      real residual = ctx->pi_rhs.a[il][jl] - Lpi;

      local_max = fmax(local_max,fabs(residual));
      local_sumsq += residual*residual;
    }
  }

  set_norms_from_local_sums(local_max,local_sumsq,&ctx->dom,
                            &ctx->pressure_last_residual_max,
                            &ctx->pressure_last_residual_rms);
}

/** Diagnose the remaining barotropic divergence after velocity projection. */
static void compute_corrected_divergence (pawsim_context * ctx)
{
  uint i,j,k,g;
  real local_max = 0;
  real local_sumsq = 0;

  /*
   * Diagnostic only: after grad(pi) is subtracted from velocity, recompute
   * div(h u). This is the physically relevant rigid-lid check.
   */
  pawsim_work_load_state(&ctx->work,&ctx->state,&ctx->dom);
  if (!pawsim_work_calc_face_thickness_no_ghost(&ctx->work,&ctx->cfg,&ctx->dom))
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
      real divergence = 0;

      for (k = 0; k < ctx->cfg.Nlay; k ++)
      {
        real hu_w = ctx->state.u[k].a[il][jl] * ctx->work.h_west[k].a[il][jl];
        real hu_e = ctx->state.u[k].a[il+1][jl] * ctx->work.h_west[k].a[il+1][jl];
        real hv_s = ctx->state.v[k].a[il][jl] * ctx->work.h_south[k].a[il][jl];
        real hv_n = ctx->state.v[k].a[il][jl+1] * ctx->work.h_south[k].a[il][jl+1];

        divergence += (hu_e - hu_w) / ctx->dx + (hv_n - hv_s) / ctx->dy;
      }

      local_max = fmax(local_max,fabs(divergence));
      local_sumsq += divergence*divergence;
    }
  }

  set_norms_from_local_sums(local_max,local_sumsq,&ctx->dom,
                            &ctx->pressure_last_divergence_max,
                            &ctx->pressure_last_divergence_rms);
}

/*
 * Solve the pressure equation with distributed red-black SOR. This is simple
 * and fully MPI-decomposed, but usually slower than multigrid at useful sizes.
 */
static uint solve_pressure_sor (pawsim_context * ctx)
{
  uint iters = 0;
  uint i,j,g,color;
  real global_max = ctx->cfg.pi_tol + 1;

  g = ctx->dom.nghost;

  /*
   * Distributed red-black SOR. The color is based on global i+j parity so the
   * ordering is independent of how the grid is split across MPI ranks.
   */
  while ((global_max > ctx->cfg.pi_tol) && (iters < ctx->cfg.maxiters))
  {
    real local_iter_max = 0;

    for (color = 0; color < 2; color ++)
    {
      real local_color_max = 0;

      pawsim_field2d_exchange_scalar_halo(&ctx->pi,&ctx->dom);

      for (i = 0; i < ctx->pi.nx; i ++)
      {
        uint il = i+g;
        uint gi = ctx->dom.i0+i;

        for (j = 0; j < ctx->pi.ny; j ++)
        {
          uint jl = j+g;
          uint gj = ctx->dom.j0+j;
          real Ow = west_operator(ctx,il,jl);
          real Oe = west_operator(ctx,il+1,jl);
          real Os = south_operator(ctx,il,jl);
          real On = south_operator(ctx,il,jl+1);
          real Osum = Ow + Oe + Os + On;
          real pi_prev,pi_new,diff;

          if (((gi+gj) & 1U) != color)
          {
            continue;
          }

          if (Osum == 0)
          {
            continue;
          }

          pi_prev = ctx->pi.a[il][jl];
          pi_new = (1-ctx->cfg.SOR_rp)*pi_prev
                 + ctx->cfg.SOR_rp / Osum
                 * (On*ctx->pi.a[il][jl+1] + Os*ctx->pi.a[il][jl-1]
                  + Oe*ctx->pi.a[il+1][jl] + Ow*ctx->pi.a[il-1][jl]
                  - ctx->pi_rhs.a[il][jl]);
          ctx->pi.a[il][jl] = pi_new;

          diff = fabs(pi_new-pi_prev);
          local_color_max = fmax(local_color_max,diff);
        }
      }

      local_iter_max = fmax(local_iter_max,local_color_max);
    }

    MPI_Allreduce(&local_iter_max,&global_max,1,MPI_DOUBLE,MPI_MAX,ctx->dom.comm);
    iters ++;
  }

  subtract_field_mean(&ctx->pi,&ctx->dom);
  pawsim_field2d_exchange_scalar_halo(&ctx->pi,&ctx->dom);
  return iters;
}

/*
 * Solve pressure by gathering to rank 0 and calling AWSIM's serial multigrid.
 * This is the closest compatibility path and a fallback for awkward failures.
 */
static uint solve_pressure_mg_gathered (pawsim_context * ctx)
{
  static bool mg_initialized = false;
  real ** global_Hc = NULL;
  real ** global_pi = NULL;
  real ** global_rhs = NULL;
  uint iters = 0;
  bool ok = true;

  /*
   * Gathered MG is both a compatibility path and a useful fallback when the
   * distributed hierarchy cannot be built. It should match AWSIM's serial
   * solver more closely than the distributed path, at the cost of rank-0 work.
   */
  if (ctx->dom.rank == 0)
  {
    global_Hc = matalloc(ctx->dom.Nx,ctx->dom.Ny);
    global_pi = matalloc(ctx->dom.Nx,ctx->dom.Ny);
    global_rhs = matalloc(ctx->dom.Nx,ctx->dom.Ny);
    ok = (global_Hc != NULL) && (global_pi != NULL) && (global_rhs != NULL);
  }

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,ctx->dom.comm);
    ok = (ok_int != 0);
  }
#endif
  if (!ok)
  {
    if (global_Hc != NULL) matfree(global_Hc);
    if (global_pi != NULL) matfree(global_pi);
    if (global_rhs != NULL) matfree(global_rhs);
    return ctx->cfg.maxiters;
  }

  /*
   * This path preserves AWSIM's multigrid behavior by gathering Hc/rhs/pi to
   * rank 0 and calling multigrid.c. Its static initialization means
   * it assumes the pressure operator geometry is fixed for the run.
   */
  if (!mg_initialized)
  {
    ok = gather_pressure_field(global_Hc,&ctx->work.Hc,&ctx->dom,300);
    if (ok && (ctx->dom.rank == 0))
    {
      ok = init_MG(ctx->dom.Nx,ctx->dom.Ny,ctx->cfg.Lx,ctx->cfg.Ly,global_Hc,
                   ctx->cfg.useWallEW,ctx->cfg.useWallNS,ctx->cfg.use_fullMG,
                   ctx->cfg.pi_tol,ctx->cfg.maxiters,ctx->cfg.omega_MG,false);
    }
#ifdef PAWSIM_USE_MPI
    {
      int ok_int = ok ? 1 : 0;
      MPI_Bcast(&ok_int,1,MPI_INT,0,ctx->dom.comm);
      ok = (ok_int != 0);
    }
#endif
    if (!ok)
    {
      if (global_Hc != NULL) matfree(global_Hc);
      if (global_pi != NULL) matfree(global_pi);
      if (global_rhs != NULL) matfree(global_rhs);
      return ctx->cfg.maxiters;
    }
    mg_initialized = true;
  }

  ok = gather_pressure_field(global_pi,&ctx->pi,&ctx->dom,310)
    && gather_pressure_field(global_rhs,&ctx->pi_rhs,&ctx->dom,320);
  if (ok && (ctx->dom.rank == 0))
  {
    iters = solve_MG(global_pi,global_rhs);
  }

#ifdef PAWSIM_USE_MPI
  MPI_Bcast(&iters,1,MPI_UNSIGNED,0,ctx->dom.comm);
#endif
  if (ok)
  {
    ok = scatter_pressure_field(&ctx->pi,global_pi,&ctx->dom,330);
  }

  if (global_Hc != NULL) matfree(global_Hc);
  if (global_pi != NULL) matfree(global_pi);
  if (global_rhs != NULL) matfree(global_rhs);

  if (!ok)
  {
    return ctx->cfg.maxiters;
  }

  pawsim_field2d_exchange_scalar_halo(&ctx->pi,&ctx->dom);
  return iters;
}

#define PAWSIM_DMG_MAX_LEVELS 32
#define PAWSIM_DMG_PRE_SMOOTH 3
#define PAWSIM_DMG_POST_SMOOTH 3
#define PAWSIM_DMG_COARSE_SMOOTH 80
#define PAWSIM_DMG_MIN_GLOBAL 8

typedef struct pawsim_dmg_level
{
  pawsim_domain dom;
  /** x is the pressure correction on this level; rhs is the residual equation. */
  pawsim_field2d x;
  pawsim_field2d rhs;
  pawsim_field2d res;
  /** Coarsened cell and face thicknesses define the variable-coefficient operator. */
  pawsim_field2d Hc;
  pawsim_field2d Hw;
  pawsim_field2d Hs;
  real dx;
  real dy;
}
pawsim_dmg_level;

typedef struct pawsim_dmg_hierarchy
{
  uint nlevels;
  pawsim_dmg_level level[PAWSIM_DMG_MAX_LEVELS];
  bool has_gathered_tail;
}
pawsim_dmg_hierarchy;

/** Coarsen one grid dimension by two, keeping a minimum size of one. */
static uint coarsen_size (uint n)
{
  return (n+1) / 2;
}

/** Free one distributed multigrid level. */
static void dmg_level_free (pawsim_dmg_level * lev)
{
  pawsim_field2d_free(&lev->x);
  pawsim_field2d_free(&lev->rhs);
  pawsim_field2d_free(&lev->res);
  pawsim_field2d_free(&lev->Hc);
  pawsim_field2d_free(&lev->Hw);
  pawsim_field2d_free(&lev->Hs);
}

/** Free all allocated distributed multigrid levels. */
static void dmg_hierarchy_free (pawsim_dmg_hierarchy * mg)
{
  uint l;

  if (mg == NULL)
  {
    return;
  }

  for (l = 0; l < mg->nlevels; l ++)
  {
    dmg_level_free(&mg->level[l]);
  }
  mg->nlevels = 0;
}

/** Create the domain metadata for a coarser multigrid level on the same ranks. */
static void dmg_init_level_domain (pawsim_domain * levdom, const pawsim_domain * basedom,
                                   uint Nx, uint Ny)
{
  memset(levdom,0,sizeof(*levdom));
  levdom->comm = basedom->comm;
  levdom->rank = basedom->rank;
  levdom->size = basedom->size;
  levdom->dims[0] = basedom->dims[0];
  levdom->dims[1] = basedom->dims[1];
  levdom->coords[0] = basedom->coords[0];
  levdom->coords[1] = basedom->coords[1];
  levdom->nbr_w = basedom->nbr_w;
  levdom->nbr_e = basedom->nbr_e;
  levdom->nbr_s = basedom->nbr_s;
  levdom->nbr_n = basedom->nbr_n;
  levdom->Nx = Nx;
  levdom->Ny = Ny;
  levdom->nghost = 1;
  levdom->periodic_x = basedom->periodic_x;
  levdom->periodic_y = basedom->periodic_y;
  /*
   * Every multigrid level is decomposed over the same Cartesian rank topology.
   * Distributed restriction/prolongation is only used when adjacent levels are
   * cleanly nested rank by rank.
   */
  levdom->i0 = block_start_rank(Nx,(uint) levdom->coords[0],(uint) levdom->dims[0]);
  levdom->j0 = block_start_rank(Ny,(uint) levdom->coords[1],(uint) levdom->dims[1]);
  levdom->nx = block_size_rank(Nx,(uint) levdom->coords[0],(uint) levdom->dims[0]);
  levdom->ny = block_size_rank(Ny,(uint) levdom->coords[1],(uint) levdom->dims[1]);
}

/** Test whether fine/coarse domains are locally nested by an exact 2:1 ratio. */
static bool dmg_nested_with_previous_level (const pawsim_dmg_level * fine,
                                            const pawsim_dmg_level * coarse)
{
  /*
   * Local 2:1 nesting lets each rank restrict/prolong without communicating
   * off-rank fine cells. Non-nested transitions are handled by the gathered
   * coarse tail instead.
   */
  return (fine->dom.Nx == 2*coarse->dom.Nx)
      && (fine->dom.Ny == 2*coarse->dom.Ny)
      && (fine->dom.i0 == 2*coarse->dom.i0)
      && (fine->dom.j0 == 2*coarse->dom.j0)
      && (fine->dom.nx == 2*coarse->dom.nx)
      && (fine->dom.ny == 2*coarse->dom.ny);
}

/*
 * Build a distributed multigrid hierarchy. Coarsening proceeds while every
 * rank's local tile remains a clean 2:1 child of the next coarser tile.
 */
static bool dmg_build_hierarchy (pawsim_dmg_hierarchy * mg, const pawsim_context * ctx)
{
  uint Nx,Ny;

  memset(mg,0,sizeof(*mg));
  /*
   * Current design: use distributed levels as far as they are clean, then use
   * a gathered tail. This keeps pressureSolver 2 valid for arbitrary grids,
   * while still giving the efficient path on power-of-two cases.
   */
  mg->has_gathered_tail = true;

  Nx = ctx->dom.Nx;
  Ny = ctx->dom.Ny;
  while (mg->nlevels < PAWSIM_DMG_MAX_LEVELS)
  {
    pawsim_dmg_level * lev = &mg->level[mg->nlevels];
    dmg_init_level_domain(&lev->dom,&ctx->dom,Nx,Ny);

    if ((lev->dom.nx == 0) || (lev->dom.ny == 0)
     || !pawsim_field2d_alloc(&lev->x,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
     || !pawsim_field2d_alloc(&lev->rhs,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
     || !pawsim_field2d_alloc(&lev->res,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
     || !pawsim_field2d_alloc(&lev->Hc,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
     || !pawsim_field2d_alloc(&lev->Hw,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
     || !pawsim_field2d_alloc(&lev->Hs,lev->dom.nx,lev->dom.ny,lev->dom.nghost))
    {
      dmg_hierarchy_free(mg);
      return false;
    }
    lev->dx = ctx->cfg.Lx / Nx;
    lev->dy = ctx->cfg.Ly / Ny;
    mg->nlevels ++;

    if ((Nx <= PAWSIM_DMG_MIN_GLOBAL) || (Ny <= PAWSIM_DMG_MIN_GLOBAL)
     || (Nx % 2 != 0) || (Ny % 2 != 0))
    {
      break;
    }

    /*
     * Stop before a 2:1 transition that would leave an odd global level.
     * The gathered coarse tail can handle that case in a rank-independent
     * way, while distributed local restriction would depend on how the odd
     * level is split across MPI ranks.
     */
    if ((Nx % 4 != 0) || (Ny % 4 != 0))
    {
      break;
    }

    {
      pawsim_domain coarse_dom;
      uint Nx_c = coarsen_size(Nx);
      uint Ny_c = coarsen_size(Ny);

      dmg_init_level_domain(&coarse_dom,&ctx->dom,Nx_c,Ny_c);
      if ((coarse_dom.nx == 0) || (coarse_dom.ny == 0))
      {
        break;
      }

      if (!((Nx == 2*Nx_c) && (Ny == 2*Ny_c)
         && (mg->level[mg->nlevels-1].dom.i0 == 2*coarse_dom.i0)
         && (mg->level[mg->nlevels-1].dom.j0 == 2*coarse_dom.j0)
         && (mg->level[mg->nlevels-1].dom.nx == 2*coarse_dom.nx)
         && (mg->level[mg->nlevels-1].dom.ny == 2*coarse_dom.ny)))
      {
        break;
      }

      Nx = Nx_c;
      Ny = Ny_c;
    }

    if (mg->nlevels > 1
     && !dmg_nested_with_previous_level(&mg->level[mg->nlevels-2],
                                        &mg->level[mg->nlevels-1]))
    {
      dmg_hierarchy_free(mg);
      return false;
    }
  }

  return true;
}

/** Copy the context pressure equation into distributed-MG level 0. */
static void dmg_copy_ctx_to_level0 (pawsim_dmg_level * lev, const pawsim_context * ctx)
{
  uint i,j,g0,g;

  g0 = ctx->dom.nghost;
  g = lev->dom.nghost;
  pawsim_field2d_zero(&lev->x);
  pawsim_field2d_zero(&lev->rhs);
  pawsim_field2d_zero(&lev->res);
  pawsim_field2d_zero(&lev->Hc);
  pawsim_field2d_zero(&lev->Hw);
  pawsim_field2d_zero(&lev->Hs);

  for (i = 0; i < lev->dom.nx; i ++)
  {
    for (j = 0; j < lev->dom.ny; j ++)
    {
      lev->x.a[i+g][j+g] = ctx->pi.a[i+g0][j+g0];
      lev->rhs.a[i+g][j+g] = ctx->pi_rhs.a[i+g0][j+g0];
      lev->Hc.a[i+g][j+g] = ctx->work.Hc.a[i+g0][j+g0];
      lev->Hw.a[i+g][j+g] = ctx->work.Hw.a[i+g0][j+g0];
      lev->Hs.a[i+g][j+g] = ctx->work.Hs.a[i+g0][j+g0];
    }
  }

  pawsim_field2d_exchange_scalar_halo(&lev->x,&lev->dom);
  pawsim_field2d_exchange_scalar_halo(&lev->rhs,&lev->dom);
  pawsim_field2d_exchange_scalar_halo(&lev->Hc,&lev->dom);
  pawsim_field2d_exchange_scalar_halo(&lev->Hw,&lev->dom);
  pawsim_field2d_exchange_scalar_halo(&lev->Hs,&lev->dom);
}

/** Copy the converged distributed-MG level-0 pressure back to the context. */
static void dmg_copy_level0_to_ctx (pawsim_context * ctx, const pawsim_dmg_level * lev)
{
  uint i,j,g0,g;

  g0 = ctx->dom.nghost;
  g = lev->dom.nghost;
  for (i = 0; i < lev->dom.nx; i ++)
  {
    for (j = 0; j < lev->dom.ny; j ++)
    {
      ctx->pi.a[i+g0][j+g0] = lev->x.a[i+g][j+g];
    }
  }
}

/** x-face pressure-operator coefficient for a distributed multigrid level. */
static real dmg_west_operator (const pawsim_dmg_level * lev, uint i, uint j,
                               bool useWallEW)
{
  uint gi = lev->dom.i0 + i - lev->dom.nghost;

  if (useWallEW && ((gi == 0) || (gi == lev->dom.Nx)))
  {
    return 0;
  }

  return lev->Hw.a[i][j] / SQUARE(lev->dx);
}

/** y-face pressure-operator coefficient for a distributed multigrid level. */
static real dmg_south_operator (const pawsim_dmg_level * lev, uint i, uint j,
                                bool useWallNS)
{
  uint gj = lev->dom.j0 + j - lev->dom.nghost;

  if (useWallNS && ((gj == 0) || (gj == lev->dom.Ny)))
  {
    return 0;
  }

  return lev->Hs.a[i][j] / SQUARE(lev->dy);
}

/** Apply the distributed level operator at one local cell. */
static real dmg_apply_operator_point (const pawsim_dmg_level * lev, uint i, uint j,
                                      bool useWallEW, bool useWallNS)
{
  real Ow = dmg_west_operator(lev,i,j,useWallEW);
  real Oe = dmg_west_operator(lev,i+1,j,useWallEW);
  real Os = dmg_south_operator(lev,i,j,useWallNS);
  real On = dmg_south_operator(lev,i,j+1,useWallNS);
  real xc = lev->x.a[i][j];

  return Oe*(lev->x.a[i+1][j]-xc) + Ow*(lev->x.a[i-1][j]-xc)
       + On*(lev->x.a[i][j+1]-xc) + Os*(lev->x.a[i][j-1]-xc);
}

/** Apply weighted Jacobi smoothing on one distributed multigrid level. */
static real dmg_jacobi_smooth (pawsim_dmg_level * lev, const pawsim_config * cfg,
                               uint nsweeps)
{
  uint sweep,i,j,g;
  real global_max = 0;

  /*
   * Weighted Jacobi is easy to parallelize because each sweep uses halo values
   * from the previous sweep. The smoother is not optimal, but it is a stable
   * first distributed-MG building block.
   */
  g = lev->dom.nghost;
  for (sweep = 0; sweep < nsweeps; sweep ++)
  {
    real local_max = 0;

    pawsim_field2d_exchange_scalar_halo(&lev->x,&lev->dom);
    for (i = 0; i < lev->dom.nx; i ++)
    {
      uint il = i+g;

      for (j = 0; j < lev->dom.ny; j ++)
      {
        uint jl = j+g;
        real Ow = dmg_west_operator(lev,il,jl,cfg->useWallEW);
        real Oe = dmg_west_operator(lev,il+1,jl,cfg->useWallEW);
        real Os = dmg_south_operator(lev,il,jl,cfg->useWallNS);
        real On = dmg_south_operator(lev,il,jl+1,cfg->useWallNS);
        real Osum = Ow + Oe + Os + On;
        real x_old,x_jacobi,x_new,diff;

        if (Osum == 0)
        {
          continue;
        }

        x_old = lev->x.a[il][jl];
        x_jacobi = (Oe*lev->x.a[il+1][jl] + Ow*lev->x.a[il-1][jl]
                  + On*lev->x.a[il][jl+1] + Os*lev->x.a[il][jl-1]
                  - lev->rhs.a[il][jl]) / Osum;
        x_new = x_old + cfg->omega_MG*(x_jacobi-x_old);
        lev->x.a[il][jl] = x_new;
        diff = fabs(x_new-x_old);
        local_max = fmax(local_max,diff);
      }
    }
    MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,lev->dom.comm);
  }

  pawsim_field2d_exchange_scalar_halo(&lev->x,&lev->dom);
  return global_max;
}

/** Compute and halo-refresh the residual on one distributed multigrid level. */
static real dmg_residual (pawsim_dmg_level * lev, const pawsim_config * cfg)
{
  uint i,j,g;
  real local_max = 0;
  real global_max = 0;

  pawsim_field2d_exchange_scalar_halo(&lev->x,&lev->dom);
  g = lev->dom.nghost;
  for (i = 0; i < lev->dom.nx; i ++)
  {
    uint il = i+g;

    for (j = 0; j < lev->dom.ny; j ++)
    {
      uint jl = j+g;
      real r = lev->rhs.a[il][jl]
             - dmg_apply_operator_point(lev,il,jl,cfg->useWallEW,cfg->useWallNS);

      lev->res.a[il][jl] = r;
      local_max = fmax(local_max,fabs(r));
    }
  }
  pawsim_field2d_exchange_scalar_halo(&lev->res,&lev->dom);
  MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,lev->dom.comm);
  return global_max;
}

/** Restrict a cell-centered field by 2x2 full weighting on nested local tiles. */
static void dmg_restrict_field (pawsim_field2d * coarse, const pawsim_domain * cdom,
                                const pawsim_field2d * fine, const pawsim_domain * fdom)
{
  uint i,j,gc,gf;

  (void) fdom;
  gc = cdom->nghost;
  gf = fine->nghost;
  pawsim_field2d_zero(coarse);

  /** Cell-centered full weighting on a locally nested 2x2 fine-cell block. */
  for (i = 0; i < cdom->nx; i ++)
  {
    for (j = 0; j < cdom->ny; j ++)
    {
      uint ifn = 2*i + gf;
      uint jfn = 2*j + gf;

      coarse->a[i+gc][j+gc] = 0.25*(fine->a[ifn][jfn]
                                  + fine->a[ifn+1][jfn]
                                  + fine->a[ifn][jfn+1]
                                  + fine->a[ifn+1][jfn+1]);
    }
  }
  pawsim_field2d_exchange_scalar_halo(coarse,cdom);
}

/** Restrict west-face operator coefficients to the next coarser level. */
static void dmg_restrict_west_faces (pawsim_field2d * coarse, const pawsim_domain * cdom,
                                     const pawsim_field2d * fine)
{
  uint i,j,gc,gf;

  gc = cdom->nghost;
  gf = fine->nghost;
  pawsim_field2d_zero(coarse);

  /** West-face coefficients are averaged along the fine-grid y direction. */
  for (i = 0; i < cdom->nx; i ++)
  {
    for (j = 0; j < cdom->ny; j ++)
    {
      uint ifn = 2*i + gf;
      uint jfn = 2*j + gf;

      coarse->a[i+gc][j+gc] = 0.5*(fine->a[ifn][jfn]
                                  + fine->a[ifn][jfn+1]);
    }
  }
  pawsim_field2d_exchange_scalar_halo(coarse,cdom);
}

/** Restrict south-face operator coefficients to the next coarser level. */
static void dmg_restrict_south_faces (pawsim_field2d * coarse, const pawsim_domain * cdom,
                                      const pawsim_field2d * fine)
{
  uint i,j,gc,gf;

  gc = cdom->nghost;
  gf = fine->nghost;
  pawsim_field2d_zero(coarse);

  /** South-face coefficients are averaged along the fine-grid x direction. */
  for (i = 0; i < cdom->nx; i ++)
  {
    for (j = 0; j < cdom->ny; j ++)
    {
      uint ifn = 2*i + gf;
      uint jfn = 2*j + gf;

      coarse->a[i+gc][j+gc] = 0.5*(fine->a[ifn][jfn]
                                  + fine->a[ifn+1][jfn]);
    }
  }
  pawsim_field2d_exchange_scalar_halo(coarse,cdom);
}

/** Prolong a nested coarse correction and add it to the fine pressure field. */
static real dmg_prolong_add (pawsim_field2d * fine, const pawsim_domain * fdom,
                             pawsim_field2d * coarse, const pawsim_domain * cdom)
{
  uint i,j,gc,gf;
  real local_max = 0;
  real global_max = 0;

  (void) fdom;
  gc = cdom->nghost;
  gf = fine->nghost;
  /*
   * Bilinear correction prolongation for clean 2:1 local nesting. The weights
   * map one coarse correction to the four surrounding fine cell centers.
   */
  pawsim_field2d_exchange_scalar_halo(coarse,cdom);
  for (i = 0; i < cdom->nx; i ++)
  {
    for (j = 0; j < cdom->ny; j ++)
    {
      uint ic = i+gc;
      uint jc = j+gc;
      real e_sw = 0.5625*coarse->a[ic][jc]
                + 0.1875*coarse->a[ic-1][jc]
                + 0.1875*coarse->a[ic][jc-1]
                + 0.0625*coarse->a[ic-1][jc-1];
      real e_se = 0.5625*coarse->a[ic][jc]
                + 0.1875*coarse->a[ic+1][jc]
                + 0.1875*coarse->a[ic][jc-1]
                + 0.0625*coarse->a[ic+1][jc-1];
      real e_nw = 0.5625*coarse->a[ic][jc]
                + 0.1875*coarse->a[ic-1][jc]
                + 0.1875*coarse->a[ic][jc+1]
                + 0.0625*coarse->a[ic-1][jc+1];
      real e_ne = 0.5625*coarse->a[ic][jc]
                + 0.1875*coarse->a[ic+1][jc]
                + 0.1875*coarse->a[ic][jc+1]
                + 0.0625*coarse->a[ic+1][jc+1];
      uint ifn = 2*i + gf;
      uint jfn = 2*j + gf;

      fine->a[ifn][jfn] += e_sw;
      fine->a[ifn+1][jfn] += e_se;
      fine->a[ifn][jfn+1] += e_nw;
      fine->a[ifn+1][jfn+1] += e_ne;
      local_max = fmax(local_max,fabs(e_sw));
      local_max = fmax(local_max,fabs(e_se));
      local_max = fmax(local_max,fabs(e_nw));
      local_max = fmax(local_max,fabs(e_ne));
    }
  }
  pawsim_field2d_exchange_scalar_halo(fine,fdom);
  MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,fdom->comm);
  return global_max;
}

/** Convert raw interpolation indices to periodic or wall-clamped indices. */
static uint interp_global_index (int idx, uint N, bool use_wall)
{
  if (N <= 1)
  {
    return 0;
  }

  if (use_wall)
  {
    if (idx < 0)
    {
      return 0;
    }
    if (idx >= (int) N)
    {
      return N-1;
    }
    return (uint) idx;
  }

  return (uint) ((idx + (int) N) % (int) N);
}

/*
 * Prolong a rank-0 global coarse correction to a distributed fine level. This
 * handles arbitrary ratios after the solve switches to the gathered tail.
 */
static real dmg_prolong_add_global_coarse (pawsim_dmg_level * fine,
                                           real ** coarse,
                                           uint Nx_c, uint Ny_c,
                                           const pawsim_config * cfg)
{
  uint i,j,g;
  real local_max = 0;
  real global_max = 0;

  /*
   * Prolong a gathered coarse correction onto a distributed fine level. This
   * handles arbitrary coarse/fine size ratios after the hierarchy switches to
   * the gathered tail.
   */
  g = fine->dom.nghost;
  for (i = 0; i < fine->dom.nx; i ++)
  {
    uint gi = fine->dom.i0+i;
    real x = (((real) gi) + 0.5) * Nx_c / fine->dom.Nx - 0.5;
    int im_raw;
    real tx;
    uint im,ip;

    if (cfg->useWallEW && (x <= 0))
    {
      im_raw = 0;
      tx = 0;
    }
    else if (cfg->useWallEW && (x >= Nx_c-1))
    {
      im_raw = (int) Nx_c-1;
      tx = 0;
    }
    else
    {
      im_raw = (int) floor(x);
      tx = x - im_raw;
    }
    im = interp_global_index(im_raw,Nx_c,cfg->useWallEW);
    ip = interp_global_index(im_raw+1,Nx_c,cfg->useWallEW);

    for (j = 0; j < fine->dom.ny; j ++)
    {
      uint gj = fine->dom.j0+j;
      real y = (((real) gj) + 0.5) * Ny_c / fine->dom.Ny - 0.5;
      int jm_raw;
      real ty;
      uint jm,jp;
      real e;

      if (cfg->useWallNS && (y <= 0))
      {
        jm_raw = 0;
        ty = 0;
      }
      else if (cfg->useWallNS && (y >= Ny_c-1))
      {
        jm_raw = (int) Ny_c-1;
        ty = 0;
      }
      else
      {
        jm_raw = (int) floor(y);
        ty = y - jm_raw;
      }
      jm = interp_global_index(jm_raw,Ny_c,cfg->useWallNS);
      jp = interp_global_index(jm_raw+1,Ny_c,cfg->useWallNS);

      e = (1-tx)*(1-ty)*coarse[im][jm]
        + tx*(1-ty)*coarse[ip][jm]
        + (1-tx)*ty*coarse[im][jp]
        + tx*ty*coarse[ip][jp];
      fine->x.a[i+g][j+g] += e;
      local_max = fmax(local_max,fabs(e));
    }
  }

  pawsim_field2d_exchange_scalar_halo(&fine->x,&fine->dom);
  MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,fine->dom.comm);
  return global_max;
}

/** Gather the current coarse level and solve it serially on rank 0. */
static real dmg_gathered_tail_solve (pawsim_dmg_level * lev, const pawsim_config * cfg)
{
  real ** global_Hc = NULL;
  real ** global_x = NULL;
  real ** global_rhs = NULL;
  real max_update = 0;
  bool ok = true;

  /*
   * Final coarse solve for levels that can no longer be coarsened cleanly. It
   * is serial by design; useful for robustness, but a known scaling bottleneck.
   */
  if (lev->dom.rank == 0)
  {
    global_Hc = matalloc(lev->dom.Nx,lev->dom.Ny);
    global_x = matalloc(lev->dom.Nx,lev->dom.Ny);
    global_rhs = matalloc(lev->dom.Nx,lev->dom.Ny);
    ok = (global_Hc != NULL) && (global_x != NULL) && (global_rhs != NULL);
  }

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,lev->dom.comm);
    ok = (ok_int != 0);
  }
#endif

  if (ok)
  {
    ok = gather_pressure_field(global_Hc,&lev->Hc,&lev->dom,400)
      && gather_pressure_field(global_x,&lev->x,&lev->dom,410)
      && gather_pressure_field(global_rhs,&lev->rhs,&lev->dom,420);
  }

  if (ok && (lev->dom.rank == 0))
  {
    (void) solve_global_pressure_sor(lev->dom.Nx,lev->dom.Ny,lev->dx,lev->dy,
                                     global_x,global_rhs,global_Hc,
                                     cfg->useWallEW,cfg->useWallNS,
                                     cfg->SOR_rp,cfg->pi_tol,cfg->maxiters);
  }

  if (ok)
  {
    ok = scatter_pressure_field(&lev->x,global_x,&lev->dom,430);
  }

#ifdef PAWSIM_USE_MPI
  MPI_Bcast(&max_update,1,MPI_DOUBLE,0,lev->dom.comm);
#endif

  if (global_Hc != NULL) matfree(global_Hc);
  if (global_x != NULL) matfree(global_x);
  if (global_rhs != NULL) matfree(global_rhs);

  if (!ok)
  {
    return cfg->pi_tol + 1;
  }

  pawsim_field2d_exchange_scalar_halo(&lev->x,&lev->dom);
  return max_update;
}

/*
 * Gather the current level, coarsen once on rank 0, solve, then prolong the
 * correction back to the distributed fine level.
 */
static real dmg_gathered_coarsened_tail (pawsim_dmg_level * lev, const pawsim_config * cfg)
{
  uint Nx_c = lev->dom.Nx/2;
  uint Ny_c = lev->dom.Ny/2;
  real ** global_Hc = NULL;
  real ** global_res = NULL;
  real ** coarse_Hc = NULL;
  real ** coarse_rhs = NULL;
  real ** coarse_x = NULL;
  real * coarse_buf = NULL;
  real max_update = 0;
  bool ok = true;

  /*
   * One more coarsening step after gathering the distributed residual. This is
   * the compromise used by 250 x 250 tests: distributed fine smoothing plus a
   * rank-0 coarse correction, without falling back to gathered MG for the
   * entire pressure solve.
   */
  if ((lev->dom.Nx % 2 != 0) || (lev->dom.Ny % 2 != 0)
   || (Nx_c == 0) || (Ny_c == 0))
  {
    return dmg_gathered_tail_solve(lev,cfg);
  }

  if (lev->dom.rank == 0)
  {
    global_Hc = matalloc(lev->dom.Nx,lev->dom.Ny);
    global_res = matalloc(lev->dom.Nx,lev->dom.Ny);
    coarse_Hc = matalloc(Nx_c,Ny_c);
    coarse_rhs = matalloc(Nx_c,Ny_c);
    coarse_x = matalloc(Nx_c,Ny_c);
    ok = (global_Hc != NULL) && (global_res != NULL) && (coarse_Hc != NULL)
      && (coarse_rhs != NULL) && (coarse_x != NULL);
  }

  coarse_buf = malloc((size_t) Nx_c*Ny_c*sizeof(real));
  ok = ok && (coarse_buf != NULL);

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,lev->dom.comm);
    ok = (ok_int != 0);
  }
#endif

  if (ok)
  {
    ok = gather_pressure_field(global_Hc,&lev->Hc,&lev->dom,440)
      && gather_pressure_field(global_res,&lev->res,&lev->dom,450);
  }

  if (ok && (lev->dom.rank == 0))
  {
    restrict_global_power2(lev->dom.Nx,lev->dom.Ny,global_Hc,coarse_Hc);
    restrict_global_power2(lev->dom.Nx,lev->dom.Ny,global_res,coarse_rhs);
    memset(*coarse_x,0,(size_t) Nx_c*Ny_c*sizeof(real));
    (void) solve_global_pressure_sor(Nx_c,Ny_c,
                                     cfg->Lx/Nx_c,cfg->Ly/Ny_c,
                                     coarse_x,coarse_rhs,coarse_Hc,
                                     cfg->useWallEW,cfg->useWallNS,
                                     cfg->SOR_rp,cfg->pi_tol,cfg->maxiters);
    memcpy(coarse_buf,*coarse_x,(size_t) Nx_c*Ny_c*sizeof(real));
  }

  if (ok)
  {
    uint i;
    real ** coarse_cols = malloc(Nx_c*sizeof(real *));

#ifdef PAWSIM_USE_MPI
    MPI_Bcast(coarse_buf,(int) (Nx_c*Ny_c),MPI_DOUBLE,0,lev->dom.comm);
#endif

    if (coarse_cols == NULL)
    {
      ok = false;
    }
    else
    {
      for (i = 0; i < Nx_c; i ++)
      {
        coarse_cols[i] = coarse_buf + i*Ny_c;
      }
      max_update = dmg_prolong_add_global_coarse(lev,coarse_cols,Nx_c,Ny_c,cfg);
      free(coarse_cols);
    }
  }

  if (global_Hc != NULL) matfree(global_Hc);
  if (global_res != NULL) matfree(global_res);
  if (coarse_Hc != NULL) matfree(coarse_Hc);
  if (coarse_rhs != NULL) matfree(coarse_rhs);
  if (coarse_x != NULL) matfree(coarse_x);
  free(coarse_buf);

  if (!ok)
  {
    return cfg->pi_tol + 1;
  }

  return max_update;
}

/*
 * Recursive distributed multigrid V-cycle. Cleanly nested levels stay
 * distributed; the coarsest awkward level uses the gathered tail.
 */
static real dmg_vcycle (pawsim_dmg_hierarchy * mg, uint l, const pawsim_config * cfg)
{
  pawsim_dmg_level * lev = &mg->level[l];
  real max_update;

  if (l == mg->nlevels-1)
  {
    if (mg->has_gathered_tail)
    {
      if ((lev->dom.Nx > PAWSIM_DMG_MIN_GLOBAL) && (lev->dom.Ny > PAWSIM_DMG_MIN_GLOBAL)
       && (lev->dom.Nx % 2 == 0) && (lev->dom.Ny % 2 == 0))
      {
        max_update = dmg_jacobi_smooth(lev,cfg,PAWSIM_DMG_PRE_SMOOTH);
        dmg_residual(lev,cfg);
        max_update = fmax(max_update,dmg_gathered_coarsened_tail(lev,cfg));
        max_update = fmax(max_update,dmg_jacobi_smooth(lev,cfg,PAWSIM_DMG_POST_SMOOTH));
        return max_update;
      }
      return dmg_gathered_tail_solve(lev,cfg);
    }
    return dmg_jacobi_smooth(lev,cfg,PAWSIM_DMG_COARSE_SMOOTH);
  }

  max_update = dmg_jacobi_smooth(lev,cfg,PAWSIM_DMG_PRE_SMOOTH);
  dmg_residual(lev,cfg);
  dmg_restrict_field(&mg->level[l+1].rhs,&mg->level[l+1].dom,&lev->res,&lev->dom);
  dmg_restrict_field(&mg->level[l+1].Hc,&mg->level[l+1].dom,&lev->Hc,&lev->dom);
  dmg_restrict_west_faces(&mg->level[l+1].Hw,&mg->level[l+1].dom,&lev->Hw);
  dmg_restrict_south_faces(&mg->level[l+1].Hs,&mg->level[l+1].dom,&lev->Hs);
  pawsim_field2d_zero(&mg->level[l+1].x);
  max_update = fmax(max_update,dmg_vcycle(mg,l+1,cfg));
  max_update = fmax(max_update,dmg_prolong_add(&lev->x,&lev->dom,&mg->level[l+1].x,&mg->level[l+1].dom));
  max_update = fmax(max_update,dmg_jacobi_smooth(lev,cfg,PAWSIM_DMG_POST_SMOOTH));
  return max_update;
}

/*
 * Distributed multigrid pressure solve with gathered coarse tail. This is the
 * production MPI path for both power-of-two and arbitrary rectangular grids.
 */
static uint solve_pressure_mg_distributed (pawsim_context * ctx)
{
  pawsim_dmg_hierarchy mg;
  uint cycles = 0;
  real max_update = ctx->cfg.pi_tol + 1;

  if (!dmg_build_hierarchy(&mg,ctx))
  {
    if (ctx->dom.rank == 0)
    {
      fprintf(stderr,"WARNING: Could not build distributed multigrid hierarchy; using gathered MG\n");
    }
    ctx->pressure_last_solver = PAWSIM_PRESSURE_SOLVER_MG_GATHERED;
    ctx->pressure_last_fallback = true;
    return solve_pressure_mg_gathered(ctx);
  }

  /** Level 0 is initialized from the current pressure guess and pressure RHS. */
  dmg_copy_ctx_to_level0(&mg.level[0],ctx);
  while ((max_update > ctx->cfg.pi_tol) && (cycles < ctx->cfg.maxiters))
  {
    max_update = dmg_vcycle(&mg,0,&ctx->cfg);
    subtract_field_mean(&mg.level[0].x,&mg.level[0].dom);
    cycles ++;
  }

  dmg_copy_level0_to_ctx(ctx,&mg.level[0]);
  dmg_hierarchy_free(&mg);
  subtract_field_mean(&ctx->pi,&ctx->dom);
  pawsim_field2d_exchange_scalar_halo(&ctx->pi,&ctx->dom);

  return cycles;
}

/*
 * Project velocities with the solved pressure gradient and optionally add the
 * pressure-gradient contribution to momentum/energy diagnostics.
 */
static void apply_pressure_correction (pawsim_context * ctx, bool update_diags)
{
  uint i,j,k,g;

  /*
   * Subtract dt*grad(pi) from every layer. Wall-normal velocity points on
   * physical boundaries are skipped because the halo fill already enforces
   * no-through-flow there.
   */
  g = ctx->dom.nghost;
  for (k = 0; k < ctx->cfg.Nlay; k ++)
  {
    for (i = 0; i < ctx->dom.nx; i ++)
    {
      uint gi = ctx->dom.i0+i;
      uint il = i+g;

      if (ctx->cfg.useWallEW && (gi == 0))
      {
        continue;
      }

      for (j = 0; j < ctx->dom.ny; j ++)
      {
        uint jl = j+g;
        real rhs_u = -ctx->cfg.dt
                   * (ctx->pi.a[il][jl] - ctx->pi.a[il-1][jl]) / ctx->dx;
        ctx->state.u[k].a[il][jl] += rhs_u;
        if (update_diags)
        {
          diag_add(ctx->work.diag_umom,PAWSIM_UMOM_GRADM,k,il,jl,
                   ctx->work.h_west[k].a[il][jl]*rhs_u);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_GRADM,k,il,jl,
                   ctx->work.h_west[k].a[il][jl]*rhs_u*ctx->state.u[k].a[il][jl]);
        }
      }
    }

    for (j = 0; j < ctx->dom.ny; j ++)
    {
      uint gj = ctx->dom.j0+j;
      uint jl = j+g;

      if (ctx->cfg.useWallNS && (gj == 0))
      {
        continue;
      }

      for (i = 0; i < ctx->dom.nx; i ++)
      {
        uint il = i+g;
        real rhs_v = -ctx->cfg.dt
                   * (ctx->pi.a[il][jl] - ctx->pi.a[il][jl-1]) / ctx->dy;
        ctx->state.v[k].a[il][jl] += rhs_v;
        if (update_diags)
        {
          diag_add(ctx->work.diag_vmom,PAWSIM_VMOM_GRADM,k,il,jl,
                   ctx->work.h_south[k].a[il][jl]*rhs_v);
          diag_add(ctx->work.diag_energy,PAWSIM_ENERGY_GRADM,k,il,jl,
                   ctx->work.h_south[k].a[il][jl]*rhs_v*ctx->state.v[k].a[il][jl]);
        }
      }
    }
  }

  pawsim_state_exchange_prognostic_halos(&ctx->state,&ctx->dom);
}

/*
 * Public rigid-lid pressure-correction driver called from initialization and
 * every time step when useRL is enabled.
 */
uint pawsim_pressure_correct_rigid_lid (pawsim_context * ctx, bool update_diags)
{
  uint iters;

  if ((ctx == NULL) || !ctx->cfg.useRL)
  {
    return 0;
  }

  /*
   * The correction sequence mirrors AWSIM: refresh pressure geometry, enforce
   * the rigid-lid thickness constraint, build RHS, solve, apply grad(pi), then
   * refresh diagnostics if requested.
   */
  pawsim_work_init_static_geometry(&ctx->work,&ctx->state,&ctx->cfg,&ctx->dom);
  pawsim_pressure_correct_thickness(ctx);
  if (!build_pressure_rhs(ctx))
  {
    return ctx->cfg.maxiters;
  }
  ctx->pressure_last_solver = ctx->cfg.pressureSolver;
  ctx->pressure_last_fallback = false;
  switch (ctx->cfg.pressureSolver)
  {
    case PAWSIM_PRESSURE_SOLVER_MG_GATHERED:
      iters = solve_pressure_mg_gathered(ctx);
      break;
    case PAWSIM_PRESSURE_SOLVER_MG_DISTRIBUTED:
      iters = solve_pressure_mg_distributed(ctx);
      break;
    case PAWSIM_PRESSURE_SOLVER_SOR:
    default:
      iters = solve_pressure_sor(ctx);
      break;
  }
  if (ctx->cfg.pressureTiming)
  {
    compute_pressure_equation_residual(ctx);
  }
  apply_pressure_correction(ctx,update_diags);
  if (ctx->cfg.pressureTiming)
  {
    compute_corrected_divergence(ctx);
  }

  return iters;
}
