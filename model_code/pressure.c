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

/** Rank-0 restriction for west-face coefficients on an even global grid. */
static void restrict_global_west_faces_power2 (uint Nx_f, uint Ny_f, real ** fine,
                                               real ** coarse)
{
  uint i,j;

  for (i = 0; i < Nx_f/2; i ++)
  {
    for (j = 0; j < Ny_f/2; j ++)
    {
      coarse[i][j] = 0.5*(fine[2*i][2*j] + fine[2*i][2*j+1]);
    }
  }
}

/** Rank-0 restriction for south-face coefficients on an even global grid. */
static void restrict_global_south_faces_power2 (uint Nx_f, uint Ny_f, real ** fine,
                                                real ** coarse)
{
  uint i,j;

  for (i = 0; i < Nx_f/2; i ++)
  {
    for (j = 0; j < Ny_f/2; j ++)
    {
      coarse[i][j] = 0.5*(fine[2*i][2*j] + fine[2*i+1][2*j]);
    }
  }
}

/** Pack a rank-0 matrix into a contiguous buffer for communicator broadcasts. */
static void pack_global_matrix (real * buf, real ** global, uint Nx, uint Ny)
{
  uint i,j;

  for (i = 0; i < Nx; i ++)
  {
    for (j = 0; j < Ny; j ++)
    {
      buf[i*Ny+j] = global[i][j];
    }
  }
}

/** Build column pointers onto a flat global matrix buffer. */
static bool make_matrix_columns (real *** cols_out, real * buf, uint Nx, uint Ny)
{
  uint i;
  real ** cols = malloc(Nx*sizeof(real *));

  if (cols == NULL)
  {
    return false;
  }
  for (i = 0; i < Nx; i ++)
  {
    cols[i] = buf + i*Ny;
  }
  *cols_out = cols;
  return true;
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
#define PAWSIM_DMG_MIN_LOCAL 8

typedef struct pawsim_dmg_level
{
  pawsim_domain dom;
  bool active;
  bool owns_comm;
  bool transfer_gathered_from_fine;
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
#ifdef PAWSIM_USE_MPI
  if (lev->owns_comm)
  {
    MPI_Comm_free(&lev->dom.comm);
  }
#endif
  lev->active = false;
  lev->owns_comm = false;
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

/** Pick a coarser rank grid that avoids very small local coarse tiles. */
static void dmg_choose_coarse_dims (uint Nx, uint Ny, const int prev_dims[2],
                                    int dims[2])
{
  dims[0] = prev_dims[0];
  dims[1] = prev_dims[1];

  /*
   * Shrink dimensions independently. Power-of-two process grids therefore
   * agglomerate smoothly, while odd process counts fall back to fewer ranks
   * once a dimension can no longer be split profitably.
   */
  while ((dims[0] > 1) && (((Nx / (uint) dims[0]) < PAWSIM_DMG_MIN_LOCAL)
        || ((uint) dims[0] > Nx)))
  {
    dims[0] = (dims[0] % 2 == 0) ? dims[0]/2 : 1;
  }
  while ((dims[1] > 1) && (((Ny / (uint) dims[1]) < PAWSIM_DMG_MIN_LOCAL)
        || ((uint) dims[1] > Ny)))
  {
    dims[1] = (dims[1] % 2 == 0) ? dims[1]/2 : 1;
  }
}

/** Test one dimension for exact local 2:1 nesting with fixed rank count. */
static bool dmg_dim_is_nested_after_coarsen (uint N, int dim)
{
  uint coord;
  uint Nc;

  if ((dim <= 0) || (N % 2 != 0))
  {
    return false;
  }

  Nc = coarsen_size(N);
  for (coord = 0; coord < (uint) dim; coord ++)
  {
    uint fine_start = block_start_rank(N,coord,(uint) dim);
    uint fine_size = block_size_rank(N,coord,(uint) dim);
    uint coarse_start = block_start_rank(Nc,coord,(uint) dim);
    uint coarse_size = block_size_rank(Nc,coord,(uint) dim);

    if ((fine_start != 2*coarse_start) || (fine_size != 2*coarse_size))
    {
      return false;
    }
  }

  return true;
}

/*
 * Return true when level 0 can use at least one clean distributed coarsening.
 * Non-nested first transitions enter the fragile gathered-transfer path before
 * any useful distributed hierarchy exists, so they should use gathered MG.
 */
static bool dmg_can_start_with_local_transfers (const pawsim_domain * dom)
{
  if ((dom->Nx <= PAWSIM_DMG_MIN_GLOBAL) || (dom->Ny <= PAWSIM_DMG_MIN_GLOBAL)
   || (dom->Nx % 2 != 0) || (dom->Ny % 2 != 0))
  {
    return false;
  }

  return dmg_dim_is_nested_after_coarsen(dom->Nx,dom->dims[0])
      && dmg_dim_is_nested_after_coarsen(dom->Ny,dom->dims[1]);
}

/*
 * Create the domain metadata for one multigrid level.
 *
 * Level 0 reuses the model communicator. Coarser levels may use a smaller
 * Cartesian communicator made from the lowest-numbered ranks in MPI_COMM_WORLD;
 * ranks outside that active set keep active=false and skip stencil work until
 * the V-cycle returns to a level they own.
 */
static bool dmg_init_level_domain (pawsim_dmg_level * lev,
                                   const pawsim_domain * basedom,
                                   uint Nx, uint Ny, const int dims[2],
                                   bool reuse_base_comm)
{
  pawsim_domain * levdom = &lev->dom;

  memset(levdom,0,sizeof(*levdom));
  levdom->Nx = Nx;
  levdom->Ny = Ny;
  levdom->nghost = 1;
  levdom->periodic_x = basedom->periodic_x;
  levdom->periodic_y = basedom->periodic_y;

  lev->active = true;
  lev->owns_comm = false;

  if (reuse_base_comm)
  {
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
  }
  else
  {
#ifdef PAWSIM_USE_MPI
    int world_rank,world_size,active_size;
    int periods[2];
    MPI_Comm active_comm;

    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    active_size = dims[0]*dims[1];
    if ((active_size <= 0) || (active_size > world_size))
    {
      return false;
    }

    lev->active = (world_rank < active_size);
    MPI_Comm_split(MPI_COMM_WORLD,lev->active ? 0 : MPI_UNDEFINED,
                   world_rank,&active_comm);
    if (lev->active)
    {
      periods[0] = basedom->periodic_x ? 1 : 0;
      periods[1] = basedom->periodic_y ? 1 : 0;
      MPI_Cart_create(active_comm,2,(int *) dims,periods,0,&levdom->comm);
      MPI_Comm_free(&active_comm);
      lev->owns_comm = true;
      MPI_Comm_rank(levdom->comm,&levdom->rank);
      MPI_Comm_size(levdom->comm,&levdom->size);
      levdom->dims[0] = dims[0];
      levdom->dims[1] = dims[1];
      MPI_Cart_coords(levdom->comm,levdom->rank,2,levdom->coords);
      MPI_Cart_shift(levdom->comm,0,1,&levdom->nbr_w,&levdom->nbr_e);
      MPI_Cart_shift(levdom->comm,1,1,&levdom->nbr_s,&levdom->nbr_n);
    }
    else
    {
      levdom->comm = MPI_COMM_NULL;
      levdom->rank = -1;
      levdom->size = 0;
      levdom->dims[0] = dims[0];
      levdom->dims[1] = dims[1];
      levdom->coords[0] = 0;
      levdom->coords[1] = 0;
      levdom->nbr_w = MPI_PROC_NULL;
      levdom->nbr_e = MPI_PROC_NULL;
      levdom->nbr_s = MPI_PROC_NULL;
      levdom->nbr_n = MPI_PROC_NULL;
    }
#else
    (void) dims;
    levdom->comm = basedom->comm;
    levdom->rank = 0;
    levdom->size = 1;
    levdom->dims[0] = 1;
    levdom->dims[1] = 1;
    levdom->coords[0] = 0;
    levdom->coords[1] = 0;
    levdom->nbr_w = basedom->periodic_x ? 0 : MPI_PROC_NULL;
    levdom->nbr_e = basedom->periodic_x ? 0 : MPI_PROC_NULL;
    levdom->nbr_s = basedom->periodic_y ? 0 : MPI_PROC_NULL;
    levdom->nbr_n = basedom->periodic_y ? 0 : MPI_PROC_NULL;
#endif
  }

  if (!lev->active)
  {
    levdom->i0 = 0;
    levdom->j0 = 0;
    levdom->nx = 0;
    levdom->ny = 0;
    return true;
  }

  levdom->i0 = block_start_rank(Nx,(uint) levdom->coords[0],(uint) levdom->dims[0]);
  levdom->j0 = block_start_rank(Ny,(uint) levdom->coords[1],(uint) levdom->dims[1]);
  levdom->nx = block_size_rank(Nx,(uint) levdom->coords[0],(uint) levdom->dims[0]);
  levdom->ny = block_size_rank(Ny,(uint) levdom->coords[1],(uint) levdom->dims[1]);
  return (levdom->nx > 0) && (levdom->ny > 0);
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
 * Build a distributed multigrid hierarchy. Adjacent clean levels use local
 * transfer operators; agglomerated or otherwise non-nested transitions are
 * marked so the V-cycle can use gathered transfer operators there.
 */
static bool dmg_build_hierarchy (pawsim_dmg_hierarchy * mg, const pawsim_context * ctx)
{
  uint Nx,Ny;
  int dims[2];

  memset(mg,0,sizeof(*mg));
  /*
   * Use distributed levels throughout the hierarchy, but allow the active rank
   * grid to shrink once coarse tiles would become too small. A gathered coarse
   * tail remains available as the final robust solve on the bottom level.
   */
  mg->has_gathered_tail = true;

  Nx = ctx->dom.Nx;
  Ny = ctx->dom.Ny;
  dims[0] = ctx->dom.dims[0];
  dims[1] = ctx->dom.dims[1];
  while (mg->nlevels < PAWSIM_DMG_MAX_LEVELS)
  {
    pawsim_dmg_level * lev = &mg->level[mg->nlevels];
    bool reuse_base_comm = (mg->nlevels == 0)
                        && (dims[0] == ctx->dom.dims[0])
                        && (dims[1] == ctx->dom.dims[1]);

    lev->transfer_gathered_from_fine = false;
    if (!dmg_init_level_domain(lev,&ctx->dom,Nx,Ny,dims,reuse_base_comm))
    {
      dmg_hierarchy_free(mg);
      return false;
    }
    if (mg->nlevels > 0)
    {
      pawsim_dmg_level * fine = &mg->level[mg->nlevels-1];
      lev->transfer_gathered_from_fine =
        !dmg_nested_with_previous_level(fine,lev);
    }

    if (lev->active)
    {
      if (!pawsim_field2d_alloc(&lev->x,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
       || !pawsim_field2d_alloc(&lev->rhs,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
       || !pawsim_field2d_alloc(&lev->res,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
       || !pawsim_field2d_alloc(&lev->Hc,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
       || !pawsim_field2d_alloc(&lev->Hw,lev->dom.nx,lev->dom.ny,lev->dom.nghost)
       || !pawsim_field2d_alloc(&lev->Hs,lev->dom.nx,lev->dom.ny,lev->dom.nghost))
      {
        dmg_hierarchy_free(mg);
        return false;
      }
    }
    lev->dx = ctx->cfg.Lx / Nx;
    lev->dy = ctx->cfg.Ly / Ny;
    mg->nlevels ++;

    if ((Nx <= PAWSIM_DMG_MIN_GLOBAL) || (Ny <= PAWSIM_DMG_MIN_GLOBAL)
     || (Nx % 2 != 0) || (Ny % 2 != 0))
    {
      break;
    }

    {
      uint Nx_c = coarsen_size(Nx);
      uint Ny_c = coarsen_size(Ny);
      int coarse_dims[2];
      uint active_cells_x;
      uint active_cells_y;

      dmg_choose_coarse_dims(Nx_c,Ny_c,dims,coarse_dims);
      active_cells_x = Nx_c / (uint) coarse_dims[0];
      active_cells_y = Ny_c / (uint) coarse_dims[1];
      if ((active_cells_x == 0) || (active_cells_y == 0))
      {
        break;
      }
      /*
       * Smaller active communicators are the fragile case for non-power-of-two
       * rank layouts. Stop here and let the bottom level use the gathered
       * coarse tail instead of creating an agglomerated level.
       */
      if ((coarse_dims[0] != dims[0]) || (coarse_dims[1] != dims[1]))
      {
        break;
      }

      Nx = Nx_c;
      Ny = Ny_c;
      dims[0] = coarse_dims[0];
      dims[1] = coarse_dims[1];
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
 * Restrict across a non-nested or agglomerating level boundary.
 *
 * The fine level gathers the residual and operator coefficients to its rank 0.
 * Rank 0 performs the same 2:1 restriction algebra used by the local path,
 * then scatters the coarse fields only to ranks that are active on the coarse
 * communicator.
 */
static bool dmg_restrict_gathered_transition (pawsim_dmg_level * fine,
                                              pawsim_dmg_level * coarse)
{
  real ** fine_res = NULL;
  real ** fine_Hc = NULL;
  real ** fine_Hw = NULL;
  real ** fine_Hs = NULL;
  real ** coarse_rhs = NULL;
  real ** coarse_Hc = NULL;
  real ** coarse_Hw = NULL;
  real ** coarse_Hs = NULL;
  bool ok = true;

  if ((fine->dom.Nx != 2*coarse->dom.Nx)
   || (fine->dom.Ny != 2*coarse->dom.Ny))
  {
    return false;
  }

  if (fine->dom.rank == 0)
  {
    fine_res = matalloc(fine->dom.Nx,fine->dom.Ny);
    fine_Hc = matalloc(fine->dom.Nx,fine->dom.Ny);
    fine_Hw = matalloc(fine->dom.Nx,fine->dom.Ny);
    fine_Hs = matalloc(fine->dom.Nx,fine->dom.Ny);
    coarse_rhs = matalloc(coarse->dom.Nx,coarse->dom.Ny);
    coarse_Hc = matalloc(coarse->dom.Nx,coarse->dom.Ny);
    coarse_Hw = matalloc(coarse->dom.Nx,coarse->dom.Ny);
    coarse_Hs = matalloc(coarse->dom.Nx,coarse->dom.Ny);
    ok = (fine_res != NULL) && (fine_Hc != NULL) && (fine_Hw != NULL)
      && (fine_Hs != NULL) && (coarse_rhs != NULL) && (coarse_Hc != NULL)
      && (coarse_Hw != NULL) && (coarse_Hs != NULL);
  }

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,fine->dom.comm);
    ok = (ok_int != 0);
  }
#endif

  if (ok)
  {
    ok = gather_pressure_field(fine_res,&fine->res,&fine->dom,500)
      && gather_pressure_field(fine_Hc,&fine->Hc,&fine->dom,510)
      && gather_pressure_field(fine_Hw,&fine->Hw,&fine->dom,520)
      && gather_pressure_field(fine_Hs,&fine->Hs,&fine->dom,530);
  }

  if (ok && (fine->dom.rank == 0))
  {
    restrict_global_power2(fine->dom.Nx,fine->dom.Ny,fine_res,coarse_rhs);
    restrict_global_power2(fine->dom.Nx,fine->dom.Ny,fine_Hc,coarse_Hc);
    restrict_global_west_faces_power2(fine->dom.Nx,fine->dom.Ny,fine_Hw,coarse_Hw);
    restrict_global_south_faces_power2(fine->dom.Nx,fine->dom.Ny,fine_Hs,coarse_Hs);
  }

  if (ok && coarse->active)
  {
    ok = scatter_pressure_field(&coarse->rhs,coarse_rhs,&coarse->dom,540)
      && scatter_pressure_field(&coarse->Hc,coarse_Hc,&coarse->dom,550)
      && scatter_pressure_field(&coarse->Hw,coarse_Hw,&coarse->dom,560)
      && scatter_pressure_field(&coarse->Hs,coarse_Hs,&coarse->dom,570);
    pawsim_field2d_zero(&coarse->x);
    pawsim_field2d_exchange_scalar_halo(&coarse->rhs,&coarse->dom);
    pawsim_field2d_exchange_scalar_halo(&coarse->Hc,&coarse->dom);
    pawsim_field2d_exchange_scalar_halo(&coarse->Hw,&coarse->dom);
    pawsim_field2d_exchange_scalar_halo(&coarse->Hs,&coarse->dom);
  }

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,fine->dom.comm);
    ok = (ok_int != 0);
  }
#endif

  if (fine_res != NULL) matfree(fine_res);
  if (fine_Hc != NULL) matfree(fine_Hc);
  if (fine_Hw != NULL) matfree(fine_Hw);
  if (fine_Hs != NULL) matfree(fine_Hs);
  if (coarse_rhs != NULL) matfree(coarse_rhs);
  if (coarse_Hc != NULL) matfree(coarse_Hc);
  if (coarse_Hw != NULL) matfree(coarse_Hw);
  if (coarse_Hs != NULL) matfree(coarse_Hs);

  return ok;
}

/*
 * Prolong across an agglomerating level boundary.
 *
 * Only active coarse ranks gather the solved correction. The resulting global
 * coarse buffer is broadcast on the fine communicator, allowing every active
 * fine rank to interpolate its own correction without joining the coarse
 * communicator.
 */
static real dmg_prolong_gathered_transition (pawsim_dmg_level * fine,
                                             pawsim_dmg_level * coarse,
                                             const pawsim_config * cfg)
{
  real ** global_x = NULL;
  real ** coarse_cols = NULL;
  real * coarse_buf = NULL;
  real max_update = 0;
  bool ok = true;

  coarse_buf = malloc((size_t) coarse->dom.Nx*coarse->dom.Ny*sizeof(real));
  if (coarse_buf == NULL)
  {
    ok = false;
  }

  if (coarse->active && (coarse->dom.rank == 0))
  {
    global_x = matalloc(coarse->dom.Nx,coarse->dom.Ny);
    ok = ok && (global_x != NULL);
  }

#ifdef PAWSIM_USE_MPI
  {
    int ok_int = ok ? 1 : 0;
    MPI_Bcast(&ok_int,1,MPI_INT,0,fine->dom.comm);
    ok = (ok_int != 0);
  }
#endif

  if (ok && coarse->active)
  {
    ok = gather_pressure_field(global_x,&coarse->x,&coarse->dom,580);
  }

  if (ok && (fine->dom.rank == 0))
  {
    pack_global_matrix(coarse_buf,global_x,coarse->dom.Nx,coarse->dom.Ny);
  }

#ifdef PAWSIM_USE_MPI
  if (ok)
  {
    MPI_Bcast(coarse_buf,(int) (coarse->dom.Nx*coarse->dom.Ny),
              MPI_DOUBLE,0,fine->dom.comm);
  }
#endif

  if (ok && make_matrix_columns(&coarse_cols,coarse_buf,coarse->dom.Nx,coarse->dom.Ny))
  {
    max_update = dmg_prolong_add_global_coarse(fine,coarse_cols,
                                               coarse->dom.Nx,coarse->dom.Ny,cfg);
  }
  else
  {
    max_update = cfg->pi_tol + 1;
  }

  if (global_x != NULL) matfree(global_x);
  free(coarse_cols);
  free(coarse_buf);
  return max_update;
}

/*
 * Recursive distributed multigrid V-cycle. Cleanly nested levels use fully
 * local transfers; agglomerating or non-nested level pairs use gathered
 * transfers across that one boundary.
 */
static real dmg_vcycle (pawsim_dmg_hierarchy * mg, uint l, const pawsim_config * cfg)
{
  pawsim_dmg_level * lev = &mg->level[l];
  real max_update;

  if (!lev->active)
  {
    return 0;
  }

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
  if (mg->level[l+1].transfer_gathered_from_fine)
  {
    if (!dmg_restrict_gathered_transition(lev,&mg->level[l+1]))
    {
      return cfg->pi_tol + 1;
    }
  }
  else
  {
    dmg_restrict_field(&mg->level[l+1].rhs,&mg->level[l+1].dom,&lev->res,&lev->dom);
    dmg_restrict_field(&mg->level[l+1].Hc,&mg->level[l+1].dom,&lev->Hc,&lev->dom);
    dmg_restrict_west_faces(&mg->level[l+1].Hw,&mg->level[l+1].dom,&lev->Hw);
    dmg_restrict_south_faces(&mg->level[l+1].Hs,&mg->level[l+1].dom,&lev->Hs);
    pawsim_field2d_zero(&mg->level[l+1].x);
  }
  max_update = fmax(max_update,dmg_vcycle(mg,l+1,cfg));
  if (mg->level[l+1].transfer_gathered_from_fine)
  {
    max_update = fmax(max_update,dmg_prolong_gathered_transition(lev,&mg->level[l+1],cfg));
  }
  else
  {
    max_update = fmax(max_update,dmg_prolong_add(&lev->x,&lev->dom,
                                                 &mg->level[l+1].x,&mg->level[l+1].dom));
  }
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

  /*
   * The distributed multigrid transfer operators need at least one clean local
   * 2:1 coarsening from the model rank layout. Otherwise the solve starts in
   * the fragile gathered-transition path, so use the gathered-MG fallback.
   */
  if (!dmg_can_start_with_local_transfers(&ctx->dom))
  {
    if (ctx->dom.rank == 0)
    {
      fprintf(stderr,
              "WARNING: Distributed MG needs locally nested grid/rank coarsening; using gathered MG for %u x %u grid on rank grid %d x %d\n",
              ctx->dom.Nx,ctx->dom.Ny,ctx->dom.dims[0],ctx->dom.dims[1]);
    }
    ctx->pressure_last_solver = PAWSIM_PRESSURE_SOLVER_MG_GATHERED;
    ctx->pressure_last_fallback = true;
    return solve_pressure_mg_gathered(ctx);
  }

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
