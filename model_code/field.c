/**
 * field.c
 *
 * Local field allocation and halo exchange implementation for PAWSIM.
 *
 * Halo exchange is split conceptually into two operations:
 *
 *  - exchange owned data across MPI rank interfaces;
 *  - fill physical ghost cells at domain walls, or periodic ghost cells in a
 *    serial build.
 *
 * This distinction matters at solid walls. A rank on a physical boundary does
 * not need data from a neighbor outside the domain; it needs AWSIM-style ghost
 * values that encode scalar reflection, wall-normal velocity antisymmetry, and
 * wall-parallel velocity symmetry.
 *
 */
#include "field.h"

#define FIELD(field,i,j) ((field)->a[(i)][(j)])

typedef enum pawsim_halo_kind
{
  PAWSIM_HALO_SCALAR,
  PAWSIM_HALO_Q,
  PAWSIM_HALO_U,
  PAWSIM_HALO_V
}
pawsim_halo_kind;

/*
 * Allocate contiguous storage and a column-pointer view.
 *
 * The a[i][j] view mirrors AWSIM's indexing style, while data remains a single
 * allocation for cheap zeroing and future MPI datatype/IO improvements.
 */
bool pawsim_field2d_alloc (pawsim_field2d * field, uint nx, uint ny, uint nghost)
{
  uint i;
  real ** cols = NULL;

  if (field == NULL)
  {
    return false;
  }

  field->nx = nx;
  field->ny = ny;
  field->nghost = nghost;
  field->sx = nx + 2*nghost;
  field->sy = ny + 2*nghost;
  field->data = calloc((size_t) field->sx*field->sy,sizeof(real));
  cols = malloc(field->sx*sizeof(real *));

  if ((field->data == NULL) || (cols == NULL))
  {
    free(field->data);
    free(cols);
    field->data = NULL;
    field->a = NULL;
    return false;
  }

  for (i = 0; i < field->sx; i ++)
  {
    cols[i] = field->data + i*field->sy;
  }
  field->a = cols;

  return true;
}

/** Release a local field and reset pointers so repeated cleanup is harmless. */
void pawsim_field2d_free (pawsim_field2d * field)
{
  if (field == NULL)
  {
    return;
  }

  free(field->a);
  free(field->data);
  field->a = NULL;
  field->data = NULL;
}

/** Zero the entire local allocation, including ghost cells. */
void pawsim_field2d_zero (pawsim_field2d * field)
{
  if ((field == NULL) || (field->data == NULL))
  {
    return;
  }

  memset(field->data,0,(size_t) field->sx*field->sy*sizeof(real));
}

/** Fill owned cells with their global coordinates for halo-exchange tests. */
void pawsim_field2d_fill_test (pawsim_field2d * field, const pawsim_domain * dom)
{
  uint i,j,g;

  if ((field == NULL) || (dom == NULL))
  {
    return;
  }

  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      FIELD(field,i+g,j+g) = (real) (dom->i0+i) + 0.001*(real) (dom->j0+j);
    }
  }
}

#ifndef PAWSIM_USE_MPI
/** Serial periodic fills replace MPI neighbor exchange in non-MPI builds. */
/** Serial-build periodic copy in y, replacing an MPI north/south exchange. */
static void fill_periodic_y_halos (pawsim_field2d * field)
{
  uint i,j,g;

  g = field->nghost;
  for (i = g; i < g+field->nx; i ++)
  {
    for (j = 0; j < g; j ++)
    {
      FIELD(field,i,j) = FIELD(field,i,field->ny+j);
      FIELD(field,i,g+field->ny+j) = FIELD(field,i,g+j);
    }
  }
}

/** Serial-build periodic copy in x, replacing an MPI east/west exchange. */
static void fill_periodic_x_halos (pawsim_field2d * field)
{
  uint i,j,g;

  g = field->nghost;
  for (j = 0; j < field->sy; j ++)
  {
    for (i = 0; i < g; i ++)
    {
      FIELD(field,i,j) = FIELD(field,field->nx+i,j);
      FIELD(field,g+field->nx+i,j) = FIELD(field,g+i,j);
    }
  }
}
#endif

/*
 * Fill south/north physical ghost cells. Scalars and wall-parallel velocities
 * are reflected symmetrically; wall-normal velocity is antisymmetric with zero
 * imposed on the wall face. q-grid fields are nearest-neighbor extrapolated.
 */
static void fill_wall_y_halos (pawsim_field2d * field, pawsim_halo_kind kind,
                               bool fill_south, bool fill_north)
{
  uint i,j,g,istart,iend;

  g = field->nghost;
  istart = (kind == PAWSIM_HALO_Q) ? 0 : g;
  iend = (kind == PAWSIM_HALO_Q) ? field->sx : g+field->nx;

  /*
   * q-grid fields may include the extra east/north boundary line on wall
   * domains, so fill the full stored x extent when reflecting y boundaries.
   */
  for (i = istart; i < iend; i ++)
  {
    if (kind == PAWSIM_HALO_V)
    {
      if (fill_south) FIELD(field,i,g) = 0;
      if (fill_north) FIELD(field,i,g+field->ny) = 0;
    }

    for (j = 0; j < g; j ++)
    {
      if (kind == PAWSIM_HALO_Q)
      {
        if (fill_south) FIELD(field,i,j) = FIELD(field,i,g);
        if (fill_north) FIELD(field,i,g+field->ny+j) = FIELD(field,i,g+field->ny);
      }
      else if (kind == PAWSIM_HALO_V)
      {
        if (fill_south) FIELD(field,i,j) = -FIELD(field,i,2*g-j);
        if (fill_north) FIELD(field,i,g+field->ny+j) = -FIELD(field,i,g+field->ny-j);
      }
      else
      {
        if (fill_south) FIELD(field,i,j) = FIELD(field,i,2*g-j-1);
        if (fill_north) FIELD(field,i,g+field->ny+j) = FIELD(field,i,g+field->ny-j-1);
      }
    }
  }
}

/*
 * Fill west/east physical ghost cells with AWSIM's staggered-wall convention.
 * U is wall-normal here; V is wall-parallel.
 */
static void fill_wall_x_halos (pawsim_field2d * field, pawsim_halo_kind kind,
                               bool fill_west, bool fill_east)
{
  uint i,j,g;

  g = field->nghost;
  /*
   * U and V live on staggered faces. The wall-normal component is set to zero
   * on the wall face, then reflected antisymmetrically into ghost cells.
   */
  for (j = 0; j < field->sy; j ++)
  {
    if (kind == PAWSIM_HALO_U)
    {
      if (fill_west) FIELD(field,g,j) = 0;
      if (fill_east) FIELD(field,g+field->nx,j) = 0;
    }

    for (i = 0; i < g; i ++)
    {
      if (kind == PAWSIM_HALO_Q)
      {
        if (fill_west) FIELD(field,i,j) = FIELD(field,g,j);
        if (fill_east) FIELD(field,g+field->nx+i,j) = FIELD(field,g+field->nx,j);
      }
      else if (kind == PAWSIM_HALO_U)
      {
        if (fill_west) FIELD(field,i,j) = -FIELD(field,2*g-i,j);
        if (fill_east) FIELD(field,g+field->nx+i,j) = -FIELD(field,g+field->nx-i,j);
      }
      else
      {
        if (fill_west) FIELD(field,i,j) = FIELD(field,2*g-i-1,j);
        if (fill_east) FIELD(field,g+field->nx+i,j) = FIELD(field,g+field->nx-i-1,j);
      }
    }
  }
}

#ifdef PAWSIM_USE_MPI
/** In MPI builds, fill y ghosts only on ranks adjacent to a physical wall. */
static void fill_physical_y_halos (pawsim_field2d * field, const pawsim_domain * dom,
                                   pawsim_halo_kind kind)
{
  if ((dom->nbr_s == MPI_PROC_NULL) || (dom->nbr_n == MPI_PROC_NULL))
  {
    fill_wall_y_halos(field,kind,dom->nbr_s == MPI_PROC_NULL,dom->nbr_n == MPI_PROC_NULL);
  }
}

/** In MPI builds, fill x ghosts only on ranks adjacent to a physical wall. */
static void fill_physical_x_halos (pawsim_field2d * field, const pawsim_domain * dom,
                                   pawsim_halo_kind kind)
{
  if ((dom->nbr_w == MPI_PROC_NULL) || (dom->nbr_e == MPI_PROC_NULL))
  {
    fill_wall_x_halos(field,kind,dom->nbr_w == MPI_PROC_NULL,dom->nbr_e == MPI_PROC_NULL);
  }
}

/** Pack g full-height columns, including corner ghost values. */
static void pack_x_full (pawsim_field2d * field, real * buf, uint istart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < g; i ++)
  {
    for (j = 0; j < field->sy; j ++)
    {
      buf[p++] = FIELD(field,istart+i,j);
    }
  }
}

/** Unpack g full-height columns, including corner ghost values. */
static void unpack_x_full (pawsim_field2d * field, real * buf, uint istart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < g; i ++)
  {
    for (j = 0; j < field->sy; j ++)
    {
      FIELD(field,istart+i,j) = buf[p++];
    }
  }
}

/** Pack g rows across owned x cells only. */
static void pack_y (pawsim_field2d * field, real * buf, uint jstart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < g; j ++)
    {
      buf[p++] = FIELD(field,g+i,jstart+j);
    }
  }
}

/** Unpack g rows across owned x cells only. */
static void unpack_y (pawsim_field2d * field, real * buf, uint jstart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < g; j ++)
    {
      FIELD(field,g+i,jstart+j) = buf[p++];
    }
  }
}

/** Pack g rows across the full local x extent, including corner ghost columns. */
static void pack_y_full (pawsim_field2d * field, real * buf, uint jstart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < field->sx; i ++)
  {
    for (j = 0; j < g; j ++)
    {
      buf[p++] = FIELD(field,i,jstart+j);
    }
  }
}

/** Unpack g rows across the full local x extent, including corner ghost columns. */
static void unpack_y_full (pawsim_field2d * field, real * buf, uint jstart)
{
  uint i,j,g,p = 0;
  g = field->nghost;
  for (i = 0; i < field->sx; i ++)
  {
    for (j = 0; j < g; j ++)
    {
      FIELD(field,i,jstart+j) = buf[p++];
    }
  }
}
#endif

/*
 * Exchange only between real MPI neighbors. Physical walls are left untouched;
 * use this for derived fields whose wall values were already constructed.
 */
void pawsim_field2d_exchange_internal_halo (pawsim_field2d * field,
                                            const pawsim_domain * dom)
{
#ifdef PAWSIM_USE_MPI
  uint g;
  size_t xcount,ycount;
  real * send_w = NULL, * send_e = NULL, * recv_w = NULL, * recv_e = NULL;
  real * send_s = NULL, * send_n = NULL, * recv_s = NULL, * recv_n = NULL;

  if ((field == NULL) || (dom == NULL))
  {
    return;
  }

  g = field->nghost;
  xcount = (size_t) g*field->sy;
  ycount = (size_t) field->sx*g;

  send_w = malloc(xcount*sizeof(real));
  send_e = malloc(xcount*sizeof(real));
  recv_w = malloc(xcount*sizeof(real));
  recv_e = malloc(xcount*sizeof(real));
  send_s = malloc(ycount*sizeof(real));
  send_n = malloc(ycount*sizeof(real));
  recv_s = malloc(ycount*sizeof(real));
  recv_n = malloc(ycount*sizeof(real));

  if ((send_w == NULL) || (send_e == NULL) || (recv_w == NULL) || (recv_e == NULL)
   || (send_s == NULL) || (send_n == NULL) || (recv_s == NULL) || (recv_n == NULL))
  {
    goto cleanup;
  }

  /*
   * Internal exchange copies full halo slabs, including corners. This is used
   * for derived q-grid work arrays after physical boundary values have already
   * been constructed locally from ghosted u/v/h.
   */
  pack_y_full(field,send_s,g);
  pack_y_full(field,send_n,field->ny);

  MPI_Sendrecv(send_s,(int) ycount,MPI_DOUBLE,dom->nbr_s,12,
               recv_n,(int) ycount,MPI_DOUBLE,dom->nbr_n,12,dom->comm,MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_n,(int) ycount,MPI_DOUBLE,dom->nbr_n,13,
               recv_s,(int) ycount,MPI_DOUBLE,dom->nbr_s,13,dom->comm,MPI_STATUS_IGNORE);

  if (dom->nbr_s != MPI_PROC_NULL) unpack_y_full(field,recv_s,0);
  if (dom->nbr_n != MPI_PROC_NULL) unpack_y_full(field,recv_n,field->ny+g);

  pack_x_full(field,send_w,g);
  pack_x_full(field,send_e,field->nx);

  MPI_Sendrecv(send_w,(int) xcount,MPI_DOUBLE,dom->nbr_w,10,
               recv_e,(int) xcount,MPI_DOUBLE,dom->nbr_e,10,dom->comm,MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_e,(int) xcount,MPI_DOUBLE,dom->nbr_e,11,
               recv_w,(int) xcount,MPI_DOUBLE,dom->nbr_w,11,dom->comm,MPI_STATUS_IGNORE);

  if (dom->nbr_w != MPI_PROC_NULL) unpack_x_full(field,recv_w,0);
  if (dom->nbr_e != MPI_PROC_NULL) unpack_x_full(field,recv_e,field->nx+g);

cleanup:
  free(send_w); free(send_e); free(recv_w); free(recv_e);
  free(send_s); free(send_n); free(recv_s); free(recv_n);
#else
  if (dom->periodic_y)
  {
    fill_periodic_y_halos(field);
  }

  if (dom->periodic_x)
  {
    fill_periodic_x_halos(field);
  }
#endif
}

/** Fill physical wall ghosts without communicating across internal rank edges. */
static void fill_physical_boundary (pawsim_field2d * field, const pawsim_domain * dom,
                                    pawsim_halo_kind kind)
{
  if ((field == NULL) || (dom == NULL))
  {
    return;
  }

#ifdef PAWSIM_USE_MPI
  fill_physical_y_halos(field,dom,kind);
  fill_physical_x_halos(field,dom,kind);
#else
  if (!dom->periodic_y)
  {
    fill_wall_y_halos(field,kind,true,true);
  }
  if (!dom->periodic_x)
  {
    fill_wall_x_halos(field,kind,true,true);
  }
#endif
}

/*
 * Full halo update for fields stored on the decomposed domain: exchange across
 * MPI rank interfaces, then fill any physical wall ghosts with the right
 * staggered-grid symmetry.
 */
static void exchange_halo_with_physical_boundary (pawsim_field2d * field,
                                                  const pawsim_domain * dom,
                                                  pawsim_halo_kind kind)
{
#ifdef PAWSIM_USE_MPI
  uint g;
  size_t xcount,ycount;
  real * send_w = NULL, * send_e = NULL, * recv_w = NULL, * recv_e = NULL;
  real * send_s = NULL, * send_n = NULL, * recv_s = NULL, * recv_n = NULL;

  if ((field == NULL) || (dom == NULL))
  {
    return;
  }

  g = field->nghost;
  xcount = (size_t) g*field->sy;
  ycount = (size_t) field->nx*g;

  send_w = malloc(xcount*sizeof(real));
  send_e = malloc(xcount*sizeof(real));
  recv_w = malloc(xcount*sizeof(real));
  recv_e = malloc(xcount*sizeof(real));
  send_s = malloc(ycount*sizeof(real));
  send_n = malloc(ycount*sizeof(real));
  recv_s = malloc(ycount*sizeof(real));
  recv_n = malloc(ycount*sizeof(real));

  if ((send_w == NULL) || (send_e == NULL) || (recv_w == NULL) || (recv_e == NULL)
   || (send_s == NULL) || (send_n == NULL) || (recv_s == NULL) || (recv_n == NULL))
  {
    goto cleanup;
  }

  /*
   * Exchange y strips first, fill y physical boundaries, then exchange x strips
   * including the y-corner data. This gives ranks adjacent in x valid corner
   * values near north/south walls.
   */
  pack_y(field,send_s,g);
  pack_y(field,send_n,field->ny);

  MPI_Sendrecv(send_s,(int) ycount,MPI_DOUBLE,dom->nbr_s,12,
               recv_n,(int) ycount,MPI_DOUBLE,dom->nbr_n,12,dom->comm,MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_n,(int) ycount,MPI_DOUBLE,dom->nbr_n,13,
               recv_s,(int) ycount,MPI_DOUBLE,dom->nbr_s,13,dom->comm,MPI_STATUS_IGNORE);

  if (dom->nbr_s != MPI_PROC_NULL) unpack_y(field,recv_s,0);
  if (dom->nbr_n != MPI_PROC_NULL) unpack_y(field,recv_n,field->ny+g);

  /** Fill y-wall ghosts before the x exchange so ranks across an x seam receive
     valid corner values adjacent to north/south physical boundaries. */
  fill_physical_y_halos(field,dom,kind);

  pack_x_full(field,send_w,g);
  pack_x_full(field,send_e,field->nx);

  MPI_Sendrecv(send_w,(int) xcount,MPI_DOUBLE,dom->nbr_w,10,
               recv_e,(int) xcount,MPI_DOUBLE,dom->nbr_e,10,dom->comm,MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_e,(int) xcount,MPI_DOUBLE,dom->nbr_e,11,
               recv_w,(int) xcount,MPI_DOUBLE,dom->nbr_w,11,dom->comm,MPI_STATUS_IGNORE);

  if (dom->nbr_w != MPI_PROC_NULL) unpack_x_full(field,recv_w,0);
  if (dom->nbr_e != MPI_PROC_NULL) unpack_x_full(field,recv_e,field->nx+g);

  fill_physical_x_halos(field,dom,kind);

cleanup:
  free(send_w); free(send_e); free(recv_w); free(recv_e);
  free(send_s); free(send_n); free(recv_s); free(recv_n);
#else
  pawsim_field2d_exchange_internal_halo(field,dom);
  fill_physical_boundary(field,dom,kind);
#endif
}

/** Fill scalar physical ghosts only; rank-interface halos are not exchanged. */
void pawsim_field2d_fill_scalar_boundary (pawsim_field2d * field, const pawsim_domain * dom)
{
  fill_physical_boundary(field,dom,PAWSIM_HALO_SCALAR);
}

/** Fill u-face physical ghosts only; rank-interface halos are not exchanged. */
void pawsim_field2d_fill_u_boundary (pawsim_field2d * field, const pawsim_domain * dom)
{
  fill_physical_boundary(field,dom,PAWSIM_HALO_U);
}

/** Fill v-face physical ghosts only; rank-interface halos are not exchanged. */
void pawsim_field2d_fill_v_boundary (pawsim_field2d * field, const pawsim_domain * dom)
{
  fill_physical_boundary(field,dom,PAWSIM_HALO_V);
}

/** Exchange a scalar/cell-centered field and apply scalar wall reflection. */
void pawsim_field2d_exchange_scalar_halo (pawsim_field2d * field, const pawsim_domain * dom)
{
  exchange_halo_with_physical_boundary(field,dom,PAWSIM_HALO_SCALAR);
}

/** Exchange a q-grid field and extrapolate physical q-boundary values. */
void pawsim_field2d_exchange_q_halo (pawsim_field2d * field, const pawsim_domain * dom)
{
  exchange_halo_with_physical_boundary(field,dom,PAWSIM_HALO_Q);
}

/** Exchange a u-face field and enforce u's wall-normal boundary condition. */
void pawsim_field2d_exchange_u_halo (pawsim_field2d * field, const pawsim_domain * dom)
{
  exchange_halo_with_physical_boundary(field,dom,PAWSIM_HALO_U);
}

/** Exchange a v-face field and enforce v's wall-normal boundary condition. */
void pawsim_field2d_exchange_v_halo (pawsim_field2d * field, const pawsim_domain * dom)
{
  exchange_halo_with_physical_boundary(field,dom,PAWSIM_HALO_V);
}

/** Backward-compatible alias for scalar halo exchange. */
void pawsim_field2d_exchange_halo (pawsim_field2d * field, const pawsim_domain * dom)
{
  pawsim_field2d_exchange_scalar_halo(field,dom);
}
