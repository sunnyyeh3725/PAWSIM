/**
 * domain.c
 *
 * MPI domain decomposition implementation for PAWSIM.
 *
 * PAWSIM uses a 2-D Cartesian decomposition. Local arrays store only the owned
 * cell-centered rectangle plus halos; global coordinates are recovered through
 * i0/j0. The block partition formula is the same for all ranks and permits
 * non-even splits, which is important for arbitrary grid sizes such as 250.
 *
 */
#include "domain.h"

/** First global index owned by a rank coordinate in a block decomposition. */
static uint block_start (uint N, uint coord, uint dim)
{
  return (N*coord) / dim;
}

/** Number of cells owned by a rank coordinate, allowing non-even splits. */
static uint block_size (uint N, uint coord, uint dim)
{
  return block_start(N,coord+1,dim) - block_start(N,coord,dim);
}

/** Thin wrapper so serial and MPI builds share the same main-program shape. */
void pawsim_mpi_init (int * argc, char *** argv)
{
  MPI_Init(argc,argv);
}

/** Symmetric wrapper around MPI_Finalize or the serial compatibility no-op. */
void pawsim_mpi_finalize (void)
{
  MPI_Finalize();
}

/*
 * Build the Cartesian rank topology and each rank's owned cell rectangle.
 * Wall flags are represented indirectly through non-periodic MPI dimensions.
 */
bool pawsim_domain_init (pawsim_domain * dom, uint Nx, uint Ny, uint nghost,
                         bool periodic_x, bool periodic_y)
{
  if (dom == NULL)
  {
    return false;
  }

  dom->comm = MPI_COMM_WORLD;
  MPI_Comm_rank(dom->comm,&dom->rank);
  MPI_Comm_size(dom->comm,&dom->size);

  dom->Nx = Nx;
  dom->Ny = Ny;
  dom->nghost = nghost;
  dom->periodic_x = periodic_x;
  dom->periodic_y = periodic_y;

#ifdef PAWSIM_USE_MPI
  {
    int periods[2];
    MPI_Comm cart_comm;

    dom->dims[0] = 0;
    dom->dims[1] = 0;
    /*
     * Let MPI choose a near-square process grid. This is simple and usually
     * good for 2-D stencil work; performance-specific layouts can be added
     * later if needed.
     */
    MPI_Dims_create(dom->size,2,dom->dims);

    periods[0] = periodic_x ? 1 : 0;
    periods[1] = periodic_y ? 1 : 0;
    MPI_Cart_create(MPI_COMM_WORLD,2,dom->dims,periods,0,&cart_comm);
    dom->comm = cart_comm;
    MPI_Comm_rank(dom->comm,&dom->rank);
    MPI_Cart_coords(dom->comm,dom->rank,2,dom->coords);
    MPI_Cart_shift(dom->comm,0,1,&dom->nbr_w,&dom->nbr_e);
    MPI_Cart_shift(dom->comm,1,1,&dom->nbr_s,&dom->nbr_n);
  }
#else
  dom->dims[0] = 1;
  dom->dims[1] = 1;
  dom->coords[0] = 0;
  dom->coords[1] = 0;
  dom->nbr_w = periodic_x ? 0 : MPI_PROC_NULL;
  dom->nbr_e = periodic_x ? 0 : MPI_PROC_NULL;
  dom->nbr_s = periodic_y ? 0 : MPI_PROC_NULL;
  dom->nbr_n = periodic_y ? 0 : MPI_PROC_NULL;
#endif

  /** Owned-cell rectangle in global cell-centered coordinates. */
  dom->i0 = block_start(Nx,(uint) dom->coords[0],(uint) dom->dims[0]);
  dom->j0 = block_start(Ny,(uint) dom->coords[1],(uint) dom->dims[1]);
  dom->nx = block_size(Nx,(uint) dom->coords[0],(uint) dom->dims[0]);
  dom->ny = block_size(Ny,(uint) dom->coords[1],(uint) dom->dims[1]);

  return (dom->nx > 0) && (dom->ny > 0);
}

/** Print the local tile, useful when checking MPI decompositions by eye. */
void pawsim_domain_print (const pawsim_domain * dom)
{
  if (dom == NULL)
  {
    return;
  }

  printf("PAWSIM rank %d/%d: coords=(%d,%d), local=%u x %u, start=(%u,%u)\n",
         dom->rank,dom->size,dom->coords[0],dom->coords[1],
         dom->nx,dom->ny,dom->i0,dom->j0);
  fflush(stdout);
}
