/**
 * domain.h
 *
 * MPI domain decomposition interface for PAWSIM.
 *
 */
#ifndef _PAWSIM_DOMAIN_H_
#define _PAWSIM_DOMAIN_H_

#include "defs.h"
#include "mpi_compat.h"

typedef struct pawsim_domain
{
  /* Cartesian communicator and rank topology. */
  MPI_Comm comm;
  int rank;
  int size;
  int dims[2];
  int coords[2];
  int nbr_w;
  int nbr_e;
  int nbr_s;
  int nbr_n;
  /* Global grid, local owned extent, and local origin in global coordinates. */
  uint Nx;
  uint Ny;
  uint nx;
  uint ny;
  uint i0;
  uint j0;
  uint nghost;
  /* True means MPI wraps in that direction; false means physical walls. */
  bool periodic_x;
  bool periodic_y;
}
pawsim_domain;

/* Initialize MPI, or the serial compatibility shim in non-MPI builds. */
void pawsim_mpi_init (int * argc, char *** argv);

/* Finalize MPI/shim state. */
void pawsim_mpi_finalize (void);

/* Build the Cartesian rank layout and each rank's owned global tile. */
bool pawsim_domain_init (pawsim_domain * dom, uint Nx, uint Ny, uint nghost,
                         bool periodic_x, bool periodic_y,
                         int requested_nx, int requested_ny);

/* Print a concise rank/local-domain summary for run logs. */
void pawsim_domain_print (const pawsim_domain * dom);

#endif
