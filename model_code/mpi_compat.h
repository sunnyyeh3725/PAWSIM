/**
 * mpi_compat.h
 *
 * Minimal MPI compatibility layer for building PAWSIM before MPI is available.
 *
 * This is intentionally tiny: it supports single-rank development builds, not
 * a real MPI emulation layer. Parallel behavior must still be tested with
 * PAWSIM_USE_MPI and mpirun.
 *
 */
#ifndef _PAWSIM_MPI_COMPAT_H_
#define _PAWSIM_MPI_COMPAT_H_

#ifdef PAWSIM_USE_MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
typedef int MPI_Request;
typedef int MPI_Status;

#define MPI_COMM_WORLD 0
#define MPI_PROC_NULL -1
#define MPI_REQUEST_NULL 0
#define MPI_DOUBLE 0
#define MPI_INT 0
#define MPI_UNSIGNED 0
#define MPI_MAX 0
#define MPI_SUM 0
#define MPI_STATUS_IGNORE ((MPI_Status *) 0)

static inline int MPI_Init (int * argc, char *** argv)
{
  (void) argc;
  (void) argv;
  return 0;
}

static inline int MPI_Finalize (void)
{
  return 0;
}

static inline int MPI_Comm_rank (MPI_Comm comm, int * rank)
{
  (void) comm;
  *rank = 0;
  return 0;
}

static inline int MPI_Comm_size (MPI_Comm comm, int * size)
{
  (void) comm;
  *size = 1;
  return 0;
}

static inline int MPI_Allreduce (const void * sendbuf, void * recvbuf, int count,
                                 int datatype, int op, MPI_Comm comm)
{
  /* Single-rank compatibility: the reduction result is just the send buffer. */
  (void) datatype;
  (void) op;
  (void) comm;
  memcpy(recvbuf,sendbuf,(size_t) count*sizeof(double));
  return 0;
}

static inline int MPI_Bcast (void * buffer, int count, int datatype, int root,
                             MPI_Comm comm)
{
  (void) buffer;
  (void) count;
  (void) datatype;
  (void) root;
  (void) comm;
  return 0;
}
#endif

#endif
