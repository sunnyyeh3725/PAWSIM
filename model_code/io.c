/**
 * io.c
 *
 * Gather-compatible binary output implementation for PAWSIM.
 *
 * Output is deliberately AWSIM/Matlab-compatible: rank 0 writes raw double
 * matrices in global i-fast/j-slow order using the same helper routines as the
 * original codebase. The current gather path favors clarity over IO scaling.
 *
 */
#include "io.h"

/*
 * Gather a decomposed cell-centered field and write it in AWSIM's raw binary
 * matrix layout so existing Matlab analysis can read PAWSIM output unchanged.
 */
bool pawsim_write_global_field2d (const char * fname,
                                  const pawsim_field2d * field,
                                  const pawsim_domain * dom)
{
  uint i,j,g;
  FILE * outfile = NULL;
  real ** global = NULL;

  if ((fname == NULL) || (field == NULL) || (dom == NULL))
  {
    return false;
  }

  g = field->nghost;

#ifdef PAWSIM_USE_MPI
  /** Rank 0 assembles the full cell-centered field before writing. */
  if (dom->rank == 0)
  {
    global = matalloc(dom->Nx,dom->Ny);
    if (global == NULL)
    {
      return false;
    }
  }

  if (dom->rank == 0)
  {
    int src;
    uint meta[4];
    real * buf = NULL;

    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        global[dom->i0+i][dom->j0+j] = field->a[i+g][j+g];
      }
    }

    for (src = 1; src < dom->size; src ++)
    {
      MPI_Recv(meta,4,MPI_UNSIGNED,src,100,dom->comm,MPI_STATUS_IGNORE);
      buf = malloc((size_t) meta[2]*meta[3]*sizeof(real));
      if (buf == NULL)
      {
        return false;
      }
      MPI_Recv(buf,(int) (meta[2]*meta[3]),MPI_DOUBLE,src,101,dom->comm,MPI_STATUS_IGNORE);
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
    for (i = 0; i < field->nx; i ++)
    {
      for (j = 0; j < field->ny; j ++)
      {
        buf[i*field->ny+j] = field->a[i+g][j+g];
      }
    }
    MPI_Send(meta,4,MPI_UNSIGNED,0,100,dom->comm);
    MPI_Send(buf,(int) (field->nx*field->ny),MPI_DOUBLE,0,101,dom->comm);
    free(buf);
  }
#else
  global = matalloc(dom->Nx,dom->Ny);
  if (global == NULL)
  {
    return false;
  }

  for (i = 0; i < field->nx; i ++)
  {
    for (j = 0; j < field->ny; j ++)
    {
      global[dom->i0+i][dom->j0+j] = field->a[i+g][j+g];
    }
  }
#endif

  if (dom->rank == 0)
  {
    outfile = fopen(fname,"w");
    if (outfile == NULL)
    {
      matfree(global);
      return false;
    }
    printMatrix(outfile,global,dom->Nx,dom->Ny);
    fclose(outfile);
    matfree(global);
  }

  return true;
}

/** Number of q-grid x indices this rank contributes to a global q output. */
static uint q_output_nx (const pawsim_field2d * field, const pawsim_domain * dom)
{
  /** Non-periodic q grids carry the extra east boundary line on the east rank. */
  return field->nx + ((dom->i0+dom->nx == dom->Nx) ? 1 : 0);
}

/** Number of q-grid y indices this rank contributes to a global q output. */
static uint q_output_ny (const pawsim_field2d * field, const pawsim_domain * dom)
{
  /** Non-periodic q grids carry the extra north boundary line on the north rank. */
  return field->ny + ((dom->j0+dom->ny == dom->Ny) ? 1 : 0);
}

/*
 * Write a q-grid field. The current PAWSIM q diagnostics use AWSIM's
 * wall-domain convention of an (Nx+1) x (Ny+1) global array.
 */
bool pawsim_write_global_qfield2d (const char * fname,
                                   const pawsim_field2d * field,
                                   const pawsim_domain * dom)
{
  uint i,j,g,qnx,qny;
  FILE * outfile = NULL;
  real ** global = NULL;

  if ((fname == NULL) || (field == NULL) || (dom == NULL))
  {
    return false;
  }

  g = field->nghost;
  qnx = q_output_nx(field,dom);
  qny = q_output_ny(field,dom);

#ifdef PAWSIM_USE_MPI
  if (dom->rank == 0)
  {
    global = matalloc(dom->Nx+1,dom->Ny+1);
    if (global == NULL)
    {
      return false;
    }
  }

  if (dom->rank == 0)
  {
    int src;
    uint meta[4];
    real * buf = NULL;

    for (i = 0; i < qnx; i ++)
    {
      for (j = 0; j < qny; j ++)
      {
        global[dom->i0+i][dom->j0+j] = field->a[i+g][j+g];
      }
    }

    for (src = 1; src < dom->size; src ++)
    {
      MPI_Recv(meta,4,MPI_UNSIGNED,src,110,dom->comm,MPI_STATUS_IGNORE);
      buf = malloc((size_t) meta[2]*meta[3]*sizeof(real));
      if (buf == NULL)
      {
        matfree(global);
        return false;
      }
      MPI_Recv(buf,(int) (meta[2]*meta[3]),MPI_DOUBLE,src,111,dom->comm,MPI_STATUS_IGNORE);
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
    uint meta[4];
    real * buf = malloc((size_t) qnx*qny*sizeof(real));
    if (buf == NULL)
    {
      return false;
    }
    meta[0] = dom->i0;
    meta[1] = dom->j0;
    meta[2] = qnx;
    meta[3] = qny;
    for (i = 0; i < qnx; i ++)
    {
      for (j = 0; j < qny; j ++)
      {
        buf[i*qny+j] = field->a[i+g][j+g];
      }
    }
    MPI_Send(meta,4,MPI_UNSIGNED,0,110,dom->comm);
    MPI_Send(buf,(int) (qnx*qny),MPI_DOUBLE,0,111,dom->comm);
    free(buf);
  }
#else
  global = matalloc(dom->Nx+1,dom->Ny+1);
  if (global == NULL)
  {
    return false;
  }

  for (i = 0; i < qnx; i ++)
  {
    for (j = 0; j < qny; j ++)
    {
      global[dom->i0+i][dom->j0+j] = field->a[i+g][j+g];
    }
  }
#endif

  if (dom->rank == 0)
  {
    outfile = fopen(fname,"w");
    if (outfile == NULL)
    {
      matfree(global);
      return false;
    }
    printMatrix(outfile,global,dom->Nx+1,dom->Ny+1);
    fclose(outfile);
    matfree(global);
  }

  return true;
}
