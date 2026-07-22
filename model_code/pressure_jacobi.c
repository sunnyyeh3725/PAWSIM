/**
 * pressure_jacobi.c
 *
 * Distributed Jacobi pressure solver prototype for PAWSIM.
 *
 * This file is retained as a simple early pressure-solver experiment. The
 * production rigid-lid paths now live in pressure.c; Jacobi is useful mainly as
 * a readable stencil/checking reference.
 *
 */
#include "pressure_jacobi.h"

#include <math.h>

uint pawsim_pressure_jacobi (pawsim_field2d * pi,
                             pawsim_field2d * rhs,
                             const pawsim_domain * dom,
                             real dx,
                             real dy,
                             real tol,
                             uint maxiters)
{
  pawsim_field2d next;
  uint i,j,g,iters;
  real idx2,idy2,denom;
  real diff,local_max,global_max;

  if ((pi == NULL) || (rhs == NULL) || (dom == NULL))
  {
    return 0;
  }

  if (!pawsim_field2d_alloc(&next,pi->nx,pi->ny,pi->nghost))
  {
    return 0;
  }

  g = pi->nghost;
  idx2 = 1 / SQUARE(dx);
  idy2 = 1 / SQUARE(dy);
  denom = 2*(idx2+idy2);
  global_max = tol + 1;
  iters = 0;

  /* Textbook Jacobi: exchange old pi, compute next, then swap by copying. */
  while ((global_max > tol) && (iters < maxiters))
  {
    pawsim_field2d_exchange_halo(pi,dom);
    local_max = 0;

    for (i = 0; i < pi->nx; i ++)
    {
      for (j = 0; j < pi->ny; j ++)
      {
        next.a[i+g][j+g] = (idx2*(pi->a[i+g-1][j+g] + pi->a[i+g+1][j+g])
                         + idy2*(pi->a[i+g][j+g-1] + pi->a[i+g][j+g+1])
                         - rhs->a[i+g][j+g]) / denom;
        diff = fabs(next.a[i+g][j+g]-pi->a[i+g][j+g]);
        local_max = fmax(local_max,diff);
      }
    }

    for (i = 0; i < pi->nx; i ++)
    {
      for (j = 0; j < pi->ny; j ++)
      {
        pi->a[i+g][j+g] = next.a[i+g][j+g];
      }
    }

    MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,dom->comm);
    iters ++;
  }

  pawsim_field2d_free(&next);
  return iters;
}
