/**
 * field.h
 *
 * Local field allocation and halo exchange interface for PAWSIM.
 *
 */
#ifndef _PAWSIM_FIELD_H_
#define _PAWSIM_FIELD_H_

#include "domain.h"

typedef struct pawsim_field2d
{
  /* nx/ny are owned cells; sx/sy include halos. */
  uint nx;
  uint ny;
  uint nghost;
  uint sx;
  uint sy;
  /* data is contiguous storage; a is an AWSIM-style a[i][j] column view. */
  real * data;
  real ** a;
}
pawsim_field2d;

/* Allocate an nx-by-ny owned field plus nghost cells on every side. */
bool pawsim_field2d_alloc (pawsim_field2d * field, uint nx, uint ny, uint nghost);

/* Release contiguous storage and the AWSIM-style column pointer view. */
void pawsim_field2d_free (pawsim_field2d * field);

/* Set the full local allocation, including halos, to zero. */
void pawsim_field2d_zero (pawsim_field2d * field);

/* Fill owned cells with a deterministic global-index pattern for halo tests. */
void pawsim_field2d_fill_test (pawsim_field2d * field, const pawsim_domain * dom);

/* Exchange rank-neighbor halos only; physical wall ghosts are left untouched. */
void pawsim_field2d_exchange_internal_halo (pawsim_field2d * field, const pawsim_domain * dom);

/* Fill physical-wall or serial-periodic scalar ghosts without MPI exchange. */
void pawsim_field2d_fill_scalar_boundary (pawsim_field2d * field, const pawsim_domain * dom);

/* Fill physical-wall or serial-periodic u-face ghosts. */
void pawsim_field2d_fill_u_boundary (pawsim_field2d * field, const pawsim_domain * dom);

/* Fill physical-wall or serial-periodic v-face ghosts. */
void pawsim_field2d_fill_v_boundary (pawsim_field2d * field, const pawsim_domain * dom);

/* Backward-compatible scalar halo update. */
void pawsim_field2d_exchange_halo (pawsim_field2d * field, const pawsim_domain * dom);

/* Exchange rank-neighbor halos and fill scalar physical boundaries. */
void pawsim_field2d_exchange_scalar_halo (pawsim_field2d * field, const pawsim_domain * dom);

/* Exchange/fill q-grid fields, including wall-domain extra boundary lines. */
void pawsim_field2d_exchange_q_halo (pawsim_field2d * field, const pawsim_domain * dom);

/* Exchange/fill u-face fields, with no-normal-flow reflection at E/W walls. */
void pawsim_field2d_exchange_u_halo (pawsim_field2d * field, const pawsim_domain * dom);

/* Exchange/fill v-face fields, with no-normal-flow reflection at N/S walls. */
void pawsim_field2d_exchange_v_halo (pawsim_field2d * field, const pawsim_domain * dom);

#endif
