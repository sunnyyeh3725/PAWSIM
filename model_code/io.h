/**
 * io.h
 *
 * Gather-compatible binary output interface for PAWSIM.
 *
 */
#ifndef _PAWSIM_IO_H_
#define _PAWSIM_IO_H_

#include "field.h"

/* Gather an h/u/v-shaped Nx-by-Ny field and write AWSIM binary layout. */
bool pawsim_write_global_field2d (const char * fname,
                                  const pawsim_field2d * field,
                                  const pawsim_domain * dom);

/* Gather q-grid data, including extra wall boundary lines, and write binary. */
bool pawsim_write_global_qfield2d (const char * fname,
                                   const pawsim_field2d * field,
                                   const pawsim_domain * dom);

#endif
