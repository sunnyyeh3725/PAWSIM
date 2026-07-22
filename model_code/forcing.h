/**
 * forcing.h
 *
 * PAWSIM optional forcing interface.
 *
 */
#ifndef _PAWSIM_FORCING_H_
#define _PAWSIM_FORCING_H_

#include "context.h"

/* Interpolate time-dependent wind records into work arrays at model time t. */
bool pawsim_forcing_update (pawsim_context * ctx, real t);

/* Add deterministic wind, barotropic forcing, and drag terms to momentum RHS. */
void pawsim_forcing_apply (pawsim_context * ctx);

#endif
