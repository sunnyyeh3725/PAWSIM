/**
 * tendency.h
 *
 * PAWSIM tderiv-equivalent tendency interface.
 *
 */
#ifndef _PAWSIM_TENDENCY_H_
#define _PAWSIM_TENDENCY_H_

#include "context.h"

/* Rebuild work arrays from ctx->state without returning a tendency state. */
void pawsim_tendency_prepare_work_arrays (pawsim_context * ctx);

/* Apply AWSIM tderiv physics in the same order, accumulating ctx->work.dt_*. */
void pawsim_tendency_compute_deterministic_core (pawsim_context * ctx,
                                                 const pawsim_state * state);

/* Derivative callback used by the Adams-Bashforth integrator. */
bool pawsim_tendency_eval (pawsim_context * ctx, real t,
                           const pawsim_state * state,
                           pawsim_state * tendency);

#endif
