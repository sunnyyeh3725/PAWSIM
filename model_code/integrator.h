/**
 * integrator.h
 *
 * PAWSIM time-integration interface.
 *
 */
#ifndef _PAWSIM_INTEGRATOR_H_
#define _PAWSIM_INTEGRATOR_H_

#include "context.h"

/*
 * Future integrators can use this AWSIM-like derivative callback shape. The
 * current AB implementation calls pawsim_tendency_eval directly.
 */
typedef bool (*pawsim_tendency_fn) (pawsim_context * ctx, real t,
                                    const pawsim_state * state,
                                    pawsim_state * tendency);

bool pawsim_integrator_step (pawsim_context * ctx, uint n);

#endif
