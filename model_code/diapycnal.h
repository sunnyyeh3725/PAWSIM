/**
 * diapycnal.h
 *
 * Diapycnal physics hooks for PAWSIM.
 *
 */
#ifndef _PAWSIM_DIAPYCNAL_H_
#define _PAWSIM_DIAPYCNAL_H_

#include "context.h"

/* Add diapycnal velocity advection and explicit vertical viscosity to momentum. */
void pawsim_diapycnal_apply_momentum (pawsim_context * ctx);

/* Prepare tracer diapycnal-diffusion fluxes for the current derivative call. */
void pawsim_diapycnal_prepare_tracer_fluxes (pawsim_context * ctx);

#endif
