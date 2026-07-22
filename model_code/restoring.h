/**
 * restoring.h
 *
 * Newtonian restoring and interface-relaxation hooks for PAWSIM.
 *
 */
#ifndef _PAWSIM_RESTORING_H_
#define _PAWSIM_RESTORING_H_

#include "context.h"

/* Build prescribed/restoring-induced interface velocity wdia. */
void pawsim_restoring_calc_wdia (pawsim_context * ctx);

/* Add direct u/v Newtonian restoring terms after other momentum physics. */
void pawsim_restoring_apply_momentum (pawsim_context * ctx);

#endif
