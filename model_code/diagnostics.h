/**
 * diagnostics.h
 *
 * PAWSIM diagnostic and model-state output interface.
 *
 */
#ifndef _PAWSIM_DIAGNOSTICS_H_
#define _PAWSIM_DIAGNOSTICS_H_

#include "context.h"

/* Allocate average accumulators requested by savefreq* input options. */
bool pawsim_diagnostics_alloc (pawsim_context * ctx);

/* Release all diagnostic accumulator storage. */
void pawsim_diagnostics_free (pawsim_context * ctx);

/* Clear online average fields at the start of a new averaging window. */
void pawsim_diagnostics_zero_averages (pawsim_context * ctx);

/* Clear momentum, thickness, energy, and tracer budget accumulators. */
void pawsim_diagnostics_zero_budgets (pawsim_context * ctx);

/* Add the current post-step state to AWSIM-compatible online averages. */
bool pawsim_diagnostics_accumulate_averages (pawsim_context * ctx);

/* Write and reset state-average diagnostics when their save time is reached. */
bool pawsim_diagnostics_write_averages_if_due (pawsim_context * ctx);

/* Write global energy and layer potential-enstrophy diagnostics if due. */
bool pawsim_diagnostics_write_EZ_if_due (pawsim_context * ctx);

/* Write and reset tendency budget diagnostics when their save times are reached. */
bool pawsim_diagnostics_write_budgets_if_due (pawsim_context * ctx, uint n);

/* Gather and write AWSIM-compatible U/V/H/B/W/P model-state files. */
bool pawsim_diagnostics_write_model_state (pawsim_context * ctx, uint n);

/* Write opt-in init_* work-array diagnostics for debugging/regression. */
bool pawsim_diagnostics_write_initialization (pawsim_context * ctx);

/* Write optional init diagnostics plus the mandatory n=0 model state. */
bool pawsim_diagnostics_write_initial_outputs (pawsim_context * ctx);

#endif
