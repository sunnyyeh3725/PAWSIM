/**
 * pressure.h
 *
 * PAWSIM rigid-lid pressure correction interface.
 *
 */
#ifndef _PAWSIM_PRESSURE_H_
#define _PAWSIM_PRESSURE_H_

#include "context.h"

/* Enforce rigid-lid total column thickness before solving for pressure. */
void pawsim_pressure_correct_thickness (pawsim_context * ctx);

/* Solve/apply the rigid-lid pressure correction and optionally record budgets. */
uint pawsim_pressure_correct_rigid_lid (pawsim_context * ctx, bool update_diags);

#endif
