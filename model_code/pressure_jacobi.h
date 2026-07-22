/**
 * pressure_jacobi.h
 *
 * Distributed Jacobi pressure solver prototype for PAWSIM.
 *
 * This is not wired into the main pressureSolver selector; see pressure.c for
 * the active SOR/gathered-MG/distributed-MG implementations.
 *
 */
#ifndef _PAWSIM_PRESSURE_JACOBI_H_
#define _PAWSIM_PRESSURE_JACOBI_H_

#include "field.h"

uint pawsim_pressure_jacobi (pawsim_field2d * pi,
                             pawsim_field2d * rhs,
                             const pawsim_domain * dom,
                             real dx,
                             real dy,
                             real tol,
                             uint maxiters);

#endif
