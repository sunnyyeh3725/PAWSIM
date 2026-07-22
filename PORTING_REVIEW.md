# PAWSIM Porting Review

This note records the current AWSIM-to-PAWSIM porting status. PAWSIM targets
scientific equivalence with AWSIM, while allowing cleaned-up implementation
details and boundary diagnostics where those do not change the intended model
state evolution.

## Ported Core

- AWSIM key/value input parsing, including the "exactly two of Nt, tmax, dt"
  time-parameter convention.
- Global AWSIM binary input files read on rank 0 and scattered to MPI-local
  arrays with halos.
- C-grid prognostic state for arbitrary `Nlay`: `u`, `v`, `h`, and optional
  active/passive tracer `b`.
- AWSIM ghost-cell boundary conditions for scalar, u-face, v-face, and q-grid
  quantities, separated from MPI rank exchange.
- Interface-height calculation, face thicknesses, q-grid vorticity/PV, and
  AL81/HK83/TW81/S75 momentum coefficients.
- Bernoulli/Montgomery pressure work arrays, including active-buoyancy terms.
- Momentum tendencies:
  - pressure/Montgomery gradient;
  - kinetic-energy gradient;
  - vector-invariant PV advection;
  - Laplacian and biharmonic viscosity;
  - wind stress;
  - static barotropic forcing;
  - linear/quadratic bottom drag;
  - linear/quadratic surface drag, lid velocity, and wind feedback;
  - lateral buoyancy gradient;
  - diapycnal velocity advection and explicit diapycnal viscosity;
  - Newtonian velocity restoring.
- Thickness tendencies:
  - mass-flux divergence;
  - thickness/interface restoring through `wdia`.
- Tracer tendencies:
  - AL81, UP3, and KT00 horizontal advection;
  - vertical/diapycnal advection;
  - explicit diapycnal diffusion and convective-enhanced diffusion;
  - Laplacian and biharmonic horizontal diffusion;
  - tracer restoring.
- AB1, AB2, and AB3 time stepping using the same derivative-callback pattern as
  AWSIM.
- Rigid-lid pressure correction with:
  - distributed SOR;
  - gathered AWSIM multigrid;
  - distributed multigrid with gathered coarse tail for awkward coarsening.
- AWSIM-compatible model state, averages, energy/enstrophy, and non-random
  budget diagnostics.

## Intentional Differences

- PAWSIM writes `init_*` work-array diagnostics only when
  `writeInitDiagnostics 1` is set. AWSIM has no equivalent output set.
- PAWSIM suppresses momentum budget diagnostics on inactive wall-normal velocity
  faces. AWSIM can record terms there before zeroing the actual wall-normal
  tendency. This is diagnostic bookkeeping only.
- The distributed pressure solver uses a different floating-point reduction and
  communication order from AWSIM. Exact long-run bitwise agreement is not a
  requirement.
- PAWSIM can run AWSIM-compatible gathered multigrid, but distributed multigrid
  is the intended production pressure path for MPI runs.

## Deferred Or User-Supplied Functionality

- AWSIM's `useTrad`/RK-style time-stepping option is not ported. PAWSIM supports
  the AB1/AB2/AB3 path used by the current test cases and production-style
  integrations.
- `OmegaxFile` and `OmegayFile` are not parsed. The present AWSIM/PAWSIM
  shallow-water dynamics use `OmegazFile` for the q-grid Coriolis term.
- AWSIM's SOR auto-optimization controls (`SOR_rp_max`, `SOR_rp_acc`, and
  `SOR_opt_freq`) are not reproduced. PAWSIM accepts `SOR_rp` directly, maps
  legacy `SOR_rp_min` to that value for compatibility, and uses multigrid as
  the main pressure-solver path.
- AWSIM random forcing is not ported. A parallel version should be designed
  around an explicit distributed RNG/FFT strategy rather than copying the serial
  FFTW state. For now, users should prescribe deterministic forcing fields; a
  time-dependent barotropic forcing reader can be added when needed.
- Heavyweight regression outputs are not distributed with the source tree. Local
  comparison scripts generate temporary outputs under `/private/tmp`.
- More extensive MPI performance validation remains future work, especially
  larger grids, more rank counts, non-square decompositions, and non-power-of-two
  grids.

## Local Validation Performed

- Short MPI smoke/regression: rank 1 versus rank 4 on the gyre smoke case.
- Distributed-pressure boundary tests: wall-wall, periodic-periodic, mixed
  wall/periodic, and 250 x 250 gathered-coarse-tail cases.
- 25-day local comparison sweep for ACC, cavity, gyre, and jets with all
  non-random diagnostics enabled. Rank 1 and rank 4 PAWSIM agree closely; AWSIM
  and PAWSIM agree to small scientific tolerances over these integrations.
