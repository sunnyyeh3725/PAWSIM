# PAWSIM

PAWSIM is the MPI-parallel AWSIM implementation.

The compatibility target is scientific equivalence with AWSIM, not permanent
bitwise identity. Short regression runs are expected to match AWSIM to roundoff
where the physics actually affects the prognostic state. Longer integrations may
separate at small amplitude through accumulated floating-point ordering
differences, pressure-solver tolerances, and deliberately cleaned-up boundary
diagnostics.

The executable:

- read an AWSIM input parameter file;
- create a 2-D Cartesian domain decomposition;
- allocate local 2-D arrays with halos;
- exchange halos;
- run AWSIM-style AB1/AB2/AB3 time stepping with distributed SOR, gathered
  multigrid, or distributed multigrid pressure correction;
- gather and write AWSIM/Matlab-compatible binary outputs.

PAWSIM is self-contained: it carries the AWSIM-compatible utility definitions
and serial multigrid implementation it still needs in this directory, so it can
be copied and built without a sibling `model_code` directory.

`Make.sh` uses an MPI C compiler wrapper and enables `PAWSIM_USE_MPI` when MPI
is available on `PATH`. It checks `mpicc`, `mpiicx`, and `mpiicc` by default.
Otherwise it builds a single-process compatibility executable with `gcc`, which
is useful for developing the PAWSIM structure before MPI is installed.

For cluster builds, override the compiler selection without editing `Make.sh`.
The preferred route is to point `PAWSIM_MPICC` at the MPI wrapper from the local
MPI installation:

```sh
PAWSIM_MPICC=/opt/intel/oneapi/mpi/latest/bin/mpiicx sh Make.sh
```

Intel MPI wrappers are small shell scripts that invoke a backend compiler. If
the wrapper exists but reports an error such as `mpiicx: command not found`,
the compiler itself is not on `PATH` in the build shell. Either source the
cluster's oneAPI setup first:

```sh
source /opt/intel/oneapi/setvars.sh
sh Make.sh
```

or pass the compiler bin directory directly to the PAWSIM build:

```sh
PAWSIM_MPICC=/opt/intel/oneapi/mpi/latest/bin/mpiicx \
PAWSIM_COMPILER_BIN=/opt/intel/oneapi/compiler/2025.2/bin \
sh Make.sh
```

If the cluster exposes `mpiicc` but not `mpiicx` cleanly, you can also tell the
wrapper which backend compiler to use:

```sh
PAWSIM_MPICC=/opt/intel/oneapi/mpi/latest/bin/mpiicc \
PAWSIM_MPI_BACKEND_CC=/opt/intel/oneapi/compiler/2025.2/bin/icx \
sh Make.sh
```

If the local setup exposes the underlying C compiler directly, such as
`/opt/intel/oneapi/compiler/2025.2/bin/icx`, force MPI mode and provide the MPI
include/link flags explicitly:

```sh
PAWSIM_CC=/opt/intel/oneapi/compiler/2025.2/bin/icx \
PAWSIM_USE_MPI=1 \
PAWSIM_MPI_CFLAGS="-I/opt/intel/oneapi/mpi/latest/include" \
PAWSIM_MPI_LDFLAGS="-L/opt/intel/oneapi/mpi/latest/lib" \
PAWSIM_MPI_LIBS="-lmpi" \
sh Make.sh
```

Set `PAWSIM_USE_MPI=0` to force a serial compatibility build. Standard `CFLAGS`,
`LDFLAGS`, and `LIBS` are also honored, and PAWSIM-specific MPI flags are
appended in MPI builds.

## Matlab Run Scripts

The existing Matlab setup scripts still generate AWSIM-style input directories.
`matlab_common/createRunScript.m` remains backward-compatible with the old
serial AWSIM call, but it also accepts optional MPI arguments:

```matlab
createRunScript(local_home_dir, run_name, model_code_dir, exec_name, ...
                use_intel, use_pbs, use_cluster, uname, ...
                cluster_addr, cluster_home_dir, ...
                use_mpi, mpi_nproc, mpi_launcher)
```

For a local PAWSIM run, set `exec_name = 'PAWSIM.exe'`,
`model_code_dir = fullfile('../../','pawsim')`, `use_mpi = true`, and
`mpi_nproc` to the desired rank count. With the default launcher, the generated
`Run.sh` uses:

```sh
mpirun -np <mpi_nproc> ./PAWSIM.exe <run_name>_in .
```

For scheduler runs, `createRunScript` now requests `mpi_nproc` tasks in the
cluster template. SGE/GridEngine templates launch with
`mpirun -np ${NSLOTS:-<mpi_nproc>}`, while the SLURM template launches with
`srun -n ${SLURM_NTASKS:-<mpi_nproc>}`. Pass a non-empty `mpi_launcher` string
to override this on a particular machine.

## Pressure Solver Options

PAWSIM still accepts AWSIM-style `use_MG` input files:

- `use_MG 0` selects distributed SOR;
- `use_MG 1` selects the current gathered rank-0 multigrid path.

For PAWSIM-specific tests, `pressureSolver` can be used instead:

- `pressureSolver 0`: distributed SOR;
- `pressureSolver 1`: gathered multigrid;
- `pressureSolver 2`: distributed multigrid.

Set `pressureTiming 1` in an input file to print per-solve pressure timing and
a summary from rank 0. The reported wall time is the maximum elapsed pressure
solve time across ranks. Timed rigid-lid solves also report pressure-equation
residual norms and the remaining barotropic divergence after the pressure
correction.

The distributed multigrid path uses distributed smoothing and restriction on
clean 2:1 levels. On coarse levels where each rank would otherwise own only a
few cells, PAWSIM agglomerates ranks by creating a smaller Cartesian
communicator for that level. Transitions that are no longer locally nested use
a gathered restriction/prolongation transfer, but smoothing and residual
evaluation on the agglomerated coarse level remain distributed across the
active ranks. This keeps power-of-two grids mostly distributed while still
allowing cases such as 250 x 250 to use `pressureSolver 2` without reverting
the whole pressure solve to gathered multigrid. Timing output marks true
whole-solver fallbacks with `fallback=1`, and the summary reports
`fallback_solves`.

## AWSIM Compatibility

PAWSIM follows the AWSIM parameter-file format and binary output layout. The
same setup files can usually be passed directly to `PAWSIM.exe`; PAWSIM-only
keys are optional extensions.

Known intentional differences:

- PAWSIM suppresses momentum budget diagnostics on inactive wall-normal
  velocity faces. AWSIM computes some diagnostic terms there and then zeros the
  actual wall-normal velocity tendency, so those entries are diagnostic-only.
- Random forcing is not ported. Users who need stochastic or time-dependent
  barotropic forcing should prescribe forcing fields through input files; a
  time-dependent barotropic forcing reader can be added when a concrete use case
  needs it.
- The distributed pressure solve can take a different floating-point path from
  AWSIM's serial/gathered solve. Short tests should match where it matters, but
  long nonlinear integrations should be compared with scientific tolerances.

## Developer Notes

The PAWSIM source is organized around the AWSIM time-step sequence:

- `config.*` reads AWSIM key/value inputs plus PAWSIM pressure extensions.
- `defs.*`, `nsode.h`, and `multigrid.*` are local compatibility/support files
  retained from AWSIM so PAWSIM does not depend on the serial source tree.
- `domain.*` owns the Cartesian MPI layout and local/global index mapping.
- `field.*` owns local storage and halo operations. Rank exchange and physical
  wall fills are separate ideas, even when a public helper performs both.
- `state.*` reads global AWSIM input files on rank 0 and scatters local tiles,
  including tracer state and active-buoyancy inputs when `useTracer` is set.
- `work.*` contains the ported AWSIM stencil machinery for face thicknesses,
  q-grid PV, Bernoulli functions, viscosity, tracer advection/diffusion,
  forcing masks, and tendencies.
- `forcing.*` applies wind, static barotropic forcing, bottom drag, and
  surface-drag/lid/wind-feedback terms.
- `restoring.*` applies Newtonian restoring and its AWSIM-style interface
  velocity contribution.
- `diapycnal.*` applies prescribed interface velocity momentum advection,
  tracer diapycnal diffusion, convective-adjustment checks, and diapycnal
  momentum viscosity.
- `tendency.*` is the PAWSIM equivalent of AWSIM's `tderiv` callback.
- `integrator.*` owns Adams-Bashforth startup/history.
- `pressure.*` owns rigid-lid pressure correction, distributed/gathered
  pressure solvers, and pressure diagnostics.
- `diagnostics.*` and `io.*` gather and write AWSIM/Matlab-compatible outputs.

The `init_*` diagnostics are intentionally verbose regression aids. They are not
normal AWSIM outputs, but they make it much easier to catch mistakes in halo
fills, q-grid ownership, pressure geometry, and tendency terms before the errors
have propagated through a long integration. They are off by default; set
`writeInitDiagnostics 1` in a PAWSIM input file to enable them.

For arbitrary grids, `pressureSolver 2` currently uses distributed fine-grid
work plus a gathered coarse tail. This is correct enough for rank-consistency
tests and avoids whole-solver fallback, but the 250 x 250 25-day gyre timing
shows that the coarse tail is a real scaling bottleneck. A future optimization
would distribute the ragged coarse transition instead of gathering it.

For best multigrid performance, choose grid sizes with a large power-of-two
factor in each horizontal direction. Exact powers of two are ideal, but sizes
such as `192 = 3*64`, `320 = 5*64`, or `384 = 3*128` remain friendly because
they halve many times before reaching an odd level. Sizes such as `200`, `250`,
or `500` are more awkward because they reach odd coarse levels much earlier.
For example, a user considering a `200 x 200` run should consider `192 x 192`
if the small resolution change is scientifically acceptable: the pressure
hierarchy goes `192 -> 96 -> 48 -> 24 -> 12 -> 6`, rather than
`200 -> 100 -> 50 -> 25`.

## Regression Tests

The main PAWSIM MPI smoke/regression is:

```sh
./test_pawsim.sh
```

The distributed pressure boundary/hybrid regression checks 1-rank versus
4-rank `pressureSolver 2` output for wall-wall, periodic-periodic, and mixed
wall/periodic configurations. It also verifies that a 250 x 250 four-rank case
uses the gathered coarse-tail path without whole-solver fallback:

```sh
./test_pawsim_pressure_boundaries.sh
```

The grid-size recommendation smoke test compares matched `192 x 192` and
`200 x 200` gyre setups with pressure timing enabled:

```sh
./test_pressure_grid_size_recommendations.sh 4 20
```

The timing ratio is machine-dependent, so this is not a strict performance
regression. It prints the coarsening paths and the measured mean pressure
solve times for the requested rank count and number of short time steps.

The broader 25-day comparison sweep used during development is:

```sh
python3 compare_examples_25d.py --ranks 1 4
```

It runs ACC, cavity, gyre, and jets cases with all non-random AWSIM diagnostics,
then writes a CSV summary and comparison plots. These outputs are useful for
local development but are not intended to be distributed as a heavyweight
regression dataset.

## Pressure Scaling Benchmarks

`benchmark_pressure_scaling.sh` builds PAWSIM and runs the same input file with
several MPI sizes:

```sh
./benchmark_pressure_scaling.sh ../runs/test_gyre/test_gyre_AL81_25d_in /private/tmp/pawsim_pressure_scaling 1 2 4 8
```

The script creates one output directory and log per rank count. Use an input
file with `pressureTiming 1` for pressure-specific timings; the script also
prints whole-run elapsed time.
