#!/bin/sh
#
# Build PAWSIM.
#
# By default this script uses an MPI C wrapper when one is available on PATH and
# otherwise builds the single-process compatibility executable.  Cluster builds
# can override the compiler selection without editing this file:
#
#   PAWSIM_MPICC=/path/to/mpicc sh Make.sh
#   PAWSIM_MPICC=/path/to/mpiicx PAWSIM_COMPILER_BIN=/path/to/compiler/bin sh Make.sh
#   PAWSIM_CC=/path/to/icx PAWSIM_USE_MPI=1 \
#     PAWSIM_MPI_CFLAGS="-I/path/to/mpi/include" \
#     PAWSIM_MPI_LDFLAGS="-L/path/to/mpi/lib" \
#     PAWSIM_MPI_LIBS="-lmpi" sh Make.sh
#
# Set PAWSIM_USE_MPI=0 to force a serial compatibility build.

set -eu

sources="defs.c multigrid.c domain.c field.c config.c io.c state.c work.c pressure.c diagnostics.c forcing.c restoring.c diapycnal.c tendency.c integrator.c pawsim.c"
objects="defs.o multigrid.o domain.o field.o config.o io.o state.o work.o pressure.o diagnostics.o forcing.o restoring.o diapycnal.o tendency.o integrator.o pawsim.o"

base_cflags="${CFLAGS:-"-O3 -Wall -Wextra"}"
ldflags="${LDFLAGS:-}"
libs="${LIBS:-"-lm"}"

want_mpi="${PAWSIM_USE_MPI:-auto}"
mpi_cc="${PAWSIM_MPICC:-${MPICC:-}}"
cc="${PAWSIM_CC:-${CC:-}}"

if [ -n "${PAWSIM_COMPILER_BIN:-}" ]
then
  PATH="${PAWSIM_COMPILER_BIN}:${PATH}"
  export PATH
fi

if [ -n "${PAWSIM_MPI_BACKEND_CC:-}" ]
then
  I_MPI_CC="${PAWSIM_MPI_BACKEND_CC}"
  MPICH_CC="${PAWSIM_MPI_BACKEND_CC}"
  export I_MPI_CC MPICH_CC
fi

if [ "${want_mpi}" != "0" ]
then
  if [ -n "${mpi_cc}" ]
  then
    cc="${mpi_cc}"
    use_mpi=1
  elif command -v mpicc >/dev/null 2>&1
  then
    cc=mpicc
    use_mpi=1
  elif command -v mpiicx >/dev/null 2>&1
  then
    cc=mpiicx
    use_mpi=1
  elif command -v mpiicc >/dev/null 2>&1
  then
    cc=mpiicc
    use_mpi=1
  elif [ "${want_mpi}" = "1" ]
  then
    if [ -z "${cc}" ]
    then
      echo "ERROR: PAWSIM_USE_MPI=1 but no MPI compiler wrapper or PAWSIM_CC/CC was set" >&2
      echo "Set PAWSIM_MPICC to an MPI wrapper, or set PAWSIM_CC plus PAWSIM_MPI_CFLAGS/PAWSIM_MPI_LDFLAGS/PAWSIM_MPI_LIBS." >&2
      exit 1
    fi
    use_mpi=1
  else
    use_mpi=0
  fi
else
  use_mpi=0
fi

if [ "${use_mpi}" -eq 1 ]
then
  cflags="${base_cflags} -DPAWSIM_USE_MPI ${PAWSIM_MPI_CFLAGS:-}"
  ldflags="${ldflags} ${PAWSIM_MPI_LDFLAGS:-}"
  libs="${libs} ${PAWSIM_MPI_LIBS:-}"
else
  cc="${cc:-gcc}"
  cflags="${base_cflags}"
fi

echo "PAWSIM build: cc=${cc}"
echo "PAWSIM build: mpi=${use_mpi}"

"${cc}" ${cflags} -I. -c ${sources}
"${cc}" ${ldflags} ${objects} ${libs} -o PAWSIM.exe
rm -f ${objects}
