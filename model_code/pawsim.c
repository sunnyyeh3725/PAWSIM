/**
 * pawsim.c
 *
 * MPI-parallel AWSIM-compatible executable.
 *
 * The main program mirrors AWSIM's high-level order: read input, allocate and
 * load state, build work arrays, perform the preliminary rigid-lid correction,
 * then enter the time stepping loop. Most physics lives in work.c/tendency.c;
 * this file owns orchestration, output scheduling, and pressure timing.
 *
 */
#include "config.h"
#include "context.h"
#include "diagnostics.h"
#include "domain.h"
#include "field.h"
#include "forcing.h"
#include "integrator.h"
#include "io.h"
#include "pressure.h"
#include "state.h"
#include "tendency.h"
#include "work.h"

#include <math.h>
#include <time.h>

#define PAWSIM_NG 3

/** Print the AWSIM-compatible command-line form. */
static void printUsage (void)
{
  fprintf(stderr,"Usage: PAWSIM.exe <input_file> <output_dir>\n");
}

/** Release all context-owned storage; safe on error exits after partial setup. */
static void freeContext (pawsim_context * ctx)
{
  /** Free in reverse ownership order so partially initialized exits are safe. */
  pawsim_diagnostics_free(ctx);
  pawsim_field2d_free(&ctx->pi);
  pawsim_field2d_free(&ctx->pi_rhs);
  pawsim_work_free(&ctx->work);
  pawsim_state_free(&ctx->tendency);
  pawsim_state_free(&ctx->tendency_1);
  pawsim_state_free(&ctx->tendency_2);
  pawsim_state_free(&ctx->state);
}

/** Return wall-clock time from MPI when available, otherwise process CPU time. */
static double pawsimWalltime (void)
{
#ifdef PAWSIM_USE_MPI
  return MPI_Wtime();
#else
  return (double) clock() / (double) CLOCKS_PER_SEC;
#endif
}

static const char * pressureSolverName (uint solver)
{
  switch (solver)
  {
    case PAWSIM_PRESSURE_SOLVER_SOR:
      return "SOR";
    case PAWSIM_PRESSURE_SOLVER_MG_GATHERED:
      return "MG_GATHERED";
    case PAWSIM_PRESSURE_SOLVER_MG_DISTRIBUTED:
      return "MG_DISTRIBUTED";
    default:
      return "UNKNOWN";
  }
}

/** Run one pressure correction and record optional per-solve timing diagnostics. */
static uint pressureCorrectTimed (pawsim_context * ctx, uint n, const char * phase,
                                  bool update_diags)
{
  double t0;
  double local_elapsed;
  double elapsed;
  uint iters;

  /** Report the maximum rank wall time, which is the elapsed solve time. */
  t0 = pawsimWalltime();
  iters = pawsim_pressure_correct_rigid_lid(ctx,update_diags);
  local_elapsed = pawsimWalltime() - t0;
  MPI_Allreduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,ctx->dom.comm);

  ctx->pressure_solve_count ++;
  if (ctx->pressure_last_fallback)
  {
    ctx->pressure_fallback_count ++;
  }
  ctx->pressure_iters_total += iters;
  ctx->pressure_walltime_total += elapsed;

  if (ctx->cfg.pressureTiming && (ctx->dom.rank == 0))
  {
    printf("PAWSIM pressure solve: phase=%s n=%u solver=%s",phase,n,
           pressureSolverName(ctx->pressure_last_solver));
    if (ctx->pressure_last_fallback)
    {
      printf(" requested=%s fallback=1",pressureSolverName(ctx->cfg.pressureSolver));
    }
    printf(" iters=%u walltime=%0.6f s residual_max=%0.6e residual_rms=%0.6e div_max=%0.6e div_rms=%0.6e\n",
           iters,elapsed,ctx->pressure_last_residual_max,ctx->pressure_last_residual_rms,
           ctx->pressure_last_divergence_max,ctx->pressure_last_divergence_rms);
    fflush(stdout);
  }

  return iters;
}

//////////////////////////////////////////
///// BEGIN READING INPUT PARAMETERS /////
//////////////////////////////////////////
static bool readInputParameters (pawsim_context * ctx)
{
  pawsim_config_defaults(&ctx->cfg);
  return pawsim_config_read(ctx->infname,&ctx->cfg,stderr)
      && pawsim_config_validate(&ctx->cfg,stderr);
}

/** Infer Lx if needed and compute the uniform C-grid spacings. */
static void setGridDimensions (pawsim_context * ctx)
{
  /** AWSIM inputs often specify Ly and infer Lx from Nx/Ny. */
  if (ctx->cfg.Lx <= 0)
  {
    ctx->cfg.Lx = ctx->cfg.Ly * ctx->cfg.Nx / ctx->cfg.Ny;
  }

  ctx->dx = ctx->cfg.Lx / ctx->cfg.Nx;
  ctx->dy = ctx->cfg.Ly / ctx->cfg.Ny;
}

/** Resolve AWSIM's "choose exactly two of Nt, tmax, dt" time convention. */
static bool setTimeParameters (pawsim_context * ctx)
{
  uint tParamCnt;

  tParamCnt = ((ctx->cfg.Nt > 0) ? 1 : 0)
            + ((ctx->cfg.tmax > 0) ? 1 : 0)
            + ((ctx->cfg.dt > 0) ? 1 : 0);

  /*
   * AWSIM convention: exactly two of Nt, tmax, and dt determine the third.
   * This avoids silent inconsistencies in copied setup files.
   */
  if (tParamCnt != 2)
  {
    fprintf(stderr,"Must specify exactly two of {Nt, tmax, dt}\n");
    return false;
  }

  if (ctx->cfg.tmin < 0)
  {
    ctx->cfg.tmin = ctx->cfg.restart ? ctx->cfg.startIdx*ctx->cfg.savefrequency : 0;
  }

  if (ctx->cfg.Nt == 0)
  {
    if (ctx->cfg.tmax <= ctx->cfg.tmin)
    {
      fprintf(stderr,"Integration end time must exceed integration start time\n");
      return false;
    }
    ctx->cfg.Nt = (uint) ceil((ctx->cfg.tmax-ctx->cfg.tmin)/ctx->cfg.dt);
  }
  if (ctx->cfg.dt == 0)
  {
    if (ctx->cfg.tmax <= ctx->cfg.tmin)
    {
      fprintf(stderr,"Integration end time must exceed integration start time\n");
      return false;
    }
    ctx->cfg.dt = (ctx->cfg.tmax-ctx->cfg.tmin) / ctx->cfg.Nt;
  }
  if (ctx->cfg.savefrequency == 0)
  {
    ctx->cfg.savefrequency = ctx->cfg.dt;
  }
  if (ctx->cfg.tmax == 0)
  {
    ctx->cfg.tmax = ctx->cfg.tmin + ctx->cfg.dt * ctx->cfg.Nt;
  }
  if (ctx->cfg.tmin >= ctx->cfg.tmax)
  {
    fprintf(stderr,"Integration start time exceeds integration end time\n");
    return false;
  }

  ctx->t = ctx->cfg.tmin;
  return true;
}
////////////////////////////////////////
///// END READING INPUT PARAMETERS /////
////////////////////////////////////////

///////////////////////////////////
///// BEGIN MEMORY ALLOCATION /////
///////////////////////////////////
static bool allocateModelState (pawsim_context * ctx)
{
  /*
   * The context keeps prognostic state, three tendency buffers for AB3, work
   * arrays for AWSIM's tderiv-style calculations, and pressure fields.
   */
  return pawsim_state_alloc(&ctx->state,&ctx->cfg,&ctx->dom)
      && pawsim_state_alloc(&ctx->tendency,&ctx->cfg,&ctx->dom)
      && pawsim_state_alloc(&ctx->tendency_1,&ctx->cfg,&ctx->dom)
      && pawsim_state_alloc(&ctx->tendency_2,&ctx->cfg,&ctx->dom)
      && pawsim_work_alloc(&ctx->work,&ctx->cfg,&ctx->dom)
      && pawsim_field2d_alloc(&ctx->pi,ctx->dom.nx,ctx->dom.ny,ctx->dom.nghost)
      && pawsim_field2d_alloc(&ctx->pi_rhs,ctx->dom.nx,ctx->dom.ny,ctx->dom.nghost)
      && pawsim_diagnostics_alloc(ctx);
}
/////////////////////////////////
///// END MEMORY ALLOCATION /////
/////////////////////////////////

/////////////////////////////////////
///// BEGIN PARAMETER DEFAULTS  /////
/////////////////////////////////////
static void setParameterDefaults (pawsim_context * ctx)
{
  pawsim_field2d_zero(&ctx->pi);
  pawsim_field2d_zero(&ctx->pi_rhs);
  pawsim_state_zero_prognostic(&ctx->tendency);
  pawsim_state_zero_prognostic(&ctx->tendency_1);
  pawsim_state_zero_prognostic(&ctx->tendency_2);
  pawsim_work_zero(&ctx->work);
  pawsim_diagnostics_zero_averages(ctx);
  pawsim_diagnostics_zero_budgets(ctx);
}
///////////////////////////////////
///// END PARAMETER DEFAULTS  /////
///////////////////////////////////

////////////////////////////////////////
///// BEGIN READING PARAMETER DATA /////
////////////////////////////////////////
static bool readParameterData (pawsim_context * ctx)
{
  if (!pawsim_state_init(&ctx->state,&ctx->cfg,&ctx->dom,stderr))
  {
    return false;
  }

  if (ctx->cfg.restart)
  {
    return pawsim_state_read_restart(&ctx->state,&ctx->cfg,&ctx->dom,
                                     ctx->outdir,ctx->cfg.startIdx,stderr);
  }

  return true;
}
//////////////////////////////////////
///// END READING PARAMETER DATA /////
//////////////////////////////////////

/////////////////////////////////////////////
///// BEGIN TIME STEPPING PRELIMINARIES /////
/////////////////////////////////////////////
static bool timeSteppingPreliminaries (pawsim_context * ctx)
{
  uint iters;

  pawsim_tendency_prepare_work_arrays(ctx);
  if (!pawsim_forcing_update(ctx,ctx->t))
  {
    return false;
  }
  pawsim_tendency_compute_deterministic_core(ctx,&ctx->state);

  /*
   * As in AWSIM, a preliminary pressure correction updates the initial
   * velocities before writing n=0 diagnostics. Work arrays are then rebuilt so
   * init_* diagnostics describe the pressure-corrected state.
   */
  if (ctx->cfg.useRL)
  {
    iters = pressureCorrectTimed(ctx,0,"prelim",false);
    if (ctx->dom.rank == 0)
    {
      printf("PAWSIM pressure iterations: %u\n",iters);
    }
    pawsim_tendency_prepare_work_arrays(ctx);
    if (!pawsim_forcing_update(ctx,ctx->t))
    {
      return false;
    }
    pawsim_tendency_compute_deterministic_core(ctx,&ctx->state);
  }

  if (!pawsim_diagnostics_write_initial_outputs(ctx))
  {
    return false;
  }
  pawsim_diagnostics_zero_budgets(ctx);

  return true;
}
///////////////////////////////////////////
///// END TIME STEPPING PRELIMINARIES /////
///////////////////////////////////////////

///////////////////////////////
///// BEGIN TIME STEPPING /////
///////////////////////////////
static bool timeStepModel (pawsim_context * ctx, uint n)
{
  return pawsim_integrator_step(ctx,n);
}

/** Write every model-state record whose save time has been reached. */
static bool writeScheduledModelState (pawsim_context * ctx)
{
  real t_save;
  real eps;

  if (ctx->cfg.savefrequency <= 0)
  {
    return true;
  }

  eps = 1e-9 * fabs(ctx->cfg.dt);
  /*
   * Saves are scheduled by model time, not by integer step count, because
   * copied AWSIM inputs may specify any two of Nt/tmax/dt.
   */
  while (true)
  {
    t_save = ctx->n_saves * ctx->cfg.savefrequency;
    if (ctx->t + eps < t_save)
    {
      break;
    }

    if (!pawsim_diagnostics_write_model_state(ctx,ctx->n_saves))
    {
      fprintf(stderr,"Unable to write model state n=%u\n",ctx->n_saves);
      return false;
    }
    ctx->n_saves ++;
  }

  return true;
}
/////////////////////////////
///// END TIME STEPPING /////
/////////////////////////////

int main (int argc, char ** argv)
{
  pawsim_context ctx;
  uint n;

  memset(&ctx,0,sizeof(ctx));
  pawsim_mpi_init(&argc,&argv);

  if (argc < 3)
  {
    printUsage();
    pawsim_mpi_finalize();
    return 1;
  }
  ctx.infname = argv[1];
  ctx.outdir = argv[2];

  if (!readInputParameters(&ctx))
  {
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  setGridDimensions(&ctx);
  if (!setTimeParameters(&ctx))
  {
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  /*
   * Domain periodicity is the inverse of the AWSIM wall flags. PAWSIM_NG=3 is
   * retained because the TW81 stencil reaches farther in the q-grid direction.
   */
  if (!pawsim_domain_init(&ctx.dom,ctx.cfg.Nx,ctx.cfg.Ny,PAWSIM_NG,!ctx.cfg.useWallEW,!ctx.cfg.useWallNS))
  {
    fprintf(stderr,"ERROR: Could not initialize PAWSIM domain\n");
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  pawsim_domain_print(&ctx.dom);

  if (!allocateModelState(&ctx))
  {
    fprintf(stderr,"ERROR: Could not allocate PAWSIM fields\n");
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  setParameterDefaults(&ctx);

  if (!readParameterData(&ctx))
  {
    fprintf(stderr,"ERROR: Could not initialize PAWSIM state\n");
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  if (ctx.dom.rank == 0)
  {
    printf("PAWSIM parsed %u x %u x %u AWSIM input\n",ctx.cfg.Nx,ctx.cfg.Ny,ctx.cfg.Nlay);
    fflush(stdout);
  }

  ctx.n_saves = ctx.cfg.restart ? ctx.cfg.startIdx : 0;
  if (ctx.cfg.savefreqAvg > 0)
  {
    /** Restarted runs continue AWSIM's diagnostic numbering convention. */
    ctx.n_avg = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqAvg) + 1;
    ctx.t_next_avg = ctx.cfg.tmin + ctx.cfg.savefreqAvg;
  }
  if (ctx.cfg.savefreqEZ > 0)
  {
    ctx.n_EZ = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqEZ) + 1;
    ctx.t_next_EZ = ctx.cfg.tmin + ctx.cfg.savefreqEZ;
  }
  if (ctx.cfg.savefreqUMom > 0)
  {
    ctx.n_avg_hu = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqUMom) + 1;
    ctx.t_next_avg_hu = ctx.cfg.tmin + ctx.cfg.savefreqUMom;
  }
  if (ctx.cfg.savefreqVMom > 0)
  {
    ctx.n_avg_hv = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqVMom) + 1;
    ctx.t_next_avg_hv = ctx.cfg.tmin + ctx.cfg.savefreqVMom;
  }
  if (ctx.cfg.savefreqThic > 0)
  {
    ctx.n_avg_h = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqThic) + 1;
    ctx.t_next_avg_h = ctx.cfg.tmin + ctx.cfg.savefreqThic;
  }
  if (ctx.cfg.savefreqEnergy > 0)
  {
    ctx.n_avg_e = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqEnergy) + 1;
    ctx.t_next_avg_e = ctx.cfg.tmin + ctx.cfg.savefreqEnergy;
  }
  if (ctx.cfg.savefreqTracer > 0)
  {
    ctx.n_avg_b = (uint) round(ctx.cfg.tmin/ctx.cfg.savefreqTracer) + 1;
    ctx.t_next_avg_b = ctx.cfg.tmin + ctx.cfg.savefreqTracer;
  }
  if (!timeSteppingPreliminaries(&ctx))
  {
    freeContext(&ctx);
    pawsim_mpi_finalize();
    return 1;
  }

  for (n = 1; n <= ctx.cfg.Nt; n ++)
  {
    if (!timeStepModel(&ctx,n))
    {
      fprintf(stderr,"ERROR: PAWSIM time step failed at n=%u\n",n);
      freeContext(&ctx);
      pawsim_mpi_finalize();
      return 1;
    }

    if (ctx.cfg.useRL)
    {
      /*
       * Rigid-lid pressure projection is applied after the explicit AB update,
       * matching AWSIM's split between tderiv and the barotropic correction.
       */
      uint iters = pressureCorrectTimed(&ctx,n,"step",true);
      if ((ctx.dom.rank == 0) && (iters == ctx.cfg.maxiters))
      {
        fprintf(stderr,"WARNING: PAWSIM pressure solve reached maxiters at n=%u\n",n);
      }
    }

    if (!writeScheduledModelState(&ctx))
    {
      freeContext(&ctx);
      pawsim_mpi_finalize();
      return 1;
    }

    if (!pawsim_diagnostics_accumulate_averages(&ctx)
     || !pawsim_diagnostics_write_averages_if_due(&ctx)
     || !pawsim_diagnostics_write_EZ_if_due(&ctx)
     || !pawsim_diagnostics_write_budgets_if_due(&ctx,n))
    {
      fprintf(stderr,"ERROR: PAWSIM diagnostic averaging failed at n=%u\n",n);
      freeContext(&ctx);
      pawsim_mpi_finalize();
      return 1;
    }
  }

  if (ctx.cfg.pressureTiming && (ctx.dom.rank == 0) && (ctx.pressure_solve_count > 0))
  {
    printf("PAWSIM pressure summary: requested=%s solves=%u fallback_solves=%u total_walltime=%0.6f s mean_walltime=%0.6f s mean_iters=%0.2f\n",
           pressureSolverName(ctx.cfg.pressureSolver),
           ctx.pressure_solve_count,
           ctx.pressure_fallback_count,
           ctx.pressure_walltime_total,
           ctx.pressure_walltime_total/ctx.pressure_solve_count,
           (double) ctx.pressure_iters_total/ctx.pressure_solve_count);
    fflush(stdout);
  }

  freeContext(&ctx);
  pawsim_mpi_finalize();
  return 0;
}
