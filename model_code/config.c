/**
 * config.c
 *
 * Input configuration implementation for PAWSIM.
 *
 * PAWSIM intentionally reads the same simple key/value input files as AWSIM.
 * New PAWSIM-only options, such as pressureSolver, pressureTiming, and
 * writeInitDiagnostics, are optional extensions; when they are absent, use_MG
 * retains its AWSIM meaning.
 *
 */
#include "config.h"

/** Copy an input string into a fixed-size PAWSIM config buffer. */
static void copy_config_string (char * dst, const char * src)
{
  size_t len;

  if (dst == NULL)
  {
    return;
  }
  if (src == NULL)
  {
    dst[0] = '\0';
    return;
  }

  len = strlen(src);
  if (len >= MAX_PARAMETER_FILENAME_LENGTH)
  {
    len = MAX_PARAMETER_FILENAME_LENGTH - 1;
  }
  memcpy(dst,src,len);
  dst[len] = '\0';
}

/*
 * Initialize all PAWSIM configuration fields before reading an AWSIM-style
 * key/value input file. Defaults follow AWSIM where there is an exact analogue.
 */
void pawsim_config_defaults (pawsim_config * cfg)
{
  if (cfg == NULL)
  {
    return;
  }

  memset(cfg,0,sizeof(*cfg));
  cfg->Nlay = 0;
  cfg->Nx = 0;
  cfg->Ny = 0;
  cfg->Nt = 0;
  cfg->startIdx = 0;
  /** Match AWSIM defaults where possible, then let input files override them. */
  cfg->timeSteppingScheme = TIMESTEPPING_AB3;
  cfg->momentumScheme = MOMENTUM_TW81;
  cfg->thicknessScheme = THICKNESS_AL81;
  cfg->tracerScheme = TRACER_AL81;
  cfg->Lx = 0;
  cfg->Ly = 0;
  cfg->dt = 0;
  cfg->tmin = -1;
  cfg->tmax = 0;
  cfg->savefrequency = 0;
  cfg->savefreqAvg = 0;
  cfg->savefreqUMom = 0;
  cfg->savefreqVMom = 0;
  cfg->savefreqThic = 0;
  cfg->savefreqEnergy = 0;
  cfg->savefreqTracer = 0;
  cfg->savefreqEZ = 0;
  cfg->KT00_sigma = 1.4;
  cfg->h0 = 0;
  cfg->hsml = 0;
  cfg->hmin_surf = 0;
  cfg->hbbl = 0;
  cfg->hmin_bot = 0;
  cfg->A2 = 0;
  cfg->A4 = 0;
  cfg->A2smag = 0;
  cfg->A4smag = 0;
  cfg->K2 = 0;
  cfg->K4 = 0;
  cfg->linDragCoeff = 0;
  cfg->quadDragCoeff = 0;
  cfg->linDragSurf = 0;
  cfg->quadDragSurf = 0;
  cfg->pi_tol = 1e-10;
  cfg->SOR_rp = 1.0;
  cfg->tauPeriod = 0;
  cfg->wDiaPeriod = 0;
  cfg->convDiff = 0;
  cfg->omega_MG = 2.0/3.0;
  cfg->tauNrecs = 1;
  cfg->wDiaNrecs = 1;
  cfg->maxiters = 10000;
  cfg->pressureSolver = PAWSIM_PRESSURE_SOLVER_SOR;
  cfg->useWind = false;
  cfg->useFbaro = false;
  cfg->useRelax = false;
  cfg->useWDia = false;
  cfg->useDiaDiff = false;
  cfg->useDiaVisc = false;
  cfg->use_MG = false;
  cfg->use_fullMG = false;
  cfg->pressureTiming = false;
  cfg->writeInitDiagnostics = false;
  cfg->restart = false;
  cfg->useWallEW = false;
  cfg->useWallNS = false;
  cfg->useRL = true;
  cfg->oceanSurfDrag = true;
  cfg->windFeedback = false;
  cfg->useTracer = false;
  cfg->useBuoyancy = false;
  cfg->useRandomForcing = false;
  cfg->useTrad = false;
  strcpy(cfg->inputDir,".");
}

/*
 * Read one whitespace-delimited key/value pair at a time from an AWSIM input
 * file. Unknown keys are tolerated so older or richer AWSIM setup files can be
 * reused while PAWSIM explicitly rejects only unsupported active machinery.
 */
bool pawsim_config_read (const char * fname, pawsim_config * cfg, FILE * errstrm)
{
  FILE * infile = NULL;
  const char * slash = NULL;
  char key[256];
  char value[256];
  bool pressureSolverSpecified = false;

  if ((fname == NULL) || (cfg == NULL))
  {
    return false;
  }

  infile = fopen(fname,"r");
  if (infile == NULL)
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Could not open input parameter file %s\n",fname);
    }
    return false;
  }

  /*
   * Relative data-file paths in AWSIM inputs are interpreted relative to the
   * input file itself, not the directory from which PAWSIM was launched.
   */
  slash = strrchr(fname,'/');
  if (slash == NULL)
  {
    strcpy(cfg->inputDir,".");
  }
  else
  {
    size_t len = (size_t) (slash - fname);
    if (len >= MAX_PARAMETER_FILENAME_LENGTH)
    {
      len = MAX_PARAMETER_FILENAME_LENGTH - 1;
    }
    memcpy(cfg->inputDir,fname,len);
    cfg->inputDir[len] = '\0';
  }

  while (fscanf(infile,"%255s",key) != EOF)
  {
    if (fscanf(infile,"%255s",value) == EOF)
    {
      if (errstrm != NULL)
      {
        fprintf(errstrm,"ERROR: Missing value for parameter %s\n",key);
      }
      fclose(infile);
      return false;
    }

    if (strcmp(key,"Nlay") == 0) cfg->Nlay = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"Nx") == 0) cfg->Nx = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"Ny") == 0) cfg->Ny = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"Nt") == 0) cfg->Nt = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"startIdx") == 0) cfg->startIdx = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"timeSteppingScheme") == 0) cfg->timeSteppingScheme = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"momentumScheme") == 0) cfg->momentumScheme = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"thicknessScheme") == 0) cfg->thicknessScheme = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"tracerScheme") == 0) cfg->tracerScheme = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"useTrad") == 0) cfg->useTrad = (atoi(value) != 0);
    else if (strcmp(key,"Lx") == 0) cfg->Lx = strtod(value,NULL);
    else if (strcmp(key,"Ly") == 0) cfg->Ly = strtod(value,NULL);
    else if (strcmp(key,"dt") == 0) cfg->dt = strtod(value,NULL);
    else if (strcmp(key,"tmin") == 0) cfg->tmin = strtod(value,NULL);
    else if (strcmp(key,"tmax") == 0) cfg->tmax = strtod(value,NULL);
    else if (strcmp(key,"savefrequency") == 0) cfg->savefrequency = strtod(value,NULL);
    else if (strcmp(key,"savefreqAvg") == 0) cfg->savefreqAvg = strtod(value,NULL);
    else if (strcmp(key,"savefreqUMom") == 0) cfg->savefreqUMom = strtod(value,NULL);
    else if (strcmp(key,"savefreqVMom") == 0) cfg->savefreqVMom = strtod(value,NULL);
    else if (strcmp(key,"savefreqThic") == 0) cfg->savefreqThic = strtod(value,NULL);
    else if (strcmp(key,"savefreqEnergy") == 0) cfg->savefreqEnergy = strtod(value,NULL);
    else if (strcmp(key,"savefreqTracer") == 0) cfg->savefreqTracer = strtod(value,NULL);
    else if (strcmp(key,"savefreqEZ") == 0) cfg->savefreqEZ = strtod(value,NULL);
    else if (strcmp(key,"KT00_sigma") == 0) cfg->KT00_sigma = strtod(value,NULL);
    else if (strcmp(key,"h0") == 0) cfg->h0 = strtod(value,NULL);
    else if (strcmp(key,"hsml") == 0) cfg->hsml = strtod(value,NULL);
    else if (strcmp(key,"hmin_surf") == 0) cfg->hmin_surf = strtod(value,NULL);
    else if (strcmp(key,"hbbl") == 0) cfg->hbbl = strtod(value,NULL);
    else if (strcmp(key,"hmin_bot") == 0) cfg->hmin_bot = strtod(value,NULL);
    else if (strcmp(key,"A2") == 0) cfg->A2 = strtod(value,NULL);
    else if (strcmp(key,"A4") == 0) cfg->A4 = strtod(value,NULL);
    else if (strcmp(key,"A2smag") == 0) cfg->A2smag = strtod(value,NULL);
    else if (strcmp(key,"A4smag") == 0) cfg->A4smag = strtod(value,NULL);
    else if (strcmp(key,"K2") == 0) cfg->K2 = strtod(value,NULL);
    else if (strcmp(key,"K4") == 0) cfg->K4 = strtod(value,NULL);
    else if (strcmp(key,"linDragCoeff") == 0) cfg->linDragCoeff = strtod(value,NULL);
    else if (strcmp(key,"quadDragCoeff") == 0) cfg->quadDragCoeff = strtod(value,NULL);
    else if (strcmp(key,"linDragSurf") == 0) cfg->linDragSurf = strtod(value,NULL);
    else if (strcmp(key,"quadDragSurf") == 0) cfg->quadDragSurf = strtod(value,NULL);
    else if (strcmp(key,"tol") == 0) cfg->pi_tol = strtod(value,NULL);
    else if (strcmp(key,"SOR_rp") == 0) cfg->SOR_rp = strtod(value,NULL);
    else if (strcmp(key,"SOR_rp_min") == 0) cfg->SOR_rp = strtod(value,NULL);
    else if (strcmp(key,"tauPeriod") == 0) cfg->tauPeriod = strtod(value,NULL);
    else if (strcmp(key,"wDiaPeriod") == 0) cfg->wDiaPeriod = strtod(value,NULL);
    else if (strcmp(key,"convDiff") == 0) cfg->convDiff = strtod(value,NULL);
    else if (strcmp(key,"omega_MG") == 0) cfg->omega_MG = strtod(value,NULL);
    else if (strcmp(key,"tauNrecs") == 0) cfg->tauNrecs = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"wDiaNrecs") == 0) cfg->wDiaNrecs = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"maxiters") == 0) cfg->maxiters = (uint) strtoul(value,NULL,10);
    else if (strcmp(key,"use_MG") == 0) cfg->use_MG = (atoi(value) != 0);
    else if (strcmp(key,"pressureSolver") == 0)
    {
      cfg->pressureSolver = (uint) strtoul(value,NULL,10);
      pressureSolverSpecified = true;
    }
    else if (strcmp(key,"use_fullMG") == 0) cfg->use_fullMG = (atoi(value) != 0);
    else if (strcmp(key,"pressureTiming") == 0) cfg->pressureTiming = (atoi(value) != 0);
    else if (strcmp(key,"writeInitDiagnostics") == 0) cfg->writeInitDiagnostics = (atoi(value) != 0);
    else if (strcmp(key,"restart") == 0) cfg->restart = (atoi(value) != 0);
    else if (strcmp(key,"useWallEW") == 0) cfg->useWallEW = (atoi(value) != 0);
    else if (strcmp(key,"useWallNS") == 0) cfg->useWallNS = (atoi(value) != 0);
    else if (strcmp(key,"useRL") == 0) cfg->useRL = (atoi(value) != 0);
    else if (strcmp(key,"oceanSurfDrag") == 0) cfg->oceanSurfDrag = (atoi(value) != 0);
    else if (strcmp(key,"windFeedback") == 0) cfg->windFeedback = (atoi(value) != 0);
    else if (strcmp(key,"useTracer") == 0) cfg->useTracer = (atoi(value) != 0);
    else if (strcmp(key,"useBuoyancy") == 0) cfg->useBuoyancy = (atoi(value) != 0);
    else if (strcmp(key,"useRandomForcing") == 0) cfg->useRandomForcing = (atoi(value) != 0);
    else if (strcmp(key,"hInitFile") == 0) copy_config_string(cfg->hInitFile,value);
    else if (strcmp(key,"uInitFile") == 0) copy_config_string(cfg->uInitFile,value);
    else if (strcmp(key,"vInitFile") == 0) copy_config_string(cfg->vInitFile,value);
    else if (strcmp(key,"bInitFile") == 0) copy_config_string(cfg->bInitFile,value);
    else if (strcmp(key,"hbFile") == 0) copy_config_string(cfg->hbFile,value);
    else if (strcmp(key,"hsFile") == 0) copy_config_string(cfg->hsFile,value);
    else if (strcmp(key,"OmegazFile") == 0) copy_config_string(cfg->OmegazFile,value);
    else if (strcmp(key,"gFile") == 0) copy_config_string(cfg->gFile,value);
    else if (strcmp(key,"tauxFile") == 0) copy_config_string(cfg->tauxFile,value);
    else if (strcmp(key,"tauyFile") == 0) copy_config_string(cfg->tauyFile,value);
    else if (strcmp(key,"uLidFile") == 0) copy_config_string(cfg->uLidFile,value);
    else if (strcmp(key,"vLidFile") == 0) copy_config_string(cfg->vLidFile,value);
    else if (strcmp(key,"FbaroXFile") == 0) copy_config_string(cfg->FbaroXFile,value);
    else if (strcmp(key,"FbaroYFile") == 0) copy_config_string(cfg->FbaroYFile,value);
    else if (strcmp(key,"uRelaxFile") == 0) copy_config_string(cfg->uRelaxFile,value);
    else if (strcmp(key,"vRelaxFile") == 0) copy_config_string(cfg->vRelaxFile,value);
    else if (strcmp(key,"hRelaxFile") == 0) copy_config_string(cfg->hRelaxFile,value);
    else if (strcmp(key,"eRelaxFile") == 0) copy_config_string(cfg->eRelaxFile,value);
    else if (strcmp(key,"bRelaxFile") == 0) copy_config_string(cfg->bRelaxFile,value);
    else if (strcmp(key,"uTimeFile") == 0) copy_config_string(cfg->uTimeFile,value);
    else if (strcmp(key,"vTimeFile") == 0) copy_config_string(cfg->vTimeFile,value);
    else if (strcmp(key,"hTimeFile") == 0) copy_config_string(cfg->hTimeFile,value);
    else if (strcmp(key,"eTimeFile") == 0) copy_config_string(cfg->eTimeFile,value);
    else if (strcmp(key,"bTimeFile") == 0) copy_config_string(cfg->bTimeFile,value);
    else if (strcmp(key,"wDiaFile") == 0) copy_config_string(cfg->wDiaFile,value);
    else if (strcmp(key,"diaDiffFile") == 0) copy_config_string(cfg->diaDiffFile,value);
    else if (strcmp(key,"diaViscUFile") == 0) copy_config_string(cfg->diaViscUFile,value);
    else if (strcmp(key,"diaViscVFile") == 0) copy_config_string(cfg->diaViscVFile,value);
    else if (strcmp(key,"bFluxFile") == 0) copy_config_string(cfg->bFluxFile,value);
  }

  fclose(infile);
  /*
   * Preserve compatibility with AWSIM input files. An explicit pressureSolver
   * wins; otherwise use_MG 0/1 maps to PAWSIM's SOR/gathered-MG choices.
   */
  if (!pressureSolverSpecified)
  {
    cfg->pressureSolver = cfg->use_MG ? PAWSIM_PRESSURE_SOLVER_MG_GATHERED
                                      : PAWSIM_PRESSURE_SOLVER_SOR;
  }
  else
  {
    cfg->use_MG = (cfg->pressureSolver != PAWSIM_PRESSURE_SOLVER_SOR);
  }
  cfg->useWind = (strlen(cfg->tauxFile) > 0) || (strlen(cfg->tauyFile) > 0);
  cfg->useFbaro = (strlen(cfg->FbaroXFile) > 0) || (strlen(cfg->FbaroYFile) > 0);
  /*
   * These booleans are derived from file presence, as in AWSIM. Users keep the
   * original setup-file style instead of passing feature flags on the command
   * line; command-line arguments remain input file and output directory only.
   */
  cfg->useRelax = (strlen(cfg->uTimeFile) > 0) || (strlen(cfg->vTimeFile) > 0)
               || (strlen(cfg->hTimeFile) > 0) || (strlen(cfg->eTimeFile) > 0)
               || (cfg->useTracer && (strlen(cfg->bTimeFile) > 0));
  cfg->useWDia = cfg->useRelax || (strlen(cfg->wDiaFile) > 0);
  cfg->useDiaDiff = (strlen(cfg->bFluxFile) > 0) || (strlen(cfg->diaDiffFile) > 0)
                 || (cfg->convDiff != 0);
  cfg->useDiaVisc = (strlen(cfg->diaViscUFile) > 0) || (strlen(cfg->diaViscVFile) > 0)
                 || (cfg->convDiff != 0);
  return true;
}

/*
 * Check that parsed inputs describe a supported PAWSIM integration. This is
 * intentionally stricter for unported AWSIM options than the parser itself.
 */
bool pawsim_config_validate (const pawsim_config * cfg, FILE * errstrm)
{
  if (cfg == NULL)
  {
    return false;
  }

  if ((cfg->Nlay == 0) || (cfg->Nx == 0) || (cfg->Ny == 0))
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Nlay, Nx and Ny must all be specified and positive\n");
    }
    return false;
  }

  if (cfg->Ly <= 0)
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Ly must be specified and positive\n");
    }
    return false;
  }

  /*
   * These constants come from defs.h. PAWSIM accepts AWSIM's spatial
   * schemes, but only the AB1/AB2/AB3 time integrators have been ported so far.
   */
  if ((cfg->timeSteppingScheme > TIMESTEPPING_AB3)
   || (cfg->momentumScheme > MOMENTUM_TW81)
   || (cfg->thicknessScheme > THICKNESS_KT00)
   || (cfg->tracerScheme > TRACER_KT00))
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Unknown or unsupported numerical scheme identifier\n");
    }
    return false;
  }

  if (cfg->h0 < 0)
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: h0 must be non-negative\n");
    }
    return false;
  }

  if ((cfg->hsml < 0) || (cfg->hmin_surf < 0) || (cfg->hbbl < 0)
   || (cfg->hmin_bot < 0) || (cfg->linDragCoeff < 0) || (cfg->quadDragCoeff < 0)
   || (cfg->linDragSurf < 0) || (cfg->quadDragSurf < 0)
   || (cfg->A2 < 0) || (cfg->A4 < 0) || (cfg->A2smag < 0) || (cfg->A4smag < 0)
   || (cfg->K2 < 0) || (cfg->K4 < 0)
   || (cfg->tauNrecs == 0) || (cfg->wDiaNrecs == 0) || (cfg->convDiff < 0))
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Invalid wind forcing parameters\n");
    }
    return false;
  }

  if ((cfg->pi_tol <= 0) || (cfg->SOR_rp <= 0) || (cfg->SOR_rp >= 2)
   || (cfg->omega_MG <= 0) || (cfg->maxiters == 0))
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Invalid pressure solver parameters\n");
    }
    return false;
  }

  if (cfg->pressureSolver > PAWSIM_PRESSURE_SOLVER_MG_DISTRIBUTED)
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: Unknown pressureSolver identifier\n");
    }
    return false;
  }

  if (cfg->useBuoyancy && !cfg->useTracer)
  {
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: useBuoyancy requires useTracer in PAWSIM\n");
    }
    return false;
  }

  if (cfg->useTrad)
  {
    /*
     * Parse-but-reject unsupported AWSIM machinery. This makes copied setup
     * files fail explicitly instead of quietly dropping physics/integration
     * choices that could alter the scientific problem.
     */
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: AWSIM useTrad time stepping is not ported in PAWSIM\n");
    }
    return false;
  }

  if (cfg->useRandomForcing)
  {
    /*
     * Random forcing needs a deliberate parallel RNG/FFT design. Until then,
     * prescribe deterministic forcing fields, including time-dependent
     * barotropic forcing once that reader is needed.
     */
    if (errstrm != NULL)
    {
      fprintf(errstrm,"ERROR: AWSIM random forcing is not ported in PAWSIM; prescribe deterministic forcing instead\n");
    }
    return false;
  }

  return true;
}
