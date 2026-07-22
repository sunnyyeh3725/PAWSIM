/**
 * config.h
 *
 * Input configuration interface for PAWSIM.
 *
 */
#ifndef _PAWSIM_CONFIG_H_
#define _PAWSIM_CONFIG_H_

#include "defs.h"

/* PAWSIM extension to AWSIM's use_MG flag. */
#define PAWSIM_PRESSURE_SOLVER_SOR 0
#define PAWSIM_PRESSURE_SOLVER_MG_GATHERED 1
#define PAWSIM_PRESSURE_SOLVER_MG_DISTRIBUTED 2

typedef struct pawsim_config
{
  /* Grid, time stepping, and scheme identifiers copied from AWSIM inputs. */
  uint Nlay;
  uint Nx;
  uint Ny;
  uint Nt;
  uint startIdx;
  uint timeSteppingScheme;
  uint momentumScheme;
  uint thicknessScheme;
  uint tracerScheme;
  real Lx;
  real Ly;
  real dt;
  real tmin;
  real tmax;
  real savefrequency;
  real savefreqAvg;
  real savefreqUMom;
  real savefreqVMom;
  real savefreqThic;
  real savefreqEnergy;
  real savefreqTracer;
  real savefreqEZ;
  real KT00_sigma;
  real h0;
  real hsml;
  real hmin_surf;
  real hbbl;
  real hmin_bot;
  real A2;
  real A4;
  real A2smag;
  real A4smag;
  real K2;
  real K4;
  /* Forcing, drag, and pressure-solver controls. */
  real linDragCoeff;
  real quadDragCoeff;
  real linDragSurf;
  real quadDragSurf;
  real pi_tol;
  real SOR_rp;
  real tauPeriod;
  real wDiaPeriod;
  real convDiff;
  real omega_MG;
  uint tauNrecs;
  uint wDiaNrecs;
  uint maxiters;
  uint pressureSolver;
  bool useWind;
  bool useFbaro;
  bool useRelax;
  bool useWDia;
  bool useDiaDiff;
  bool useDiaVisc;
  bool use_MG;
  bool use_fullMG;
  bool pressureTiming;
  bool writeInitDiagnostics;
  bool restart;
  bool useWallEW;
  bool useWallNS;
  bool useRL;
  bool oceanSurfDrag;
  bool windFeedback;
  bool useTracer;
  bool useBuoyancy;
  /* Parsed only so unsupported AWSIM requests fail explicitly. */
  bool useRandomForcing;
  bool useTrad;
  /* inputDir is inferred from the parameter-file path for relative data files. */
  char inputDir[MAX_PARAMETER_FILENAME_LENGTH];
  char hInitFile[MAX_PARAMETER_FILENAME_LENGTH];
  char uInitFile[MAX_PARAMETER_FILENAME_LENGTH];
  char vInitFile[MAX_PARAMETER_FILENAME_LENGTH];
  char bInitFile[MAX_PARAMETER_FILENAME_LENGTH];
  char hbFile[MAX_PARAMETER_FILENAME_LENGTH];
  char hsFile[MAX_PARAMETER_FILENAME_LENGTH];
  char OmegazFile[MAX_PARAMETER_FILENAME_LENGTH];
  char gFile[MAX_PARAMETER_FILENAME_LENGTH];
  char tauxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char tauyFile[MAX_PARAMETER_FILENAME_LENGTH];
  char uLidFile[MAX_PARAMETER_FILENAME_LENGTH];
  char vLidFile[MAX_PARAMETER_FILENAME_LENGTH];
  char FbaroXFile[MAX_PARAMETER_FILENAME_LENGTH];
  char FbaroYFile[MAX_PARAMETER_FILENAME_LENGTH];
  char uRelaxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char vRelaxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char hRelaxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char eRelaxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char bRelaxFile[MAX_PARAMETER_FILENAME_LENGTH];
  char uTimeFile[MAX_PARAMETER_FILENAME_LENGTH];
  char vTimeFile[MAX_PARAMETER_FILENAME_LENGTH];
  char hTimeFile[MAX_PARAMETER_FILENAME_LENGTH];
  char eTimeFile[MAX_PARAMETER_FILENAME_LENGTH];
  char bTimeFile[MAX_PARAMETER_FILENAME_LENGTH];
  char wDiaFile[MAX_PARAMETER_FILENAME_LENGTH];
  char diaDiffFile[MAX_PARAMETER_FILENAME_LENGTH];
  char diaViscUFile[MAX_PARAMETER_FILENAME_LENGTH];
  char diaViscVFile[MAX_PARAMETER_FILENAME_LENGTH];
  char bFluxFile[MAX_PARAMETER_FILENAME_LENGTH];
}
pawsim_config;

/* Populate AWSIM-compatible defaults plus PAWSIM extension defaults. */
void pawsim_config_defaults (pawsim_config * cfg);

/* Read a simple AWSIM key/value parameter file. */
bool pawsim_config_read (const char * fname, pawsim_config * cfg, FILE * errstrm);

/* Check required parameters and supported numerical options. */
bool pawsim_config_validate (const pawsim_config * cfg, FILE * errstrm);

#endif
