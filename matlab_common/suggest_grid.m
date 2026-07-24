%%%
%%% suggest_grid.m
%%%
%%% Suggest PAWSIM grid dimensions and MPI rank layouts near a desired grid.
%%%
%%% The distributed multigrid path is happiest when each rank owns local tile
%%% dimensions that can be halved cleanly. For a rank layout Px x Py, the first
%%% distributed coarsening is locally nested when Nx is divisible by 2*Px and
%%% Ny is divisible by 2*Py. More factors of two in Nx/Px and Ny/Py generally
%%% mean more clean distributed multigrid levels before the gathered coarse
%%% tail is used.
%%%
%%% Usage:
%%%
%%%   s = suggest_grid(300,200,6)
%%%   s = suggest_grid(300,200,6,'max_delta',64,'nshow',12)
%%%   s = suggest_grid(300,200,6,'aspect_weight',0.25)
%%%
%%% Optional name/value arguments:
%%%
%%%   max_delta      Search +/- max_delta grid points around each dimension.
%%%                  Default: max(64,ceil(0.1*max(Nx_desired,Ny_desired))).
%%%   nshow          Number of suggestions to print/return. Default: 10.
%%%   aspect_weight  Penalty weight for changing aspect ratio. Default: 0.1.
%%%   mpi_dims       Restrict search to a specific [mpiNx mpiNy] layout.
%%%   valid_only     Return only layouts that can start distributed MG cleanly
%%%                  when such layouts exist. Default: true.
%%%
%%% Returns a struct array with fields:
%%%
%%%   Nx, Ny                 Suggested global grid size.
%%%   mpi_dims               [Px Py] MPI Cartesian rank layout.
%%%   local_nx, local_ny     Cells per rank for this layout.
%%%   clean_levels           Number of exact local 2:1 coarsenings.
%%%   first_coarsening_ok    True if PAWSIM distributed MG can start cleanly.
%%%   mpiNx, mpiNy           Input parameters to force this rank layout.
%%%   rel_grid_change        Relative grid-size change from the request.
%%%   score                  Lower is better.
%%%
function suggestions = suggest_grid (Nx_desired,Ny_desired,nranks,varargin)

  max_delta = max(64,ceil(0.1*max(Nx_desired,Ny_desired)));
  nshow = 10;
  aspect_weight = 0.1;
  mpi_dims = [];
  valid_only = true;

  if (mod(length(varargin),2) ~= 0)
    error('Optional arguments must be name/value pairs');
  end
  for n = 1:2:length(varargin)
    key = char(varargin{n});
    val = varargin{n+1};
    switch lower(key)
      case 'max_delta'
        max_delta = val;
      case 'nshow'
        nshow = val;
      case 'aspect_weight'
        aspect_weight = val;
      case 'mpi_dims'
        mpi_dims = val;
      case 'valid_only'
        valid_only = val;
      otherwise
        error(['Unknown option ',key]);
    end
  end

  if ((Nx_desired < 1) || (Ny_desired < 1) || (nranks < 1))
    error('Nx_desired, Ny_desired, and nranks must be positive');
  end

  if (isempty(mpi_dims))
    layouts = factor_layouts(nranks);
  else
    if ((numel(mpi_dims) ~= 2) || (prod(mpi_dims) ~= nranks))
      error('mpi_dims must be [mpiNx mpiNy] with product nranks');
    end
    layouts = reshape(mpi_dims,1,2);
  end
  cand = struct([]);
  ncand = 0;
  aspect0 = Nx_desired / Ny_desired;

  for l = 1:size(layouts,1)
    px = layouts(l,1);
    py = layouts(l,2);

    local_x_min = max(1,floor((Nx_desired-max_delta)/px));
    local_x_max = max(1,ceil((Nx_desired+max_delta)/px));
    local_y_min = max(1,floor((Ny_desired-max_delta)/py));
    local_y_max = max(1,ceil((Ny_desired+max_delta)/py));

    for lx = local_x_min:local_x_max
      Nx = lx*px;
      for ly = local_y_min:local_y_max
        Ny = ly*py;

        rel_area = abs(Nx*Ny - Nx_desired*Ny_desired) ...
                 / (Nx_desired*Ny_desired);
        rel_shape = abs(log((Nx/Ny)/aspect0));
        rel_dims = abs(Nx-Nx_desired)/Nx_desired ...
                 + abs(Ny-Ny_desired)/Ny_desired;
        levels = min(pow2_factor(lx),pow2_factor(ly));
        first_ok = ((mod(lx,2) == 0) && (mod(ly,2) == 0));

        %%% Lower score is better. Reward clean multigrid levels strongly,
        %%% but keep the suggested global grid close to the requested one.
        score = rel_dims + rel_area + aspect_weight*rel_shape ...
              - 0.15*levels;
        if (~first_ok)
          score = score + 10;
        end

        ncand = ncand + 1;
        cand(ncand).Nx = Nx;
        cand(ncand).Ny = Ny;
        cand(ncand).mpi_dims = [px py];
        cand(ncand).mpiNx = px;
        cand(ncand).mpiNy = py;
        cand(ncand).local_nx = lx;
        cand(ncand).local_ny = ly;
        cand(ncand).clean_levels = levels;
        cand(ncand).first_coarsening_ok = first_ok;
        cand(ncand).rel_grid_change = rel_dims;
        cand(ncand).score = score;
      end
    end
  end

  if (isempty(cand))
    suggestions = cand;
    return;
  end

  if (valid_only)
    valid = [cand.first_coarsening_ok];
    if (any(valid))
      cand = cand(valid);
    end
  end

  scores = [cand.score];
  [~,idx] = sort(scores);
  idx = idx(1:min(nshow,length(idx)));
  suggestions = cand(idx);

  if (nargout == 0)
    print_suggestions(suggestions);
    clear suggestions;
  end

end

function layouts = factor_layouts (nranks)

  layouts = [];
  for px = 1:nranks
    if (mod(nranks,px) == 0)
      py = nranks / px;
      layouts = [layouts; px py]; %#ok<AGROW>
    end
  end

end

function n = pow2_factor (m)

  n = 0;
  while ((m > 0) && (mod(m,2) == 0))
    n = n + 1;
    m = m / 2;
  end

end

function print_suggestions (s)

  fprintf(' rank grid   global grid   local tile   clean levels   rel change   input params\n');
  fprintf(' ---------   -----------   ----------   ------------   ----------   ------------\n');
  for n = 1:length(s)
    fprintf(' %3dx%-3d    %5dx%-5d   %5dx%-5d   %6d         %9.4f   mpiNx %d, mpiNy %d\n', ...
            s(n).mpi_dims(1),s(n).mpi_dims(2), ...
            s(n).Nx,s(n).Ny, ...
            s(n).local_nx,s(n).local_ny, ...
            s(n).clean_levels,s(n).rel_grid_change,s(n).mpiNx,s(n).mpiNy);
  end

end
