function export_fixtures(out_dir)
% Export Octave reference fixtures for the Python port.
%
% Octave only. Shim sources live in shims/*.m.in and are copied into a
% temporary directory as .m files. Nothing named pagemtimes.m, eul2rotm.m,
% and so on is written into the repository, so MATLAB users can safely run
% addpath(genpath(pwd)).
%
% The two known bugs are patched on a temporary copy of fun/ before the
% reference methods run. bug_impact.mat also runs the unpatched copies.
% plane_TRO2016 is not exported: Octave has no lsqnonlin.

  if ~exist('OCTAVE_VERSION', 'builtin')
    error('export_fixtures is an Octave tool and must not be run in MATLAB.');
  end
  if nargin < 1 || isempty(out_dir)
    error('export_fixtures(out_dir) needs an output directory');
  end
  out_dir = char(out_dir);
  if ~exist(out_dir, 'dir')
    mkdir(out_dir);
  end

  global EXPORT_TMP EXPORT_DELETE
  global FIXTURE_DX FIXTURE_H
  global LINE_PROBE_NNZ LINE_PROBE_NCOLS
  global MEAS_PROBE_XR MEAS_PROBE_XL
  EXPORT_TMP = '';
  EXPORT_DELETE = false;
  cleaner = onCleanup(@cleanup_export_tmp); %#ok<NASGU>
  page_screen_output(0);

  here = fileparts(mfilename('fullpath'));
  repo = canonicalize_file_name(fullfile(here, '..', '..', '..'));
  fprintf('export_fixtures\n  repo %s\n  out  %s\n', repo, out_dir);

  work = prepare_trees(repo, fullfile(here, 'shims'));
  EXPORT_TMP = work.tmp;
  activate(work.fun_patched, work.shim_dir, '');
  assert_resolved(work.shim_dir);
  assert_shims();
  fprintf('shims ok\n');

  g = base_geometry();
  save_struct(fullfile(out_dir, 'shim_reference.mat'), shim_cases());
  fprintf('shim_reference.mat\n');

  activate(work.fun_patched, work.shim_dir, '');
  export_simulators(out_dir, g);
  export_methods(out_dir, g);
  export_bug_impact(out_dir, g, work);
  fprintf('fixtures written\n');
  EXPORT_DELETE = true;
end

function cleanup_export_tmp()
  global EXPORT_TMP EXPORT_DELETE
  if isempty(EXPORT_TMP)
    return
  end
  if EXPORT_DELETE
    try
      rmdir(EXPORT_TMP, 's');
    catch err
      fprintf(2, 'could not remove %s (%s)\n', EXPORT_TMP, err.message);
    end
  else
    fprintf(2, 'export failed; temp dir kept at %s\n', EXPORT_TMP);
  end
end

function work = prepare_trees(repo, shim_src)
  tmp = tempname;
  mkdir(tmp);
  work.tmp = tmp;
  work.shim_dir = fullfile(tmp, 'shims');
  work.fun_patched = fullfile(tmp, 'fun_patched');
  work.fun_original = fullfile(tmp, 'fun_original');
  mkdir(work.shim_dir);
  names = {'pagemtimes', 'pagemrdivide', 'eul2rotm', 'axang2rotm', ...
    'rotm2axang', 'quat2rotm', 'rotm2eul'};
  for i = 1:numel(names)
    src = fullfile(shim_src, [names{i} '.m.in']);
    dst = fullfile(work.shim_dir, [names{i} '.m']);
    if ~exist(src, 'file')
      error('missing shim source %s', src);
    end
    copyfile(src, dst);
  end
  copyfile(fullfile(repo, 'fun'), work.fun_patched);
  copyfile(fullfile(repo, 'fun'), work.fun_original);
  apply_bugfixes(work.fun_patched);
  install_probes(work.fun_patched);
  install_probes(work.fun_original);
  install_dx_capture(work.fun_patched);
  buggy = 'ub(:,(1:(mt-1))+(mt-1)) = RH2Wn(:,:,i)*Rsa*us + ut;';
  orig = fileread(fullfile(work.fun_original, 'calib', 'line_RCIM2022.m'));
  if isempty(strfind(orig, buggy)) %#ok<STREMP>
    error('unpatched tree is missing the line_RCIM2022 bug');
  end
end

function apply_bugfixes(fun_dir)
  line_old = '    ub(:,(1:(mt-1))+(mt-1)) = RH2Wn(:,:,i)*Rsa*us + ut;';
  line_new = '    ub(:,(1:(mt-1))+(mt-1)*(i-1)) = RH2Wn(:,:,i)*Rsa*us + ut;';
  meas_old = '    xr = min(pcOx(it)+rO(it),zMax) - 1e-4;';
  meas_new = '    xr = min(pcOx(it)+rO(it),xMax) - 1e-4;';
  line_path = fullfile(fun_dir, 'calib', 'line_RCIM2022.m');
  meas_path = fullfile(fun_dir, 'sim', 'simMEAS2023.m');
  must_replace(line_path, line_old, line_new);
  must_replace(meas_path, meas_old, meas_new);
  if ~isempty(strfind(fileread(line_path), line_old)) %#ok<STREMP>
    error('line_RCIM2022 bug remains after patch');
  end
  if ~isempty(strfind(fileread(meas_path), meas_old)) %#ok<STREMP>
    error('simMEAS2023 bug remains after patch');
  end
end

function install_probes(fun_dir)
  svd_old = '% SVD for ub*';
  svd_new = ['global LINE_PROBE_NNZ LINE_PROBE_NCOLS;', newline, ...
    'LINE_PROBE_NNZ = sum(any(abs(ub) > 0, 1));', newline, ...
    'LINE_PROBE_NCOLS = size(ub, 2);', newline, ...
    '% SVD for ub*'];
  img_old = '    % image points';
  img_new = ['    if ~exist(''probe_xr'', ''var'')', newline, ...
    '      probe_xr = zeros(1, mt);', newline, ...
    '      probe_xl = zeros(1, mt);', newline, ...
    '    end', newline, ...
    '    probe_xr(it) = xr;', newline, ...
    '    probe_xl(it) = xl;', newline, ...
    '    % image points'];
  tail_old = 'tH2Wn = reshape(tH2W,3,mt,mR) + pagemtimes(Lt,randn(3,mt,mR));    % 3 x mt x m';
  tail_new = [tail_old, newline, ...
    'global MEAS_PROBE_XR MEAS_PROBE_XL;', newline, ...
    'MEAS_PROBE_XR = probe_xr;', newline, ...
    'MEAS_PROBE_XL = probe_xl;'];
  must_replace(fullfile(fun_dir, 'calib', 'line_RCIM2022.m'), svd_old, svd_new);
  must_replace(fullfile(fun_dir, 'sim', 'simMEAS2023.m'), img_old, img_new);
  must_replace(fullfile(fun_dir, 'sim', 'simMEAS2023.m'), tail_old, tail_new);
end

function install_dx_capture(fun_dir)
  dx_old = '        dx = H\g;';
  dx_new = ['        dx = H\g;', newline, ...
    '        global FIXTURE_DX FIXTURE_H;', newline, ...
    '        if isempty(FIXTURE_DX), FIXTURE_DX = dx; FIXTURE_H = H; end'];
  files = {'plane_Jnt_GH.m', 'plane_Alt_GH.m', 'plane_Jnt_GH_decoupled.m'};
  for i = 1:numel(files)
    must_replace(fullfile(fun_dir, 'calib', files{i}), dx_old, dx_new);
  end
end

function must_replace(path, old, new)
  txt = fileread(path);
  hits = strfind(txt, old);
  if numel(hits) ~= 1
    error('expected one occurrence in %s, found %d', path, numel(hits));
  end
  fid = fopen(path, 'w');
  if fid < 0
    error('cannot write %s', path);
  end
  fwrite(fid, strrep(txt, old, new));
  fclose(fid);
end

function activate(fun_dir, shim_dir, other_fun)
  if ~isempty(other_fun)
    remove_tree(other_fun);
  end
  addpath(genpath(fun_dir));
  addpath(shim_dir, '-begin');
  clear -f line_RCIM2022 simMEAS2023
  rehash
end

function remove_tree(root)
  entries = strsplit(genpath(root), pathsep);
  active = strsplit(path(), pathsep);
  for i = 1:numel(entries)
    entry = entries{i};
    if ~isempty(entry) && any(strcmp(active, entry))
      rmpath(entry);
    end
  end
end

function assert_resolved(shim_dir)
  names = {'pagemtimes', 'pagemrdivide', 'eul2rotm', 'axang2rotm', ...
    'rotm2axang', 'quat2rotm', 'rotm2eul'};
  for i = 1:numel(names)
    found = which(names{i});
    if isempty(strfind(found, shim_dir)) %#ok<STREMP>
      error('%s resolved to %s, not the temporary shim', names{i}, found);
    end
  end
end

function g = base_geometry()
  g.xMax = 5;
  g.zMax = 2;
  g.tO2H = [12; -7; 30];
  g.RO2H = axang2rotm([1 2 3 pi / 5]);
  g.pL = [0; 0; 0];
  v = [0.2; -0.3; 0.9];
  g.vL = v / norm(v);
  g.lL = 50;
  g.pP = g.pL;
  g.nP = g.vL;
  g.lP = 50;
  g.nPlane = 3;
  g.pPs = repmat(g.pL, 1, 3);
  g.nPs = [1 0 0; 0 1 0; 0 0 1];
  g.lPs = repmat(g.lP, 1, 3);
  g.pS = [0; 0; 40];
  g.rS = 7.5;
  g.alphaMax = 45 * pi / 180;
  g.betaMax = 45 * pi / 180;
  g.mAlpha = 4;
  g.mBeta = 4;
  % round(4/sqrt(3)) is 2, so 4 shots per plane and plane_IROS2015 does not
  % recover. 6 gives round(6/sqrt(3)) = 3 and 9 shots per plane.
  g.mIros = 6;
  g.mt = 3;
  g.tol = 1e-5;
  g.iImax = 30;
  g.iOmax = 5;
  g.tO2H0 = g.tO2H + [0.4; -0.3; 0.2];
  g.RO2H0 = g.RO2H * axang2rotm([0.3 -0.2 0.1 0.05]);
  n0 = g.nP + [0.02; -0.01; 0];
  g.n0 = n0 / norm(n0);
  g.d0 = g.pP.' * g.n0;
  g.sigma0 = [0.1 0.01];
end

function [Sigmat, SigmaR] = tiny_pose_cov()
  Sigmat = 1e-18 * eye(3);
  SigmaR = 1e-18 * eye(3);
end

function export_simulators(out_dir, g)
  [Sigmat, SigmaR] = tiny_pose_cov();
  sigmaO = 0;

  rng(42);
  [tH2Wn, RH2Wn, pOn, vOn, scanO] = simPlane( ...
    g.pP, g.nP, g.lP, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, g.mAlpha, g.mBeta);
  s = struct('schema', 1, 'pP', g.pP, 'nP', g.nP, 'lP', g.lP, ...
    'xMax', g.xMax, 'zMax', g.zMax, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'Sigmat', Sigmat, 'SigmaR', SigmaR, 'sigmaO', sigmaO, ...
    'alphaMax', g.alphaMax, 'betaMax', g.betaMax, ...
    'mAlpha', g.mAlpha, 'mBeta', g.mBeta, ...
    'tH2Wn', tH2Wn, 'RH2Wn', RH2Wn, 'pOn', pOn, 'vOn', vOn, 'scanO', scanO);
  save_struct(fullfile(out_dir, 'sim_plane.mat'), s);
  fprintf('sim_plane.mat\n');

  rng(43);
  [tH2Wn, RH2Wn, pOn] = simIROS2015( ...
    g.nPlane, g.pPs, g.nPs, g.lPs, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, g.mIros, g.mIros);
  s = struct('schema', 1, 'nPlane', g.nPlane, 'pPs', g.pPs, 'nPs', g.nPs, ...
    'lPs', g.lPs, 'xMax', g.xMax, 'zMax', g.zMax, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'Sigmat', Sigmat, 'SigmaR', SigmaR, 'sigmaO', sigmaO, ...
    'alphaMax', g.alphaMax, 'betaMax', g.betaMax, ...
    'mAlpha', g.mIros, 'mBeta', g.mIros, ...
    'tH2Wn', tH2Wn, 'RH2Wn', RH2Wn, 'pOn', pOn);
  save_struct(fullfile(out_dir, 'sim_iros2015.mat'), s);
  fprintf('sim_iros2015.mat\n');

  rng(44);
  m = 8;
  [tH2Wn, RH2Wn, lOn] = simRCIM2020( ...
    g.pP, g.nP, g.lP, g.xMax, g.zMax, g.tO2H, g.RO2H, 0, 0, 0, m);
  s = struct('schema', 1, 'pP', g.pP, 'nP', g.nP, 'lP', g.lP, ...
    'xMax', g.xMax, 'zMax', g.zMax, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'et', 0, 'eR', 0, 'eO', 0, 'm', m, ...
    'tH2Wn', tH2Wn, 'RH2Wn', RH2Wn, 'lOn', lOn);
  save_struct(fullfile(out_dir, 'sim_rcim2020.mat'), s);
  fprintf('sim_rcim2020.mat\n');

  rng(45);
  [tH2Wn, RH2Wn, pOn] = simRCIM2022( ...
    g.pL, g.vL, g.lL, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  s = struct('schema', 1, 'pL', g.pL, 'vL', g.vL, 'lL', g.lL, ...
    'xMax', g.xMax, 'zMax', g.zMax, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'Sigmat', Sigmat, 'SigmaR', SigmaR, 'sigmaO', sigmaO, ...
    'alphaMax', g.alphaMax, 'betaMax', g.betaMax, ...
    'mAlpha', 3, 'mBeta', 3, 'mt', g.mt, ...
    'tH2Wn', tH2Wn, 'RH2Wn', RH2Wn, 'pOn', pOn);
  save_struct(fullfile(out_dir, 'sim_rcim2022.mat'), s);
  fprintf('sim_rcim2022.mat\n');

  rng(46);
  [tH2Wn, RH2Wn, pcOe] = simMEAS2023( ...
    g.pS, g.rS, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  s = struct('schema', 1, 'pS', g.pS, 'rS', g.rS, ...
    'xMax', g.xMax, 'zMax', g.zMax, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'Sigmat', Sigmat, 'SigmaR', SigmaR, 'sigmaO', sigmaO, ...
    'alphaMax', g.alphaMax, 'betaMax', g.betaMax, ...
    'mAlpha', 3, 'mBeta', 3, 'mt', g.mt, ...
    'tH2Wn', tH2Wn, 'RH2Wn', RH2Wn, 'pcOe', pcOe);
  save_struct(fullfile(out_dir, 'sim_meas2023.mat'), s);
  fprintf('sim_meas2023.mat\n');
end

function export_methods(out_dir, g)
  [Sigmat, SigmaR] = tiny_pose_cov();
  sigmaO = 0;
  id = tic;
  rng(42);
  [tH, RH, pOn, vOn] = simPlane( ...
    g.pP, g.nP, g.lP, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, g.mAlpha, g.mBeta);
  fprintf('shared simPlane %.2fs\n', toc(id));

  run_gh(out_dir, 'plane_jnt_gh.mat', 'jnt', g, tH, RH, pOn, vOn);
  run_gh(out_dir, 'plane_alt_gh.mat', 'alt', g, tH, RH, pOn, vOn);
  run_gh(out_dir, 'plane_jnt_gh_decoupled.mat', 'decoupled', g, tH, RH, pOn, vOn);

  id = tic;
  [te, Re, ne] = plane_RCIM2020(g.tO2H0, g.RO2H0, tH, RH, pOn, g.tol);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  s = method_base(g, tH, RH, pOn, vOn, te, Re, et, eR);
  s.ne = ne;
  save_struct(fullfile(out_dir, 'plane_rcim2020.mat'), s);
  fprintf('plane_rcim2020 err_t %.3g err_R %.3g deg  %.2fs\n', et, eR, toc(id));

  id = tic;
  rng(43);
  [tHi, RHi, pOi] = simIROS2015( ...
    g.nPlane, g.pPs, g.nPs, g.lPs, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, g.mIros, g.mIros);
  [te, Re, ne] = plane_IROS2015(g.tO2H0, g.RO2H0, tHi, RHi, pOi, g.tol);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  s = struct('schema', 1, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'tO2H0', g.tO2H0, 'RO2H0', g.RO2H0, 'tol', g.tol, ...
    'tH2Wn', tHi, 'RH2Wn', RHi, 'pOn', pOi, ...
    'tO2He', te, 'RO2He', Re, 'ne', ne, 'err_t', et, 'err_R_deg', eR);
  save_struct(fullfile(out_dir, 'plane_iros2015.mat'), s);
  fprintf('plane_iros2015 err_t %.3g err_R %.3g deg  %.2fs\n', et, eR, toc(id));

  id = tic;
  rng(45);
  [tHl, RHl, pOl] = simRCIM2022( ...
    g.pL, g.vL, g.lL, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  [te, Re] = line_RCIM2022(tHl, RHl, pOl);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  s = struct('schema', 1, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'tH2Wn', tHl, 'RH2Wn', RHl, 'pOn', pOl, ...
    'tO2He', te, 'RO2He', Re, 'err_t', et, 'err_R_deg', eR);
  save_struct(fullfile(out_dir, 'line_rcim2022.mat'), s);
  fprintf('line_rcim2022 err_t %.3g err_R %.3g deg  %.2fs\n', et, eR, toc(id));

  id = tic;
  rng(46);
  [tHs, RHs, pc] = simMEAS2023( ...
    g.pS, g.rS, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  [te, Re] = sphere_MEAS2023(tHs, RHs, pc);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  s = struct('schema', 1, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'tH2Wn', tHs, 'RH2Wn', RHs, 'pcOe', pc, ...
    'tO2He', te, 'RO2He', Re, 'err_t', et, 'err_R_deg', eR);
  save_struct(fullfile(out_dir, 'sphere_meas2023.mat'), s);
  fprintf('sphere_meas2023 err_t %.3g err_R %.3g deg  %.2fs\n', et, eR, toc(id));
end

function run_gh(out_dir, filename, kind, g, tH, RH, pOn, vOn)
  global FIXTURE_DX FIXTURE_H
  FIXTURE_DX = [];
  FIXTURE_H = [];
  id = tic;
  if strcmp(kind, 'jnt')
    [te, Re, ne, de, tWe, RWe, sigmae] = plane_Jnt_GH( ...
      g.tO2H0, g.RO2H0, g.n0, g.d0, g.sigma0, tH, RH, pOn, vOn, ...
      g.iImax, g.iOmax, g.tol);
  elseif strcmp(kind, 'alt')
    [te, Re, ne, de, tWe, RWe, sigmae] = plane_Alt_GH( ...
      g.tO2H0, g.RO2H0, g.n0, g.d0, g.sigma0, tH, RH, pOn, vOn, ...
      g.iImax, g.iOmax, g.tol);
  elseif strcmp(kind, 'decoupled')
    [te, Re, ne, de, tWe, RWe, sigmae] = plane_Jnt_GH_decoupled( ...
      g.tO2H0, g.RO2H0, g.n0, g.d0, g.sigma0, tH, RH, pOn, vOn, ...
      g.iImax, g.iOmax, g.tol);
  else
    error('unknown GH kind %s', kind);
  end
  if isempty(FIXTURE_DX) || isempty(FIXTURE_H)
    error('%s did not capture the first dx and H', kind);
  end
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  s = method_base(g, tH, RH, pOn, vOn, te, Re, et, eR);
  s.ne = ne;
  s.de = de;
  s.tH2We = tWe;
  s.RH2We = RWe;
  s.sigmae = sigmae;
  s.dx0 = FIXTURE_DX;
  s.H0 = FIXTURE_H;
  save_struct(fullfile(out_dir, filename), s);
  fprintf('%s err_t %.3g err_R %.3g deg  dx0 %s  %.2fs\n', ...
    filename, et, eR, mat2str(size(FIXTURE_DX)), toc(id));
end

function s = method_base(g, tH, RH, pOn, vOn, te, Re, et, eR)
  s = struct('schema', 1, 'tO2H', g.tO2H, 'RO2H', g.RO2H, ...
    'tO2H0', g.tO2H0, 'RO2H0', g.RO2H0, 'n0', g.n0, 'd0', g.d0, ...
    'sigma0', g.sigma0, 'iImax', g.iImax, 'iOmax', g.iOmax, 'tol', g.tol, ...
    'tH2Wn', tH, 'RH2Wn', RH, 'pOn', pOn, 'vOn', vOn, ...
    'tO2He', te, 'RO2He', Re, 'err_t', et, 'err_R_deg', eR);
end

function export_bug_impact(out_dir, g, work)
  s = line_impact(g, work);
  m = meas_impact(g, work);
  names = fieldnames(m);
  for i = 1:numel(names)
    s.(names{i}) = m.(names{i});
  end
  save_struct(fullfile(out_dir, 'bug_impact.mat'), s);
  fprintf('bug_impact.mat\n');
end

function s = line_impact(g, work)
  global LINE_PROBE_NNZ LINE_PROBE_NCOLS
  levels = [0; 0.1; 0.5; 1];
  nL = numel(levels);
  nR = 6;
  mA = 3;
  mB = 3;
  mt = 3;
  et_p = zeros(nR, nL);
  et_o = zeros(nR, nL);
  eR_p = zeros(nR, nL);
  eR_o = zeros(nR, nL);
  nnz_p = zeros(nR, nL);
  nnz_o = zeros(nR, nL);
  ncols = mA * mB * (mt - 1);
  for iL = 1:nL
    sig = levels(iL);
    [Sigmat, SigmaR, sigmaO] = level_cov(sig, 0);
    for iR = 1:nR
      rng(1000 + 20 * iL + iR);
      activate(work.fun_patched, work.shim_dir, work.fun_original);
      [tH, RH, pO] = simRCIM2022( ...
        g.pL, g.vL, g.lL, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
        Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, mA, mB, mt);
      [et_p(iR, iL), eR_p(iR, iL), nnz_p(iR, iL), ncols] = line_once( ...
        work.fun_patched, work.shim_dir, work.fun_original, g, tH, RH, pO);
      [et_o(iR, iL), eR_o(iR, iL), nnz_o(iR, iL), ncols] = line_once( ...
        work.fun_original, work.shim_dir, work.fun_patched, g, tH, RH, pO);
    end
    fprintf('line sig %.2f  median |dt| patched %.4g original %.4g  nnz %g/%g\n', ...
      sig, median(et_p(:, iL)), median(et_o(:, iL)), ...
      median(nnz_o(:, iL)), median(nnz_p(:, iL)));
  end
  if isempty(LINE_PROBE_NCOLS)
    error('line probe did not run');
  end
  s = struct('schema', 1, 'line_m', mA * mB, 'line_mt', mt, ...
    'line_sigmat', levels, 'line_ncols', ncols, ...
    'line_err_t_patched', et_p, 'line_err_t_original', et_o, ...
    'line_err_R_patched', eR_p, 'line_err_R_original', eR_o, ...
    'line_nnz_patched', nnz_p, 'line_nnz_original', nnz_o);
end

function [et, eR, nnz, ncols] = line_once(fun_dir, shim_dir, other_fun, g, tH, RH, pO)
  global LINE_PROBE_NNZ LINE_PROBE_NCOLS
  activate(fun_dir, shim_dir, other_fun);
  LINE_PROBE_NNZ = [];
  LINE_PROBE_NCOLS = [];
  [te, Re] = line_RCIM2022(tH, RH, pO);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
  if isempty(LINE_PROBE_NNZ)
    error('line probe missing nnz');
  end
  nnz = LINE_PROBE_NNZ;
  ncols = LINE_PROBE_NCOLS;
end

function s = meas_impact(g, work)
  [xr_p, xl_p] = meas_borders(work.fun_patched, work.shim_dir, work.fun_original, g, 7);
  [xr_o, xl_o] = meas_borders(work.fun_original, work.shim_dir, work.fun_patched, g, 7);
  fprintf('meas xr patched max %.6f original max %.6f (xMax %g zMax %g)\n', ...
    max(xr_p), max(xr_o), g.xMax, g.zMax);
  nR = 6;
  sig = 0.2;
  sigmaO = 0.05;
  et_p = zeros(nR, 1);
  et_o = zeros(nR, 1);
  eR_p = zeros(nR, 1);
  eR_o = zeros(nR, 1);
  for iR = 1:nR
    [et_p(iR), eR_p(iR)] = meas_once( ...
      work.fun_patched, work.shim_dir, work.fun_original, g, 3000 + iR, sig, sigmaO);
    [et_o(iR), eR_o(iR)] = meas_once( ...
      work.fun_original, work.shim_dir, work.fun_patched, g, 3000 + iR, sig, sigmaO);
  end
  fprintf('meas noisy median |dt| patched %.4g original %.4g\n', median(et_p), median(et_o));
  s = struct('meas_xMax', g.xMax, 'meas_zMax', g.zMax, 'meas_rS', g.rS, ...
    'meas_mt', g.mt, 'meas_xr_patched', xr_p, 'meas_xl_patched', xl_p, ...
    'meas_xr_original', xr_o, 'meas_xl_original', xl_o, ...
    'meas_sigmat', sig, 'meas_sigmaO', sigmaO, ...
    'meas_err_t_patched', et_p, 'meas_err_t_original', et_o, ...
    'meas_err_R_patched', eR_p, 'meas_err_R_original', eR_o);
end

function [xr, xl] = meas_borders(fun_dir, shim_dir, other_fun, g, seed)
  global MEAS_PROBE_XR MEAS_PROBE_XL
  activate(fun_dir, shim_dir, other_fun);
  [Sigmat, SigmaR, sigmaO] = level_cov(0, 0);
  rng(seed);
  MEAS_PROBE_XR = [];
  MEAS_PROBE_XL = [];
  simMEAS2023(g.pS, g.rS, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  if isempty(MEAS_PROBE_XR)
    error('simMEAS2023 probe missing');
  end
  xr = MEAS_PROBE_XR;
  xl = MEAS_PROBE_XL;
end

function [et, eR] = meas_once(fun_dir, shim_dir, other_fun, g, seed, sig, sigmaO)
  activate(fun_dir, shim_dir, other_fun);
  [Sigmat, SigmaR, sigmaO] = level_cov(sig, sigmaO);
  rng(seed);
  [tH, RH, pc] = simMEAS2023(g.pS, g.rS, g.xMax, g.zMax, g.tO2H, g.RO2H, ...
    Sigmat, SigmaR, sigmaO, g.alphaMax, g.betaMax, 3, 3, g.mt);
  [te, Re] = sphere_MEAS2023(tH, RH, pc);
  [et, eR] = calibrationError(g.tO2H, g.RO2H, te, Re);
end

function [Sigmat, SigmaR, sigmaO] = level_cov(sig, sigmaO)
  if sig == 0
    Sigmat = 1e-18 * eye(3);
    SigmaR = 1e-18 * eye(3);
  else
    Sigmat = (sig ^ 2) * eye(3);
    SigmaR = ((sig * pi / 180) ^ 2) * eye(3);
  end
end

function s = shim_cases()
  eul = [ ...
    0.2 -0.4 0.5; ...
    0.0 1.5707963267948966 0.3; ...
    1.0 0.0 -1.0; ...
    -0.7 0.4 1.1; ...
    0.3 -1.2 0.8];
  ax = [ ...
    1 2 3 0.4; ...
    0 0 1 0; ...
    1 0 0 pi; ...
    -2 0.5 0.25 1.2; ...
    0 1 0 pi / 2];
  q = [ ...
    1 0 0 0; ...
    0.5 0.5 0.5 0.5; ...
    2 0 0 0; ...
    0.1 -0.2 0.3 -0.4; ...
    0 1 0 0];
  A = zeros(3, 3, 4);
  B = zeros(3, 1, 4);
  for k = 1:4
    A(:, :, k) = [k 0 0; 0 1 0; 1 2 k];
    B(:, :, k) = [k; -1; 0.5];
  end
  S = reshape([2 -1 0.5 4], 1, 1, 4);
  M = [1 2; 0 -1; 3 0.5];
  V = zeros(2, 1, 4);
  for k = 1:4
    V(:, :, k) = [k; 0.25 * k];
  end
  At = zeros(3, 2, 4);
  Bt = zeros(3, 3, 4);
  for k = 1:4
    At(:, :, k) = [k 1; 0 2; -1 k];
    Bt(:, :, k) = eye(3) * k;
    Bt(1, 2, k) = 0.3;
  end
  A4 = zeros(3, 3, 2, 2);
  B4 = zeros(3, 1, 2, 2);
  for i = 1:2
    for j = 1:2
      A4(:, :, i, j) = eye(3) * (i + 2 * j);
      A4(1, 3, i, j) = 0.2 * i;
      B4(:, :, i, j) = [i; j; 1];
    end
  end
  SR = reshape([0.5 -2 3], 1, 1, 3);
  AR = A(:, :, 1:3);
  Ea = zeros(5, 2, 3);
  Eb = zeros(2, 2, 3);
  for k = 1:3
    Eb(:, :, k) = [2 0.1; 0.2 1.5] * (0.5 + k);
    Ea(:, :, k) = reshape(1:10, 5, 2) + k;
  end
  s = struct();
  s.schema = 1;
  s.eul_zyx_in = eul;
  s.eul_zyx_R = eul2rotm(eul, 'ZYX');
  s.eul_zxy_in = eul;
  s.eul_zxy_R = eul2rotm(eul, 'ZXY');
  s.axang_in = ax;
  s.axang_R = axang2rotm(ax);
  s.rotm_in = s.axang_R;
  s.rotm_axang = rotm2axang(s.axang_R);
  s.rotm_eul_zyx = rotm2eul(s.axang_R, 'ZYX');
  s.rotm_eul_zxy = rotm2eul(s.axang_R, 'ZXY');
  s.quat_wxyz = q;
  s.quat_R = quat2rotm(q);
  s.pm_a_plain = A;
  s.pm_b_plain = B;
  s.pm_c_plain = pagemtimes(A, B);
  s.pm_a_scalar = S;
  s.pm_b_scalar = A;
  s.pm_c_scalar = pagemtimes(S, A);
  s.pm_a_bcast = M;
  s.pm_b_bcast = V;
  s.pm_c_bcast = pagemtimes(M, V);
  s.pm_a_trans = At;
  s.pm_b_trans = Bt;
  s.pm_c_trans = pagemtimes(At, 'transpose', Bt, 'none');
  s.pm_a_4d = A4;
  s.pm_b_4d = B4;
  s.pm_c_4d = pagemtimes(A4, B4);
  s.pm_a_rscalar = AR;
  s.pm_b_rscalar = SR;
  s.pm_c_rscalar = pagemtimes(AR, SR);
  s.pr_a = Ea;
  s.pr_b = Eb;
  s.pr_c = pagemrdivide(Ea, Eb);
  s.pr_a_bcast = Ea;
  s.pr_b_bcast = Eb(:, :, 1);
  s.pr_c_bcast = pagemrdivide(Ea, Eb(:, :, 1));
end

function assert_shims()
  A = reshape(1:12, 2, 2, 3);
  B = reshape(1:6, 2, 1, 3);
  C = pagemtimes(A, B);
  for k = 1:3
    assert_close(C(:, :, k), A(:, :, k) * B(:, :, k), 'plain pages');
  end
  S = reshape([2 3 4], 1, 1, 3);
  C = pagemtimes(S, A);
  for k = 1:3
    assert_close(C(:, :, k), S(k) * A(:, :, k), 'left scalar page');
  end
  C = pagemtimes(A, S);
  for k = 1:3
    assert_close(C(:, :, k), A(:, :, k) * S(k), 'right scalar page');
  end
  M = [1 2; 3 4; 5 6];
  V = reshape(1:8, 2, 1, 4);
  C = pagemtimes(M, V);
  for k = 1:4
    assert_close(C(:, :, k), M * V(:, :, k), 'broadcast matrix');
  end
  At = reshape(1:24, 3, 2, 4);
  Bt = reshape(1:36, 3, 3, 4);
  C = pagemtimes(At, 'transpose', Bt, 'none');
  for k = 1:4
    assert_close(C(:, :, k), At(:, :, k).' * Bt(:, :, k), 'transpose');
  end
  RH = reshape(1:45, 3, 3, 5);
  s = reshape([0.5 -1 2 3 -0.25], 1, 1, 5);
  C = pagemtimes(RH, s);
  neT = [0.2 -0.4 0.8];
  block = [C C RH];
  D = pagemtimes(neT, block);
  for k = 1:5
    assert_close(C(:, :, k), RH(:, :, k) * s(k), 'rcim scalar');
    assert_close(D(:, :, k), neT * block(:, :, k), 'rcim row');
  end
  rng(1);
  A4 = randn(3, 3, 2, 2);
  B4 = randn(3, 4, 2, 2);
  C4 = pagemtimes(A4, B4);
  for i = 1:2
    for j = 1:2
      assert_close(C4(:, :, i, j), A4(:, :, i, j) * B4(:, :, i, j), '4d');
    end
  end
  v = [1; 2; 3] * 50;
  r = rand(1, 1, 3, 2);
  Cv = pagemtimes(v, r);
  if ~isequal(size(Cv), [3 1 3 2])
    error('shim self-check failed: 4d scalar size %s', mat2str(size(Cv)));
  end
  E = zeros(2, 2, 4);
  AA = randn(2, 6, 4);
  for k = 1:4
    E(:, :, k) = [2 0.1; 0.1 3];
  end
  X = pagemrdivide(pagetranspose(AA), E);
  Atp = pagetranspose(AA);
  for k = 1:4
    assert_close(X(:, :, k) * E(:, :, k), Atp(:, :, k), 'rdivide');
  end
  e = [0.2 -0.4 0.5];
  R = eul2rotm(e, 'ZYX');
  e2 = rotm2eul(R, 'ZYX');
  assert_close(eul2rotm(e2, 'ZYX'), R, 'eul roundtrip');
  assert_close(e2, e, 'eul angles');
  assert_close(quat2rotm([2 0 0 0]), eye(3), 'quat normalize');
  aR = axang2rotm([1 2 3 0.4]);
  assert_close(axang2rotm(rotm2axang(aR)), aR, 'axang roundtrip');
end

function assert_close(A, B, msg)
  d = max(abs(A(:) - B(:)));
  if ~(d < 1e-8)
    error('shim self-check failed (%s): max abs %.3g', msg, d);
  end
end

function save_struct(path, s)
  if exist(path, 'file')
    delete(path);
  end
  save('-v7', path, '-struct', 's');
end
