% run_sqat.m -- step 2 of the SQAT bridge (examples/09_sqat_bridge/README.md).
%
% What this script does, and nothing else:
%   1. adds a clone of SQAT (github.com/ggrecow/SQAT, GPL-3.0; never vendored in rota-order) to the
%      MATLAB path with its documented initialiser startup_SQAT,
%   2. loads the signals written by export_signals.py (build/sqat_bridge/signals/*.mat:
%      calibrated sound pressure in Pa at 48 kHz),
%   3. calls DOCUMENTED PUBLIC functions of SQAT on them (as described in the SQAT README and in
%      the help header of each function), with every parameter given explicitly:
%         Loudness_ISO532_1                 ISO 532-1:2017, time-varying method
%         Sharpness_DIN45692_from_loudness  DIN 45692:2009, from the specific loudness above
%                                           (fallback: Sharpness_DIN45692 from the signal)
%         Roughness_Daniel1997              Daniel & Weber (1997)
%         Tonality_ECMA418_2                ECMA-418-2, 4th edition (2025)
%   4. saves the time series to build/sqat_bridge/results/<name>_sqat.mat (MAT v7, readable by
%      scipy.io.loadmat), and writes results/versions.txt and results/log.txt.
%
% SQAT is used as a black box: only documented inputs and outputs appear here. No SQAT code is
% copied into rota-order. Binning along rpm happens in import_results.py.
%
% Units and conventions: time in s; loudness in sone (and loudness level in phon), sharpness in
% acum, roughness in asper, tonality in tu_HMS, tonal frequency in Hz. Input: sound pressure in
% Pa (instantaneous values; SQAT computes its own RMS levels).
%
% Every call is wrapped in try/catch: a failing call is logged and the others continue.
%
% Location of SQAT: environment variable SQAT_DIR, else build/sqat_bridge/SQAT under the
% repository root. Run from anywhere, e.g. from the repository root:
%     run('examples/09_sqat_bridge/run_sqat.m')

global SQAT_BRIDGE_LOG
SQAT_BRIDGE_LOG = '';

% ---- explicit parameters ---------------------------------------------------------------------
P.loudness_field = 0;              % ISO 532-1: 0 = free field, 1 = diffuse field
P.loudness_method = 2;             % ISO 532-1: 2 = time-varying (from the audio signal)
P.loudness_time_skip = 0.5;        % s, only affects SQAT's own statistics (N5 ...)
P.sharpness_weight = 'DIN45692';   % DIN 45692 weighting function g(z)
P.sharpness_time_skip = 0.5;       % s, statistics only
P.roughness_time_skip = 1.5;       % s, statistics only
P.tonality_field = 'free-frontal'; % ECMA-418-2 outer and middle ear filter: free field, frontal
P.tonality_time_skip = 0.304;      % s, ECMA-418-2 section 6.2.9 minimum (statistics only)
P.show = false;                    % no figures

% ---- locations (found from this file's location, the current folder or the CI workspace) ----
thisDir = fileparts(mfilename('fullpath'));
rootDir = find_root({thisDir, pwd, getenv('GITHUB_WORKSPACE')});
if isempty(rootDir)
    error('run_sqat:noRoot', ...
          'Cannot find build/sqat_bridge/signals above %s or %s; run export_signals.py first.', thisDir, pwd);
end
baseDir = fullfile(rootDir, 'build', 'sqat_bridge');
sigDir = fullfile(baseDir, 'signals');
resDir = fullfile(baseDir, 'results');
if ~isfolder(resDir)
    mkdir(resDir);
end
fid = fopen(fullfile(resDir, 'log.txt'), 'w');
if fid > 0
    fclose(fid);
    SQAT_BRIDGE_LOG = fullfile(resDir, 'log.txt');
end
logmsg('run_sqat.m started; root folder: %s', rootDir);
try
    set(0, 'DefaultFigureVisible', 'off');
catch
end

% ---- SQAT on the path ------------------------------------------------------------------------
sqatDir = getenv('SQAT_DIR');
if isempty(sqatDir)
    sqatDir = fullfile(baseDir, 'SQAT');
end
if ~isfile(fullfile(sqatDir, 'startup_SQAT.m'))
    error('run_sqat:noSQAT', 'SQAT not found in %s (set SQAT_DIR or clone it there).', sqatDir);
end
try
    addpath(sqatDir);
    startup_SQAT([sqatDir filesep]);   % documented initialiser: adds the SQAT folders to the path
    logmsg('SQAT initialised from %s', sqatDir);
catch ME
    % README of SQAT: adding psychoacoustic_metrics, sound_level_meter and utilities suffices.
    logmsg('[warn] startup_SQAT failed (%s); adding the metric folders instead', oneline(ME.message));
    addpath(genpath(fullfile(sqatDir, 'psychoacoustic_metrics')));
    addpath(genpath(fullfile(sqatDir, 'sound_level_meter')));
    addpath(genpath(fullfile(sqatDir, 'utilities')));
end
needed = {'Loudness_ISO532_1', 'Sharpness_DIN45692_from_loudness', 'Sharpness_DIN45692', ...
          'Roughness_Daniel1997', 'Tonality_ECMA418_2'};
for i = 1:numel(needed)
    logmsg('which %s: %s', needed{i}, which(needed{i}));
end

% ---- versions -------------------------------------------------------------------------------
try
    fid = fopen(fullfile(resDir, 'versions.txt'), 'w');
    fprintf(fid, 'MATLAB version: %s\n', version);
    fprintf(fid, 'MATLAB release: %s\n', version('-release'));
    fprintf(fid, 'Computer: %s\n', computer);
    fprintf(fid, 'SQAT folder: %s\n', sqatDir);
    fprintf(fid, 'SQAT ref: %s\n', getenv('SQAT_REF'));
    fprintf(fid, 'SQAT commit: %s\n', getenv('SQAT_COMMIT'));
    fprintf(fid, '\nName | Version | Release | Date\n');
    v = ver;
    for i = 1:numel(v)
        fprintf(fid, '%s | %s | %s | %s\n', v(i).Name, v(i).Version, v(i).Release, v(i).Date);
    end
    fclose(fid);
    logmsg('MATLAB %s (%s); SQAT %s (%s)', version, version('-release'), getenv('SQAT_REF'), getenv('SQAT_COMMIT'));
catch ME
    logmsg('could not write versions.txt: %s', oneline(ME.message));
end

% ---- loop over the signals -------------------------------------------------------------------
files = dir(fullfile(sigDir, '*.mat'));
if isempty(files)
    error('run_sqat:noSignals', 'No .mat files in %s; run export_signals.py first.', sigDir);
end
[~, order] = sort({files.name});
files = files(order);
for k = 1:numel(files)
    process_signal(fullfile(sigDir, files(k).name), resDir, P);
end
logmsg('run_sqat.m finished');

% =================================================================================================
% Local functions (must stay at the end of the script file)
% =================================================================================================

function process_signal(file, resDir, P)
[~, base] = fileparts(file);
logmsg('---- %s (%s)', base, file);
R = struct();
R.name = base;
R.params = P;
R.matlab_release = version('-release');
R.sqat_ref = getenv('SQAT_REF');
R.sqat_commit = getenv('SQAT_COMMIT');
try
    S = load(file);
    x = double(S.x(:));
    fs = double(S.fs);
    logmsg('n = %d samples, fs = %g Hz, RMS %.4f Pa', numel(x), fs, sqrt(mean(x.^2)));

    % -- ISO 532-1:2017 time-varying loudness (Zwicker), free field
    [R, L] = run_call(R, 'loudness', ...
        'Loudness_ISO532_1', {x, fs, P.loudness_field, P.loudness_method, P.loudness_time_skip, P.show});
    if R.loudness_ok
        R.loudness_t = col(L.time);
        R.loudness_sone = col(L.InstantaneousLoudness);
        R.loudness_phon = col(L.InstantaneousLoudnessLevel);
        R.loudness_N5 = getfield_or_nan(L, 'N5');
        R.loudness_Nmean = getfield_or_nan(L, 'Nmean');
    end

    % -- DIN 45692:2009 sharpness: from the specific loudness of the call above (same loudness);
    %    if that call failed or the function is unavailable, from the signal.
    done = false;
    if R.loudness_ok
        [R, Sh] = run_call(R, 'sharpness', 'Sharpness_DIN45692_from_loudness', ...
            {L.InstantaneousSpecificLoudness, P.sharpness_weight, L.time, P.sharpness_time_skip, P.show});
        done = R.sharpness_ok;
    end
    if ~done
        [R, Sh] = run_call(R, 'sharpness', 'Sharpness_DIN45692', ...
            {x, fs, P.sharpness_weight, P.loudness_field, P.loudness_method, P.sharpness_time_skip, P.show, P.show});
    end
    if R.sharpness_ok
        R.sharpness_t = col(Sh.time);
        R.sharpness_acum = col(Sh.InstantaneousSharpness);
    end

    % -- Daniel & Weber (1997) roughness
    [R, Ro] = run_call(R, 'roughness', 'Roughness_Daniel1997', {x, fs, P.roughness_time_skip, P.show});
    if R.roughness_ok
        R.roughness_t = col(Ro.time);
        R.roughness_asper = col(Ro.InstantaneousRoughness);
    end

    % -- ECMA-418-2 (4th edition, 2025) tonality, Sottek Hearing Model, mono input
    [R, To] = run_call(R, 'tonality', 'Tonality_ECMA418_2', {x, fs, P.tonality_field, P.tonality_time_skip, P.show});
    if R.tonality_ok
        R.tonality_t = col(To.timeOut);
        R.tonality_tu = col(To.tonalityTDep);
        R.tonality_freq_hz = col(To.tonalityTDepFreqs);
        R.tonality_avg = getfield_or_nan(To, 'tonalityAvg');
    end
catch ME
    logmsg('[FAIL] %s: %s', base, oneline(ME.message));
    R.error = oneline(ME.message);
end
close all force
outFile = fullfile(resDir, [base '_sqat.mat']);
try
    save(outFile, '-struct', 'R', '-v7');           % MAT-file version 7: readable by scipy.io.loadmat
    logmsg('saved %s', outFile);
catch ME
    logmsg('[save error] %s: %s', outFile, oneline(ME.message));
end
end

function [R, out] = run_call(R, id, fn, args)
% One call with try/catch; R.<id>_ok, _call, _seconds and _msg record what happened.
R.([id '_ok']) = 0;
R.([id '_call']) = describe_call(fn, args);
R.([id '_seconds']) = NaN;
R.([id '_msg']) = '';
out = struct();
lastwarn('');
t0 = tic;
try
    out = feval(fn, args{:});
    R.([id '_ok']) = 1;
    R.([id '_seconds']) = toc(t0);
    w = lastwarn;
    if ~isempty(w)
        R.([id '_msg']) = ['warning: ' oneline(w)];
        logmsg('[warn] %s: %s', id, oneline(w));
    end
    logmsg('[ok]   %s, %.1f s: %s', id, R.([id '_seconds']), R.([id '_call']));
catch ME
    R.([id '_seconds']) = toc(t0);
    R.([id '_msg']) = sprintf('[%s] %s', ME.identifier, oneline(ME.message));
    logmsg('[FAIL] %s, %.1f s: %s -> %s', id, R.([id '_seconds']), R.([id '_call']), R.([id '_msg']));
    if ~isempty(ME.stack)
        logmsg('       at %s line %d', ME.stack(1).name, ME.stack(1).line);
    end
end
close all force
end

function v = col(v)
v = double(v(:));
end

function v = getfield_or_nan(s, f)
if isfield(s, f)
    v = double(s.(f));
else
    v = NaN;
end
end

function root = find_root(starts)
% First folder, searching upwards from each start, that contains build/sqat_bridge/signals.
root = '';
for i = 1:numel(starts)
    d = starts{i};
    if isempty(d) || ~ischar(d)
        continue
    end
    for level = 1:6
        if isfolder(fullfile(d, 'build', 'sqat_bridge', 'signals'))
            root = d;
            return
        end
        parent = fileparts(d);
        if isempty(parent) || strcmp(parent, d)
            break
        end
        d = parent;
    end
end
end

function s = describe_call(fn, args)
% Text form of a call: function name and arguments (arrays shown by size only).
parts = cell(1, numel(args));
for i = 1:numel(args)
    a = args{i};
    if ischar(a)
        parts{i} = ['''' a ''''];
    elseif (isnumeric(a) || islogical(a)) && isscalar(a)
        parts{i} = sprintf('%.10g', double(a));
    elseif isnumeric(a)
        parts{i} = sprintf('<%s %dx%d>', class(a), size(a, 1), size(a, 2));
    else
        parts{i} = ['<' class(a) '>'];
    end
end
s = [fn '(' strjoin(parts, ', ') ')'];
end

function s = oneline(s)
s = regexprep(char(s), '\s+', ' ');
if numel(s) > 400
    s = [s(1:400) '...'];
end
end

function logmsg(varargin)
% Print to the console and append to log.txt.
global SQAT_BRIDGE_LOG
msg = sprintf(varargin{:});
fprintf('%s\n', msg);
if ~isempty(SQAT_BRIDGE_LOG)
    fid = fopen(SQAT_BRIDGE_LOG, 'a');
    if fid > 0
        fprintf(fid, '%s\n', msg);
        fclose(fid);
    end
end
end
