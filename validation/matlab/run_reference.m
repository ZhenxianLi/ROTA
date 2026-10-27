% run_reference.m -- black-box reference run for the ROTA validation (ROADMAP 5.2, 5.5).
%
% What this script does, and nothing else:
%   1. loads the signals written by validation/export_mat.py (validation/signals/*.mat),
%   2. calls DOCUMENTED PUBLIC functions of the Signal Processing Toolbox on them, with every
%      parameter given explicitly (window, resolution, overlap, amplitude type, filter
%      bandwidth, decoupling, initial ridge point ...) and no reliance on defaults,
%   3. saves the outputs to validation/reference/<case>_ref.mat, and writes
%      validation/reference/versions.txt (release and toolbox versions) and
%      validation/reference/log.txt (every call made, warnings and errors).
%
% The functions are used as black boxes: only their documented inputs and outputs appear
% here. No toolbox code is opened, stepped into or reused, and no other data than the signals
% of rota.synth / validation/make_signals.py are processed. All comparison logic lives in
% validation/compare.py; nothing is evaluated here.
%
% Units: time in s, frequency in Hz, speed in rpm, order in cycles per revolution.
% Amplitude conventions are recorded next to each output (<id>_amplitude).
%
% Every call is wrapped in try/catch: a failure is logged and the remaining calls continue.
% A call may be given several variants; they are tried in order and the first one that runs
% is kept. Later variants drop optional arguments, never change the quantity being
% estimated; <id>_variant and <id>_call in the output file say which one ran, with its
% arguments. The Name,Value comma form is used throughout (valid in every release).
%
% Run from anywhere, e.g. from the repository root:
%     run('validation/matlab/run_reference.m')

global ROTA_REF_LOG
ROTA_REF_LOG = '';

% ---- locations (found from this file's own location, so independent of the current folder) -----
% run() changes into the folder of the script while it runs, so a path relative to the current
% folder cannot be trusted: search upwards from the script, the current folder and the CI workspace
% for the folder that holds validation/signals.
thisDir = fileparts(mfilename('fullpath'));
rootDir = find_root({thisDir, pwd, getenv('GITHUB_WORKSPACE')});
if isempty(rootDir)
    error('run_reference:noRoot', ...
          'Cannot find validation/signals above %s or %s; run validation/export_mat.py first.', thisDir, pwd);
end
sigDir = fullfile(rootDir, 'validation', 'signals');
refDir = fullfile(rootDir, 'validation', 'reference');
if ~isfolder(refDir)
    mkdir(refDir);
end

fid = fopen(fullfile(refDir, 'log.txt'), 'w');
if fid > 0
    fclose(fid);
    ROTA_REF_LOG = fullfile(refDir, 'log.txt');
end
logmsg('run_reference.m started; root folder: %s', rootDir);

% ---- versions -------------------------------------------------------------------------------
try
    set(0, 'DefaultFigureVisible', 'off');
catch
end
try
    fid = fopen(fullfile(refDir, 'versions.txt'), 'w');
    fprintf(fid, 'MATLAB version: %s\n', version);
    fprintf(fid, 'MATLAB release: %s\n', version('-release'));
    fprintf(fid, 'Computer: %s\n', computer);
    try
        fprintf(fid, 'Signal Processing Toolbox licence available: %d\n', license('test', 'Signal_Toolbox'));
    catch
    end
    fprintf(fid, '\nName | Version | Release | Date\n');
    v = ver;
    for i = 1:numel(v)
        fprintf(fid, '%s | %s | %s | %s\n', v(i).Name, v(i).Version, v(i).Release, v(i).Date);
    end
    fclose(fid);
    logmsg('MATLAB %s (%s)', version, version('-release'));
catch ME
    logmsg('could not write versions.txt: %s', oneline(ME.message));
end

% ---- loop over the signal files -------------------------------------------------------------
files = dir(fullfile(sigDir, '*.mat'));
if isempty(files)
    error('run_reference:noSignals', 'No .mat files in %s; run validation/export_mat.py first.', sigDir);
end
[~, sortIdx] = sort({files.name});
files = files(sortIdx);
% Part of the calls to make (environment variable ROTA_REF_PART, set by the CI matrix so that the
% slow Vold-Kalman calls of the long cases run in jobs of their own): 'all' or empty (every call),
% 'main' (all but the Vold-Kalman calls of the case comparison), 'vkwave' (orderwaveform only) or
% 'vktrack' (ordertrack in its Vold-Kalman form only). A part other than 'all' saves
% <case>_ref_<part>.mat; compare.py merges the parts of a case.
part = lower(strtrim(getenv('ROTA_REF_PART')));
if isempty(part)
    part = 'all';
end
if ~any(strcmp(part, {'all', 'main', 'vkwave', 'vktrack'}))
    error('run_reference:part', 'Unknown ROTA_REF_PART "%s" (all, main, vkwave or vktrack).', part);
end
logmsg('part: %s', part);
for k = 1:numel(files)
    process_case(fullfile(sigDir, files(k).name), refDir, part);
end
logmsg('run_reference.m finished');

% =================================================================================================
% Local functions (must stay at the end of the script file)
% =================================================================================================

function process_case(file, refDir, part)
% Run the applicable calls of one part on one case file and save <case>_ref.mat (part 'all') or
% <case>_ref_<part>.mat.
[~, base] = fileparts(file);
logmsg('---- case %s (%s), part %s', base, file, part);
doMain = any(strcmp(part, {'all', 'main'}));
doVkWave = any(strcmp(part, {'all', 'vkwave'}));
doVkTrack = any(strcmp(part, {'all', 'vktrack'}));
R = struct();
R.case_name = base;
try
    S = load(file);
    par = S.par;
    x = S.x(:);
    fs = double(S.fs);
    rpm = S.rpm(:);
    t = S.t(:);
    orders = double(S.orders(:)).';
    K = numel(orders);
    R.case_name = char(S.case_name);
    R.matlab_version = version;
    R.matlab_release = version('-release');
    logmsg('n = %d samples, fs = %g Hz, %d order(s), rpm %g ... %g', numel(x), fs, K, rpm(1), rpm(end));

    % -- tachometer speed (T5): pulses per revolution, state levels, output rate and fit explicit
    if doMain && par.do_tachorpm ~= 0
        tacho = S.tacho(:);
        common = {'PulsesPerRev', par.ppr, 'StateLevels', par.state_levels};
        R = run_call(R, 'tachorpm_smooth', {'rpm', 't', 'tp'}, { ...
            mkvariant('tachorpm', [{tacho, fs}, common, {'OutputFs', par.tacho_output_fs, 'FitType', 'smooth', 'FitPoints', par.fit_points}], ...
                      'pulses per rev, state levels, output rate, smooth fit with break points'), ...
            mkvariant('tachorpm', [{tacho, fs}, common, {'FitType', 'smooth', 'FitPoints', par.fit_points}], ...
                      'as above without the output rate'), ...
            mkvariant('tachorpm', [{tacho, fs}, {'PulsesPerRev', par.ppr}], ...
                      'pulses per rev only (other arguments at their defaults)')});
        R = run_call(R, 'tachorpm_linear', {'rpm', 't', 'tp'}, { ...
            mkvariant('tachorpm', [{tacho, fs}, common, {'OutputFs', par.tacho_output_fs, 'FitType', 'linear'}], ...
                      'pulses per rev, state levels, output rate, linear fit'), ...
            mkvariant('tachorpm', [{tacho, fs}, common, {'FitType', 'linear'}], ...
                      'as above without the output rate')});
    end

    % -- order-rpm map, average order spectrum, order tracks read from the map.
    %    'res' is a resolution bandwidth (window ENBW times the bin spacing); it is given as
    %    par.order_resolution_matlab = ENBW * par.order_resolution, so that the bin spacing and
    %    the window length are those of ROTA (export_mat.py). The same holds for rpmfreqmap.
    if doMain && par.do_maps ~= 0
        nv = {'Window', par.window_order_map, 'OverlapPercent', par.overlap_percent, ...
              'Amplitude', par.amplitude_map, 'Scale', 'linear'};
        res = par.order_resolution_matlab;
        R = run_call(R, 'rpmordermap', {'map', 'order', 'rpm', 'time', 'res'}, { ...
            mkvariant('rpmordermap', [{x, fs, rpm, res}, nv], ...
                      'order resolution, window, overlap percent, amplitude and scale explicit'), ...
            mkvariant('rpmordermap', {x, fs, rpm, res}, ...
                      'order resolution only (other arguments at their defaults)')});
        R.rpmordermap_res_in = res;
        R.rpmordermap_amplitude = par.amplitude_map;
        if R.rpmordermap_ok ~= 0
            map = R.rpmordermap_map;
            ordAxis = R.rpmordermap_order;
            rpmMap = R.rpmordermap_rpm;
            timeMap = R.rpmordermap_time;
            R = run_call(R, 'orderspectrum', {'spec', 'order'}, { ...
                mkvariant('orderspectrum', {map, ordAxis, 'Amplitude', par.amplitude_map}, ...
                          'order map, order axis, amplitude explicit'), ...
                mkvariant('orderspectrum', {map, ordAxis}, 'order map and order axis only')});
            R.orderspectrum_amplitude = par.amplitude_map;
            % The map form takes no name-value arguments (they apply only with a signal input,
            % error signal:rpmmap:MAPORDERTRACKNOARGS): amplitude type and scale are those of the
            % map, i.e. the 'Amplitude' and 'Scale' given to rpmordermap above.
            R = run_call(R, 'ordertrack_map', {'mag', 'rpm', 'time'}, { ...
                mkvariant('ordertrack', {map, ordAxis, rpmMap, timeMap, orders}, ...
                          'order map, its axes and the order list (amplitude and scale of the map)')});
            R.ordertrack_map_amplitude = par.amplitude_map;
        else
            R = skip_call(R, 'orderspectrum', 'rpmordermap failed');
            R = skip_call(R, 'ordertrack_map', 'rpmordermap failed');
        end
    end

    % -- frequency-rpm map
    if doMain && par.do_freqmap ~= 0
        nv = {'Window', par.window_freq_map, 'OverlapPercent', par.overlap_percent, ...
              'Amplitude', par.amplitude_map, 'Scale', 'linear'};
        res = par.freq_resolution_matlab;
        R = run_call(R, 'rpmfreqmap', {'map', 'freq', 'rpm', 'time', 'res'}, { ...
            mkvariant('rpmfreqmap', [{x, fs, rpm, res}, nv], ...
                      'frequency resolution, window, overlap percent, amplitude and scale explicit'), ...
            mkvariant('rpmfreqmap', {x, fs, rpm, res}, ...
                      'frequency resolution only (other arguments at their defaults)')});
        R.rpmfreqmap_res_in = res;
        R.rpmfreqmap_amplitude = par.amplitude_map;
    end

    % -- Vold-Kalman: waveforms and tracked magnitudes, bandwidth and decoupling explicit.
    %    'Bandwidth' is given as par.vk_bandwidth_matlab, the full width between the two -3 dB
    %    points, which is twice ROTA's bandwidth (the -3 dB offset from the carrier); the probe
    %    case VKBW below measures what the argument does (compare.py, table 10).
    if par.do_vk ~= 0
        refIdx = ones(1, K);                       % every order refers to the one speed column
        bw = par.vk_bandwidth_matlab;
        dec = par.vk_decouple ~= 0;
        R.vk_bandwidth_matlab = bw;
    end
    if doVkWave && par.do_vk ~= 0
        R = run_call(R, 'orderwaveform', {'xrec'}, { ...
            mkvariant('orderwaveform', {x, fs, rpm, orders, refIdx, 'FilterOrder', par.vk_filter_order, ...
                      'Bandwidth', bw, 'Decouple', dec}, ...
                      'filter order, bandwidth and decoupling explicit'), ...
            mkvariant('orderwaveform', {x, fs, [rpm rpm], orders, refIdx, 'FilterOrder', par.vk_filter_order, ...
                      'Bandwidth', bw, 'Decouple', dec}, ...
                      'as above, speed given as two identical columns'), ...
            mkvariant('orderwaveform', {x, fs, rpm, orders, refIdx, 'Bandwidth', bw, 'Decouple', dec}, ...
                      'bandwidth and decoupling explicit, filter order at its default')});
    end
    if doVkTrack && par.do_vk ~= 0
        % One speed column: no RPMREFIDX (ordertrack accepts it only with a speed matrix, error
        % signal:rpmmap:RPMMustBeMatrix); every order then refers to that column.
        R = run_call(R, 'ordertrack_vk', {'mag', 'rpm', 'time'}, { ...
            mkvariant('ordertrack', {x, fs, rpm, orders, 'Amplitude', par.vk_amplitude, ...
                      'Scale', 'linear', 'Bandwidth', bw, 'Decouple', dec}, ...
                      'Vold-Kalman form, one speed column: amplitude, scale, bandwidth and decoupling explicit'), ...
            mkvariant('ordertrack', {x, fs, [rpm rpm], orders, refIdx, 'Amplitude', par.vk_amplitude, ...
                      'Scale', 'linear', 'Bandwidth', bw, 'Decouple', dec}, ...
                      'as above, speed given as two identical columns with RPMREFIDX'), ...
            mkvariant('ordertrack', {x, fs, rpm, orders, 'Amplitude', par.vk_amplitude, ...
                      'Bandwidth', bw, 'Decouple', dec}, ...
                      'Vold-Kalman form, one speed column, without the scale argument')});
        R.ordertrack_vk_amplitude = par.vk_amplitude;
    end

    % -- Vold-Kalman bandwidth probe (VKBW): one order at constant speed and fixed tones at known
    %    offsets from it. The single-order filter is run with 'Bandwidth' par.vk_bandwidth_matlab
    %    (as in the comparisons above) and with the same number as ROTA's bandwidth; compare.py
    %    reads the gain at every offset off the returned waveforms.
    if doMain && isfield(par, 'do_vkprobe') && par.do_vkprobe ~= 0
        R = run_call(R, 'vkprobe', {'xrec'}, { ...
            mkvariant('orderwaveform', {x, fs, rpm, par.probe_order, 1, 'FilterOrder', par.vk_filter_order, ...
                      'Bandwidth', par.vk_bandwidth_matlab, 'Decouple', false}, ...
                      'one order, filter order, Bandwidth = par.vk_bandwidth_matlab, no decoupling')});
        R.vkprobe_bandwidth = par.vk_bandwidth_matlab;
        R = run_call(R, 'vkprobe_same', {'xrec'}, { ...
            mkvariant('orderwaveform', {x, fs, rpm, par.probe_order, 1, 'FilterOrder', par.vk_filter_order, ...
                      'Bandwidth', par.vk_bandwidth, 'Decouple', false}, ...
                      'as above with Bandwidth = par.vk_bandwidth (the number given to ROTA)')});
        R.vkprobe_same_bandwidth = par.vk_bandwidth;
    end

    % -- tacholess speed: order and one point (time, frequency) on its ridge, from the exact speed
    if doMain && par.do_rpmtrack ~= 0
        p = [par.rpmtrack_t0, par.rpmtrack_f0];
        fres = par.rpmtrack_resolution;
        R = run_call(R, 'rpmtrack', {'rpm', 'tout'}, { ...
            mkvariant('rpmtrack', {x, fs, par.rpmtrack_order, p, 'Method', 'stft', ...
                      'FrequencyResolution', fres, 'PowerPenalty', Inf, 'FrequencyPenalty', 0, ...
                      'StartTime', t(1), 'EndTime', t(end)}, ...
                      'order, initial ridge point and all name-value arguments explicit'), ...
            mkvariant('rpmtrack', {x, fs, par.rpmtrack_order, p, 'Method', 'stft', ...
                      'FrequencyResolution', fres, 'PowerPenalty', Inf, 'FrequencyPenalty', 0}, ...
                      'as above without start and end time'), ...
            mkvariant('rpmtrack', {x, fs, par.rpmtrack_order, p, 'Method', 'stft', ...
                      'FrequencyResolution', fres}, ...
                      'method and frequency resolution only'), ...
            mkvariant('rpmtrack', {x, fs, par.rpmtrack_order, p}, ...
                      'order and initial ridge point only (other arguments at their defaults)')});
    end
catch ME
    R.case_error = oneline(ME.message);
    logmsg('[case error] %s: %s', base, oneline(ME.message));
end

if strcmp(part, 'all')
    outFile = fullfile(refDir, [base '_ref.mat']);
else
    outFile = fullfile(refDir, [base '_ref_' part '.mat']);
end
try
    save(outFile, '-struct', 'R', '-v7');           % MAT-file version 7: readable by scipy.io.loadmat
    logmsg('saved %s', outFile);
catch ME
    logmsg('[save error] %s: %s; trying the default file format', outFile, oneline(ME.message));
    try
        save(outFile, '-struct', 'R');
        logmsg('saved %s (default format)', outFile);
    catch ME2
        logmsg('[save error] %s: %s', outFile, oneline(ME2.message));
    end
end
end

function root = find_root(starts)
% First folder, searching upwards (at most 5 levels) from each start, that contains validation/signals.
root = '';
for i = 1:numel(starts)
    d = starts{i};
    if isempty(d) || ~ischar(d)
        continue
    end
    for level = 1:6
        if isfolder(fullfile(d, 'validation', 'signals'))
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

function v = mkvariant(fn, args, label)
% One way to call a function: its name, the arguments and a short label.
v = struct('fn', fn, 'args', {args}, 'label', label);
end

function R = run_call(R, id, outNames, variants)
% Try the variants in order until one runs. Outputs go to R.<id>_<name>; R.<id>_ok is 1 or 0.
R.([id '_ok']) = 0;
R.([id '_variant']) = 0;
R.([id '_call']) = describe_call(variants{1});
R.([id '_msg']) = '';
R.([id '_seconds']) = NaN;
errs = '';
nout = numel(outNames);
for iv = 1:numel(variants)
    v = variants{iv};
    desc = describe_call(v);
    lastwarn('');
    t0 = tic;
    try
        out = {};
        for no = nout:-1:1                       % ask for fewer outputs if the call refuses
            try
                out = cell(1, no);
                [out{:}] = feval(v.fn, v.args{:});
                break
            catch ME1
                if no == 1 || ~too_many_outputs(ME1)
                    rethrow(ME1);
                end
            end
        end
        sec = toc(t0);
        wmsg = lastwarn;
        for k = 1:numel(out)
            R.([id '_' outNames{k}]) = out{k};
        end
        R.([id '_ok']) = 1;
        R.([id '_variant']) = iv;
        R.([id '_call']) = desc;
        R.([id '_seconds']) = sec;
        R.([id '_msg']) = errs;
        if numel(out) < nout
            R.([id '_msg']) = [R.([id '_msg']) sprintf('only %d of %d outputs returned. ', numel(out), nout)];
            logmsg('[note] %s: only %d of %d outputs returned', id, numel(out), nout);
        end
        if ~isempty(wmsg)
            R.([id '_msg']) = [R.([id '_msg']) 'warning: ' oneline(wmsg)];
            logmsg('[warn] %s: %s', id, oneline(wmsg));
        end
        logmsg('[ok]   %s, variant %d (%s), %.1f s', id, iv, v.label, sec);
        logmsg('       %s', desc);
        return
    catch ME
        msg = sprintf('variant %d (%s): [%s] %s', iv, v.label, ME.identifier, oneline(ME.message));
        logmsg('[FAIL] %s, %s', id, msg);
        errs = [errs msg ' | ']; %#ok<AGROW>
    end
end
R.([id '_msg']) = errs;
end

function R = skip_call(R, id, reason)
% Record a call that was not made because its input is missing.
R.([id '_ok']) = 0;
R.([id '_variant']) = 0;
R.([id '_call']) = '';
R.([id '_msg']) = ['skipped: ' reason];
R.([id '_seconds']) = NaN;
logmsg('[skip] %s: %s', id, reason);
end

function tf = too_many_outputs(ME)
tf = ~isempty(strfind(ME.identifier, 'TooManyOutputs')) || ...
     ~isempty(strfind(lower(ME.message), 'too many output'));
end

function s = describe_call(v)
% Text form of a call: function name and arguments (long arrays shown by size only).
parts = cell(1, numel(v.args));
for i = 1:numel(v.args)
    a = v.args{i};
    if ischar(a)
        parts{i} = ['''' a ''''];
    elseif (isnumeric(a) || islogical(a)) && isscalar(a)
        parts{i} = sprintf('%.10g', double(a));
    elseif (isnumeric(a) || islogical(a)) && numel(a) <= 8
        parts{i} = ['[' strtrim(sprintf('%.10g ', double(a))) ']'];
    elseif isnumeric(a)
        parts{i} = sprintf('<%s %dx%d>', class(a), size(a, 1), size(a, 2));
    else
        parts{i} = ['<' class(a) '>'];
    end
end
s = [v.fn '(' strjoin(parts, ', ') ')'];
end

function s = oneline(s)
% Single-line, length-limited text (error and warning messages).
s = regexprep(char(s), '\s+', ' ');
if numel(s) > 400
    s = [s(1:400) '...'];
end
end

function logmsg(varargin)
% Print to the console and append to log.txt.
global ROTA_REF_LOG
msg = sprintf(varargin{:});
fprintf('%s\n', msg);
if ~isempty(ROTA_REF_LOG)
    fid = fopen(ROTA_REF_LOG, 'a');
    if fid > 0
        fprintf(fid, '%s\n', msg);
        fclose(fid);
    end
end
end
