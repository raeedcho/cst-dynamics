function trial_data = convertSMILEtoTD(smile_data,params)
% CONVERTSMILETOTD converts SMILE trial data type to TrialData
% Outputs:
%   trial_data - TrialData structure
% Inputs:
%   smile_data - data structure to convert from SMILE type to TrialData
%   params - Parameters struct
%       .bin_size - bin size in seconds to convert to (default: 0.001)
%       .array_alias - cell array for renaming array names.
%           First col is name to replace, second col is replacement name. Rows are different aliases
%       .valid_sort_idx - list of array channel unit indices to keep. Defaults to 1:10
%       .meta - struct containing meta information to add to file (default: emtpy struct)
%           Note that monkey, date, task are extracted from smile_data already.
%
% Written by Raeed Chowdhury Feb 2020

    % check if params was passed in
    if nargin<2
        params = struct();
    end

    num_random_targs=8;
    array_name = 'MC';
    assignParams(who,params);

    % first get unit guide over all of smile_data
    unit_guide = get_smile_unit_guide(smile_data,params);

    % split up M1 and PMd for Prez data
    % ch 1-32 & 97-128 are PMd
    % ch 33-96 are M1
    split_unit_guide_rows = struct(...
        'M1',unit_guide(:,1)>=33 & unit_guide(:,1)<=96, ...
        'PMd',unit_guide(:,1)<32 | unit_guide(:,1)>96 ...
    );
    
    
    % fill in missing CST cursor samples
    params.fill_err = false;
%     try
        [smile_data,miss4] = errorCursorSaveFix(smile_data);
%     catch ME
%         params.fill_err = true;
%     end

    % Loop over smile_data, which is already split up by trial
    td_cell = cell(1,length(smile_data));
    for trialnum = 1:length(smile_data)
        % initialize with all the stuff we want
        td_cell{trialnum} = struct(...
            'monkey',NaN,...
            'task',NaN,...
            'date_time',NaN,...
            'trial_id',NaN,...
            'result',NaN,...
            'bin_size',NaN,...
            'lambda',NaN,...
            'ct_location',NaN,...
            'rt_locations',NaN,...
            'idx_startTime',NaN,...
            'idx_ctHoldTime',NaN,...
            'idx_pretaskHoldTime',NaN,...
            'idx_goCueTime',NaN,...
            'idx_rtgoCueTimes',NaN,...
            'idx_rtHoldTimes',NaN,...
            'idx_cstStartTime',NaN,...
            'idx_cstEndTime',NaN,...
            'idx_rewardTime',NaN,...
            'idx_failTime',NaN,...
            'idx_endTime',NaN...
            );
    
        td_cell{trialnum} = parse_smile_meta(td_cell{trialnum},smile_data(trialnum),params);
        td_cell{trialnum} = parse_smile_events(td_cell{trialnum},smile_data(trialnum),params);
        td_cell{trialnum} = parse_smile_behavior(td_cell{trialnum},smile_data(trialnum),params);
        td_cell{trialnum} = parse_smile_eye_data(td_cell{trialnum},smile_data(trialnum),params);
        td_cell{trialnum} = parse_smile_spikes(td_cell{trialnum},smile_data(trialnum),unit_guide,params);

        td_cell{trialnum} = assign_aborts(td_cell{trialnum});

        % split up M1 and PMd channels
        arrays = fieldnames(split_unit_guide_rows);
        for arraynum = 1:length(arrays)
            split_array_name = arrays{arraynum};
            array_unit_guide_rows = split_unit_guide_rows.(split_array_name);
            td_cell{trialnum}.(sprintf('%s_unit_guide',split_array_name)) = unit_guide(array_unit_guide_rows,:);

            full_spikes = td_cell{trialnum}.(sprintf('%s_spikes',array_name));
            td_cell{trialnum}.(sprintf('%s_spikes',split_array_name)) = full_spikes(:,array_unit_guide_rows);
        end

        td_cell{trialnum} = rmfield(td_cell{trialnum},'timevec');
    end
    % check fields for consistency
    num_fields = cellfun(@(x) length(fieldnames(x)),td_cell);
    if length(unique(num_fields))>1
        warning('something has gone wrong')
    end
    trial_data=cat(2,td_cell{:});

    
    
%     trial_data = reorderTDfields(trial_data);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sub-functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trial = parse_smile_meta(in_trial,smile_trial,params)
    % Get meta data out of smile trial data structure

    % default params
    meta = [];
    num_random_targs = 8;
    assignParams(who,params)

    % copy over input trial
    trial = in_trial;
    trial.monkey = smile_trial.Overview.subjectName;
    trial.date_time = datestr(datenum(num2str(smile_trial.Overview.date),'yyyymmddHHMMSS'),'yyyy/mm/dd HH:MM:SS');
    trial.trial_id = sscanf(smile_trial.Overview.trialNumber,'Trial%d');

    % simplify task name
    if startsWith(smile_trial.Overview.trialName,'RandomTargetTask_20220630')
        trial.task = 'RTT';
    elseif startsWith(smile_trial.Overview.trialName,'CST')
        trial.task = 'CST';
    elseif startsWith(smile_trial.Overview.trialName,'CenterOut')
        trial.task = 'CO';
    else
        trial.task = smile_trial.Overview.trialName;
    end

    % get center target location
    ct_reach_state = smile_trial.Parameters.StateTable(strcmpi(vertcat(smile_trial.Parameters.StateTable.stateName),'Reach to Center'));
    center_target_idx = strcmpi(ct_reach_state.StateTargets.names,'starttarget') | strcmpi(ct_reach_state.StateTargets.names,'start');
    trial.ct_location = ct_reach_state.StateTargets.location(center_target_idx,:);
    trial.ct_location(2) = -trial.ct_location(2); % phasespace inverts y-axis for some reason

    % get reach target location
    if startsWith(smile_trial.Overview.trialName,'RandomTargetTask_20220630')
        trial.rt_locations = zeros(num_random_targs,3);
        for targetnum = 1:num_random_targs
            targ_reach_state = smile_trial.Parameters.StateTable(strcmpi(vertcat(smile_trial.Parameters.StateTable.stateName),sprintf('Reach to Target %d',targetnum)));
            targ_idx = find(strcmpi(targ_reach_state.StateTargets.names,sprintf('randomtarg%d',targetnum)),1);
            trial.rt_locations(targetnum,:) = targ_reach_state.StateTargets.location(targ_idx,:);
        end
        trial.rt_locations(:,2) = -trial.rt_locations(:,2);
    end

    % trial result (0-failure, 1-success, 2-error/abort)
    switch smile_trial.Overview.trialStatus
        case 0
            trial.result = 'F';
        case 1
            trial.result = 'R';
        case 2
            trial.result = 'A';
    end

    % get lambda for CST trials
    if contains(smile_trial.Overview.trialName,'CST')
        trial.lambda = smile_trial.Parameters.ForceParameters.initialLambda;
    end

    if ~isempty(meta)
        fnames = fieldnames(meta);
        for fieldnum = 1:length(fnames)
            trial.(fnames{fieldnum}) = meta.(fnames{fieldnum});
        end
    end
end

function trial = parse_smile_events(in_trial,smile_trial,params)
    % parse out events from smile trial data structure (also add timevec to trial based on end time and bin_size)
    % assume that we only care about stuff from time zero (which should end up being idx_startTime) to end of trial

    % default params
    bin_size = 0.001;
    num_random_targs = 8;
    assignParams(who,params)

    % copy over input trial
    trial = in_trial;

    if ~isfield(trial,'bin_size') || isnan(trial.bin_size)
        trial.bin_size = bin_size;
    end

    % figure out timevector for this trial (in seconds)
    % smile_data specifies everything as milliseconds
    if ~isfield(trial,'timevec') || isnan(trial.timevec)
        state_transitions = smile_trial.TrialData.stateTransitions;
        endTime = state_transitions(2,state_transitions(1,:)==-1);
        end_time_in_s = double(endTime)/1000;
        trial.timevec = 0:bin_size:end_time_in_s;
    end

    % startTime is probably state 1, but it generally seems to be called 'Reach to Center' for RTT and CST in Prez
    trial.idx_startTime = timevec_event_index(smile_trial,'Reach to Center',trial.timevec);

    % ambiguous cue center hold
    trial.idx_ctHoldTime = timevec_event_index(smile_trial,'Hold Center (Ambiguous Cue)',trial.timevec);

    % task specific stuff
    if contains(smile_trial.Overview.trialName,'CST with Ambiguous Center Start')
        % center hold
        trial.idx_pretaskHoldTime = timevec_event_index(smile_trial,'Hold Center (CST Cue)',trial.timevec);

        % go cue
        trial.idx_goCueTime = timevec_event_index(smile_trial,'Control System',trial.timevec);

        % fail
        if smile_trial.Overview.trialStatus==0
            trial.idx_failTime = timevec_event_index(smile_trial,'Failure Display',trial.timevec);
        end
    elseif contains(smile_trial.Overview.trialName,'RandomTargetTask_20220630')
        
        % center hold
        trial.idx_pretaskHoldTime = timevec_event_index(smile_trial,'Hold Center (RTT Cue)',trial.timevec);

        % first go cue
        trial.idx_goCueTime = timevec_event_index(smile_trial,'Reach to Target 1',trial.timevec);

        % random target go cues
        % NOTE: this only picks out the cue to reach for the random target and the time the monkey reaches and holds in the target (ignoring aborts)
        trial.idx_rtgoCueTimes = zeros(1,num_random_targs);
        trial.idx_rtHoldTimes = zeros(1,num_random_targs);
        for targetnum = 1:num_random_targs
            rt_cue_times = timevec_event_index(smile_trial,sprintf('Reach to Target %d',targetnum),trial.timevec);
            rt_hold_times = timevec_event_index(smile_trial,sprintf('Hold at Target %d',targetnum),trial.timevec);
            trial.idx_rtgoCueTimes(targetnum) = rt_cue_times(1);
            trial.idx_rtHoldTimes(targetnum) = rt_hold_times(end);
        end

        % fail
        if smile_trial.Overview.trialStatus==0
            for targetnum = 1:num_random_targs
                possible_idx_failTime = timevec_event_index(smile_trial,sprintf('Target %d Failure'),trial.timevec);
                if ~isnan(possible_idx_failTime)
                    trial.idx_failTime=possible_idx_failTime;
                    break
                end
            end
        end
    elseif contains(smile_trial.Overview.trialName,'CenterOut_20181112')
        % center hold
        trial.idx_ctHoldTime = timevec_event_index(smile_trial,'Center Hold',trial.timevec);

        % go cue
        trial.idx_goCueTime = timevec_event_index(smile_trial,'Reach to Target',trial.timevec);

        % target hold
%         trial.idx_otHoldTime = timevec_event_index(smile_trial,'Target Hold',trial.timevec);

        % fail
        if smile_trial.Overview.trialStatus==0
            trial.idx_failTime = timevec_event_index(smile_trial,'Failure (Center)',trial.timevec);
            if isnan(trial.idx_failTime)
                trial.idx_failTime = timevec_event_index(smile_trial,'Target Failure',trial.timevec);
                if isnan(trial.idx_failTime)
                    trial.idx_failTime = timevec_event_index(smile_trial,'Non-Attempt',trial.timevec);
                end
            end
        end
    end

    % reward
    if smile_trial.Overview.trialStatus==1
        trial.idx_rewardTime = timevec_event_index(smile_trial,'Success',trial.timevec);
    end

    % assume last index of the timevec is the end time (might be off by one if SMILE data changes to allow non-int transition times)
    trial.idx_endTime = length(trial.timevec);
end

function idx = timevec_event_index(smile_trial,event_name,timevec)
    % helper function to figure out index for nearest time in timevec (assuming timevec is in seconds)
    state_transitions = smile_trial.TrialData.stateTransitions;
    statenum = find(strcmpi(cat(1,smile_trial.Parameters.StateTable.stateName),event_name));

    if isempty(statenum)
        idx = NaN;
        return
    end
    
    query_time = state_transitions(2,state_transitions(1,:)==statenum);
    idx = interp1(timevec,1:length(timevec),double(query_time)/1000,'nearest');
    
    if isempty(idx)
        idx = NaN;
    end
end

function trial = parse_smile_behavior(in_trial,smile_trial,params)
    % parse out behavior (hand position and cursor position) from smile data struct
    % assumes in_trial has timevec based on the endTime found in parse_smile_events

    assert(isfield(in_trial,'timevec'),'Missing temporary timevector in trial data for some reason...')

    % assume we're going for millisecond binning
    bin_size = 0.001;
    fill_err = false;
    assignParams(who,params)
    
    % copy over input trial
    trial = in_trial;

    % first get the phasespace marker data
    phasespace_freq = double(smile_trial.TrialData.Marker.frequency);
    phasespace_data = smile_trial.TrialData.Marker.rawPositions;
    phasespace_mask = [1 -1 1]; %phasespace inverts y axis for some reason
    
    % filtering params
    filter_n = 50;
    kaiser_beta = 20;

    if ~isempty(phasespace_data)
        visual_timevec = phasespace_data(:,7)/1000;
        hand_timevec = phasespace_data(:,6)/1000;
        marker_pos = phasespace_data(:,2:4) .* repmat(phasespace_mask,size(phasespace_data,1),1);

        % resample hand pos data to new timevector
        [temp_resampled,t_resamp] = resample_signals(marker_pos,hand_timevec,struct( ...
            'bin_size',bin_size, ...
            'samprate',phasespace_freq,...
            'filter_n',filter_n,...
            'kaiser_beta',kaiser_beta...
            ));
        trial.hand_pos = interp1(t_resamp,temp_resampled,trial.timevec');
        trial.rel_hand_pos = trial.hand_pos-trial.ct_location;

        % resample cursor pos data to new timevector
        [temp_resampled,t_resamp] = resample_signals(marker_pos(:,1:2),visual_timevec,struct( ...
            'bin_size',bin_size, ...
            'samprate',phasespace_freq,...
            'filter_n',filter_n,...
            'kaiser_beta',kaiser_beta...
            ));
        trial.cursor_pos = interp1(t_resamp,temp_resampled,trial.timevec');
        trial.rel_cursor_pos = trial.cursor_pos-trial.ct_location(1:2);

        % if this is a CST trial, we need to add cursor info during the Control System state (but only if there wasn't an error filling missing samples)
        trial.cst_cursor_command = nan(size(trial.cursor_pos));
        if ~isempty(smile_trial.TrialData.Marker.errorCursor)
            cursor_data = smile_trial.TrialData.Marker.errorCursor;
            cursor_timevec = cursor_data(:,4)/1000; % this is not necessarily regularly spaced but should be after the fix
            cursor_pos = cursor_data(:,1:2);
            cursor_pos(:,2) = -cursor_pos(:,2); %phasespace inverts y-axis for some reason
            
            if fill_err
                warning('Unable to fill missing CST samples for date %s, trying to interpolate samples',trial.date_time)

                % Note: following section is unnecessary after cursor position samples are filled in with Nicole's fix (errorCursorSaveFix)
                % interpolate to uniform sampling rate (assuming an on-average consistent sampling rate)
                dt = (cursor_timevec(end)-cursor_timevec(1))/length(cursor_timevec);
                tGrid = (cursor_timevec(1):dt:cursor_timevec(end))';
                cursor_pos_interp = zeros(length(tGrid),size(cursor_pos,2));
                for i=1:size(cursor_pos,2)
                    cursor_pos_interp(:,i) = interp1(cursor_timevec,cursor_pos(:,i),tGrid);
                end
                
                cursor_timevec = tGrid;
                cursor_pos = cursor_pos_interp;
            else
                dt = mean(diff(cursor_timevec));
            end

            % resample signals to match new time vector (assumes a uniform sampling rate)
            [temp_resampled,t_resamp] = resample_signals(cursor_pos,cursor_timevec,struct( ...
                'bin_size',bin_size, ...
                'samprate',1/dt,...
                'filter_n',filter_n,...
                'kaiser_beta',kaiser_beta...
                ));
            cursor_pos_resamp = interp1(t_resamp,temp_resampled,trial.timevec');

            % replace old cursor pos with this one where applicable
            error_cursor_idx = ~isnan(cursor_pos_resamp(:,1));
            trial.cursor_pos(error_cursor_idx,:) = cursor_pos_resamp(error_cursor_idx,:);
            trial.rel_cursor_pos(error_cursor_idx,:) = trial.cursor_pos(error_cursor_idx,:)-trial.ct_location(1:2);
            
            trial.cst_cursor_command(error_cursor_idx,1) = trial.lambda*(trial.rel_hand_pos(error_cursor_idx,1)+trial.rel_cursor_pos(error_cursor_idx,1));
            trial.cst_cursor_command(error_cursor_idx,2) = zeros(sum(error_cursor_idx),1);
            
            % get start and end indices (not the same as go and reward times)
            trial.idx_cstStartTime = find(error_cursor_idx,1,'first');
            trial.idx_cstEndTime = find(error_cursor_idx,1,'last');
        end
    else
        trial.hand_pos = [];
        trial.cursor_pos = [];
        trial.rel_hand_pos = [];
        trial.rel_cursor_pos = [];
        trial.cst_cursor_command = [];
    end
end

function trial = parse_smile_eye_data(in_trial,smile_trial,params)
    % parse out eye tracking (eye position and pupil diameter) from smile data struct
    % assumes in_trial has timevec based on the endTime found in parse_smile_events

    assert(isfield(in_trial,'timevec'),'Missing temporary timevector in trial data for some reason...')
    
    % params
    bin_size = 0.001;
    blink_thresh = -9;
    assignParams(who,params)
    
    % copy over input trial
    trial = in_trial;

    % assert length of analog channel data matches the timevec
    % (tests millisecond timing assumption, mostly. Could be off by one maybe.)
    analog_chan_timevec = (1:size(smile_trial.TrialData.analogData,1))'/double(1e3);
    assert(...
        analog_chan_timevec(end)==in_trial.timevec(end),...
        'Length of analog channel data does not match length of trial time...'...
    )

    % map out correspondence between new signal name and old channel names
    eye_sig_map = struct(...
        'raw_left_eye_pos',{{'Left Eye X','Left Eye Y'}},...
        'raw_left_pupil',{{'Left Pupil'}}...
    );
    all_channel_names = smile_trial.Definitions.analogChannelNames;
    eye_sig_names = fieldnames(eye_sig_map);

    % extract each signal
    blink_mask = true(length(trial.timevec),1);
    for eye_sig_num = 1:length(eye_sig_names)
        sig_name = eye_sig_names{eye_sig_num};
        channel_names = eye_sig_map.(sig_name);
        if ~all(ismember(channel_names,all_channel_names))
            warning('Missing part of %s signal',sig_name)
        end

        chan_idx = ismember(all_channel_names,channel_names);
        chan_data = smile_trial.TrialData.analogData(:,chan_idx);

        if ~isclose(bin_size,1e-3)
            % resample signals to new bin size
            warning('Resampling signals to new bin size...not sure if this will work well')
            [binned_chan_data,t_binned] = resample_signals(chan_data,analog_chan_timevec,struct( ...
                'bin_size',bin_size, ...
                'samprate',1e3,...
                'filter_n',filter_n,...
                'kaiser_beta',kaiser_beta...
            ));
        else
            % already at the right sampling rate, just pass through
            binned_chan_data = chan_data;
            t_binned=analog_chan_timevec;
        end

        % align to timevec using interpolation
        % (this should probably do nothing, but is here to make sure things play well)
        sig_data = interp1(t_binned,binned_chan_data,trial.timevec');

        % first point will be extrapolated, since we don't have time 0
        % so we look for other extrapolations
        if any(any(isnan(sig_data(2:end,:))))
            warning('Probably some extrapolation going on in the eye data...')
        end

        % rule in non-blink points through mask (if all signals are below thresh, mask them)
        blink_mask = blink_mask & all(sig_data<blink_thresh,2);

        % flip y position
        if ismember('Left Eye Y',channel_names)
            flip_idx = ismember(channel_names,'Left Eye Y');
            sig_data(:,flip_idx) = -sig_data(:,flip_idx);
        end

        % assign to trial data
        trial.(sig_name) = sig_data;
    end

%     if any(trial.left_pupil<blink_thresh)
%         fprintf('we have a blink')
%     end

    % remove eye blinks
    for eye_sig_num = 1:length(eye_sig_names)
        sig_name = eye_sig_names{eye_sig_num};
        trial.(sig_name)(blink_mask,:) = NaN;
    end
end

function trial = parse_smile_spikes(in_trial,smile_trial,unit_guide,params)
    % parse out neural activity from smile data struct
    
    % import parameters
    array_alias = {};
    array_name = smile_trial.TrialData.TDT.brainArea;
    assignParams(who,params);

    % copy over input trial
    trial = in_trial;

    % Extract info from smile_trial
    if isempty(array_name)
        warning('Array name is empty...using MC as a fallback name')
        array_name = 'MC';
    end

    % alias array names if we want
    if ~isempty(array_alias)
        for array_idx = 1:size(array_alias,1)
            if strcmp(array_name,array_alias{1,1})
                array_name = array_alias{1,2};
            end
        end
    end

    array_name = strrep(array_name,' ','');

    % construct a matrix of binned spikes and fill with spikes
    binned_spikes = zeros(length(trial.timevec),size(unit_guide,1));

    % fill matrix with histcounts
    spike_times = double(smile_trial.TrialData.TDT.snippetInfo');
    if ~isempty(spike_times)
        t_bin_edges = [trial.timevec trial.timevec(end)+mode(diff(trial.timevec))];
        for unit_idx = 1:size(unit_guide,1)
            unit_to_bin = unit_guide(unit_idx,:);
            filter_idx = spike_times(:,1)==unit_to_bin(1) & spike_times(:,2)==unit_to_bin(2);
            binned_spikes(:,unit_idx) = histcounts(spike_times(filter_idx,3)/1000,t_bin_edges)';
        end
    end

    % add to trial
    trial.(sprintf('%s_unit_guide',array_name)) = unit_guide;
    trial.(sprintf('%s_spikes',array_name)) = binned_spikes;
end

function unit_guide = get_smile_unit_guide(smile_data,params)
    % function to extract unit guide from all trials of smile_data
    
    % set up params
    valid_sort_idx = 1:10;
    assignParams(who,params);

    % concatenate snippet info from all trials of smile_data
    TrialData = horzcat(smile_data.TrialData);
    for trialnum = 1:length(TrialData)
        TrialData(trialnum).TDT.timestampUnits = 'ms';
    end
    TDT = horzcat(TrialData.TDT);
    snippetInfo = horzcat(TDT.snippetInfo);

    % get unit guide
    spike_times = double(snippetInfo');
    if ~isempty(spike_times)
        unit_guide = unique(spike_times(:,1:2),'rows');
        keep_units = ismember(unit_guide(:,2),valid_sort_idx);
        unit_guide = unit_guide(keep_units,:);
    else
        unit_guide = [-1 -1];
    end
end

function trial = assign_aborts(in_trial,params)
    % function to reassign pre-go-cue failures to abort result ('A')
    trial = in_trial;
    if trial.result=='F' && isnan(trial.idx_goCueTime)
        trial.result='A';
    end
end

function equality = isclose(A,B)
    % function to compare floating point numbers given a tolerance
    
    tol = 1e-10; % tolerance for floating point comparisons
    equality = abs(A-B)<tol; 
end

function [Data,miss4] = errorCursorSaveFix(Data)
    %ERRORCURSORSAVEFIX Recover missing samples from successful CST trials
    %   Fix for data collected with MonkeyHost version 09.28.2017.
    %   Missing samples were dropped during sending to Host; these samples were
    %   still used during online CST control.
    %   Up to three consecutive missing samples can be recovered.
    %   Note: This function only interpolates for successful trials.
    %
    %   Arguments
    %   ---------
    %   Data : 1xN struct array
    %       Data structure.
    %
    %   Returns
    %   ---------
    %   Data : 1xN struct array
    %       Data structure with corrected CST data.
    %   miss4 : vector
    %       Trials with >=4 samples still missing.
    %
    %   Basic Usage::
    %       >> [Data,miss4] = errorCursorSaveFix(Data)
    %
    %   2018 Nicole McClain
    
    %% Extract successful CST trials
    trialTypes = arrayfun(@(data) {data.Overview.trialName}, Data);
    cstTrials = ~cellfun(@isempty,regexpi(trialTypes,'CST'));
    
    %Extract CST trial data
    trls = find(cstTrials & ~replayTrials); %find indices
    if isempty(trls)
        error('No CST trials in this session.')
    end
    cstData = Data(trls);
    
    %% Process CST trials
    miss4 = false(size(trls)); %set up to check for missing samples
    for i = 1:length(cstData)
        %% Extract known trial information
        ref = cstData(i).Parameters.ForceParameters.refValue;
        l = cstData(i).Parameters.ForceParameters.initialLambda;
        if isempty(cstData(i).TrialData.Marker.rawPositions)
            continue
        end
        pspX = cstData(i).TrialData.Marker.rawPositions(:,2); %x axis input
        stateNames = [cstData(i).Parameters.StateTable.stateName];
        cstState = find(strcmp(stateNames,'Control System'));
        stateTrans = cstData(i).TrialData.stateTransitions;
        %(State time)-1 because Samples occur at mod(trialClock-1,10) == 0
        cstStartT = double(stateTrans(2,stateTrans(1,:)==cstState))-1;
        if isempty(cstStartT)
            continue
        end
        erc = cstData(i).TrialData.Marker.errorCursor;
        %Handle translation issues (e.g. double-sized or multiple z-values)
        erc = erc(erc(:,3)==erc(1,3),:);
        [x,t,u] = deal(erc(:,1),erc(:,4),erc(:,6));
        idx = (t/10-ceil(cstStartT/10))+1; %sample index
        uPSP = [pspX; 0]-ref; %possible input values
        %Check if samples start at xFull(1) == ref
        if xor(idx(1)==1,isclose(x(1),ref))
            error('Indexing mismatch on trial %i',trls(i));
        end
        
        %% Preallocate CST data to number of samples given by sample index
        [xFull,tFull,uFull] = deal(NaN(idx(end),1));
        [xFull(idx),tFull(idx),uFull(idx)] = deal(x,t,u);
        
        %% Calculate missing first sample
        if (idx(1)-1) == 1 %one missing sample
            uPoss = [uPSP(1:find(isclose(uPSP,u(1)))); 0];
            if length(uPoss)==1; uPoss = uPSP; end
            xPoss = 0+(exp(l*0.01)-1)*uPoss;
            if ~any(isclose(xPoss,x(1))) %check for rounding issues
                error('No matching output for trial %i',trls(i));
            end
            xFull(1) = 0;
            uFull(1) = uPoss(find(isclose(xPoss,x(1)),1));
            tFull(1) = ceil(cstStartT/10)*10;
        elseif (idx(1)-1) == 2 %two missing samples
            uPoss = [uPSP(1:find(isclose(uPSP,u(1)))); 0];
            if length(uPoss)==1; uPoss = uPSP; end
            xPoss = 0+(exp(l*0.01)-1)*uPoss;
            [xPoss1,uPossG] = meshgrid(xPoss,uPoss);
            xPoss2 = (exp(l*0.01)*xPoss1+(exp(l*0.01)-1)*uPossG);
            if ~any(isclose(xPoss2,x(1))) %check for rounding issues
                error('No matching output for trial %i',trls(i))
            end
            xFull(2) = xPoss1(find(isclose(xPoss2,x(1)),1));
            uFull(2) = uPossG(find(isclose(xPoss2,x(1)),1));
            xFull(1) = 0;
            uFull(1) = uPoss(find(isclose(xPoss,xFull(2)),1));
            tFull(1:2) = ceil(cstStartT/10)*10+([0 10]);
        elseif (idx(1)-1) == 3 %three missing samples
            uPoss = [uPSP(1:find(isclose(uPSP,u(1)))); 0];
            if length(uPoss)==1; uPoss = uPSP; end
            xPoss = 0+(exp(l*0.01)-1)*uPoss;
            [xPoss1,uPossG] = meshgrid(xPoss,uPoss);
            xPoss2 = (exp(l*0.01)*xPoss1+(exp(l*0.01)-1)*uPossG);
            [xPoss2G,uPossG3] = meshgrid(xPoss2,uPoss);
            xPoss3 = (exp(l*0.01)*xPoss2G+(exp(l*0.01)-1)*uPossG3);
            if ~any(sum(sum(isclose(xPoss3,x(1))))) %check for rounding issues
                error('No matching output for trial %i',trls(i))
            end
            xFull(3) = xPoss2G(find(isclose(xPoss3,x(1)),1));
            uFull(3) = uPossG3(find(isclose(xPoss3,x(1)),1));
            xFull(2) = xPoss1(find(isclose(xPoss2,xFull(3)),1));
            uFull(2) = uPossG(find(isclose(xPoss2,xFull(3)),1));
            xFull(1) = 0;
            uFull(1) = uPossG(find(isclose(xPoss1,xFull(2)),1));
            tFull(1:3) = ceil(cstStartT/10)*10+([0 10 20]);
        elseif (idx(1)-1) > 3 %too many missing samples to fix
            miss4(i) = true;
        end
        
        %% Fill in missing samples
        nMiss = diff(idx)-1; %number of missing samples
        for j = 1:length(nMiss)
            %check potential output for all
            if nMiss(j) == 1 %simple case, one missing sample
                uPoss = [uPSP(find(isclose(uPSP,u(j))):find(isclose(uPSP,u(j+1)))); 0];
                if length(uPoss)==1; uPoss = uPSP; end
                xNext = exp(l*0.01)*x(j)+(exp(l*0.01)-1)*u(j);
                xPoss = (exp(l*0.01)*xNext+(exp(l*0.01)-1)*uPoss);
                xFull(idx(j)+1) = xNext;
                if ~any(isclose(xPoss,x(j+1))) %check for rounding issues
                    error('No matching output for trial %i',trls(i))
                end
                uFull(idx(j)+1) = uPoss(find(isclose(xPoss,x(j+1)),1));
                tFull(idx(j)+1) = t(j)+10;
            elseif nMiss(j) == 2 %two missing samples
                uPoss = [uPSP(find(isclose(uPSP,u(j))):find(isclose(uPSP,u(j+1)))); 0];
                if length(uPoss)==1; uPoss = uPSP; end
                xNext = exp(l*0.01)*x(j)+(exp(l*0.01)-1)*u(j);
                xPoss1 = (exp(l*0.01)*xNext+(exp(l*0.01)-1)*uPoss);
                [xPoss1,uPossG] = meshgrid(xPoss1,uPoss);
                xPoss2 = (exp(l*0.01)*xPoss1+(exp(l*0.01)-1)*uPossG);
                xFull(idx(j)+1) = xNext;
                if ~any(sum(isclose(xPoss2,x(j+1)))) %check for rounding issues
                    error('No matching output for trial %i',trls(i));
                end
                uFull(idx(j)+2) = uPossG(find(isclose(xPoss2,x(j+1)),1));
                xFull(idx(j)+2) = xPoss1(find(isclose(xPoss2,x(j+1)),1));
                uFull(idx(j)+1) = uPossG(find(isclose(xPoss1,xFull(idx(j)+2)),1));
                tFull(idx(j)+(1:2)) = t(j)+10*(1:2);
            elseif nMiss(j) == 3
                uPoss = [uPSP(find(isclose(uPSP,u(j))):find(isclose(uPSP,u(j+1)))); 0];
                if length(uPoss)==1; uPoss = uPSP; end
                xNext = exp(l*0.01)*x(j)+(exp(l*0.01)-1)*u(j);
                xPoss1 = (exp(l*0.01)*xNext+(exp(l*0.01)-1)*uPoss);
                [xPoss1,uPossG] = meshgrid(xPoss1,uPoss);
                xPoss2 = (exp(l*0.01)*xPoss1+(exp(l*0.01)-1)*uPossG);
                [xPoss2G,uPossG3] = meshgrid(xPoss2,uPoss);
                xPoss3 = (exp(l*0.01)*xPoss2G+(exp(l*0.01)-1)*uPossG3);
                xFull(idx(j)+1) = xNext;
                if ~any(sum(sum(isclose(xPoss3,x(j+1))))) %check for rounding issues
                    error('No matching output for trial %i',trls(i))
                end
                xFull(idx(j)+3) = xPoss2G(find(isclose(xPoss3,x(j+1)),1));
                uFull(idx(j)+3) = uPossG3(find(isclose(xPoss3,x(j+1)),1));
                xFull(idx(j)+2) = xPoss1(find(isclose(xPoss2,xFull(idx(j)+3)),1));
                uFull(idx(j)+2) = uPossG(find(isclose(xPoss2,xFull(idx(j)+3)),1));
                uFull(idx(j)+1) = uPossG(find(isclose(xPoss1,xFull(idx(j)+2)),1));
                tFull(idx(j)+(1:3)) = t(j)+10*(1:3);
            elseif nMiss(j) > 3 %too many missing samples to fix
                miss4(i) = true;
            end
        end
        
        %% Check for additional missing samples
        if any(isnan(xFull)) && ~miss4(i)
            error('Samples still missing on trial %i',trls(i))
        end
        
        %% Create full error cursor data array
        newErc = [xFull,ones(idx(end),1)*erc(1,2:3),tFull,ones(idx(end),1)*l,uFull];
        cstData(i).TrialData.Marker.errorCursor = newErc;
    end
    
    %% Report trials with missing samples
    miss4 = trls(miss4);
    if ~isempty(miss4)
        warning('More than three samples missing on trials\n%s',...
            sprintf('%i ',miss4))
    end
    
    %% Add processed data back into structure
    Data(trls) = cstData;
    
    end