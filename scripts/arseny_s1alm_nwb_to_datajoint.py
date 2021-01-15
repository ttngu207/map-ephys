import pynwb
import pathlib
import numpy as np
from pynwb import NWBHDF5IO
from tqdm import tqdm
import sys
import datajoint as dj
import re
import json
import uuid

db_prefix = dj.config['custom']['database.prefix']

cf = dj.create_virtual_module('cf', db_prefix + 'cf')
lab = dj.create_virtual_module('lab', db_prefix + 'lab')
experiment = dj.create_virtual_module('experiment', db_prefix + 'experiment')
ephys = dj.create_virtual_module('ephys', db_prefix + 'ephys')
misc = dj.create_virtual_module('misc', db_prefix + 'misc')

photostim_dict = {p['photo_stim']: p for p in experiment.Photostim.fetch('KEY')}
fiducials_type_dict = {(p['tracking_device_id'], p['video_fiducial_name']): p
                       for p in experiment.VideoFiducialsType.fetch('KEY')}


def ingest_to_pipeline(nwb_filepath):
    io = NWBHDF5IO(nwb_filepath.as_posix(), mode='r')
    nwbfile = io.read()
    subject_key = {'subject_id': nwbfile.subject.subject_id}

    # =============================== SUBJECT ===========================
    if subject_key not in lab.Subject.proj():
        # lab.Subject
        subject = {**subject_key,
                   'cage_number': int(re.search('cage_number: (\d+)', nwbfile.subject.description).groups()[0]),
                   'username': nwbfile.experimenter[0],
                   'date_of_birth': nwbfile.subject.date_of_birth,
                   'sex': nwbfile.subject.sex,
                   'animal_source': re.search('source: (\w+);', nwbfile.subject.description).groups()[0]}
        # lab.Subject.GeneModification
        gene_modifications = nwbfile.subject.genotype.split(' x ')

        lab.Subject.insert1(subject)
        lab.Subject.GeneModification.insert([{**subject_key, 'gene_modification': gene_mod}
                                             for gene_mod in gene_modifications if gene_mod])

    # =============================== SESSION ===========================
    session_key = {**subject_key, 'session': int(nwbfile.identifier.split('_')[-1])}
    experiment.Session.insert1({**session_key, 'session_date': nwbfile.session_start_time.date(),
                                'username': nwbfile.experimenter[0], 'rig': 'ephys'})
    experiment.SessionComment.insert1({**session_key, 'session_comment': nwbfile.data_collection})

    # ======================== EXTRACELLULAR & CLUSTERING ===========================
    # Electrode Group
    electrode_group_keys = {}
    for egroup_no, egroup in nwbfile.electrode_groups.items():
        probe_type, probe_part_no = egroup.device.name.split('_')
        egroup_key = {**session_key, 'electrode_group': int(egroup_no)}
        electrode_group_keys[int(egroup_no)] = egroup_key
        insert_location = {k: None if v == 'None' else v
                           for k, v in json.loads(egroup.location).items()}

        ephys.Probe.insert1({'probe_part_no': probe_part_no,
                             'probe_type': probe_type, 'probe_comment': ''}, skip_duplicates=True)
        ephys.ElectrodeGroup.insert1({**egroup_key,
                                      'probe_part_no': probe_part_no})
        ephys.ElectrodeGroup.Position.insert1({**egroup_key, **insert_location,
                                               'cf_annotation_type': 'manipulator'}, ignore_extra_fields=True)

    # Unit
    units_df = nwbfile.units.to_dataframe()
    trials_df = nwbfile.trials.to_dataframe()

    unit_list, unit_celltype_list, unit_position_list = [], [], []
    unit_spikes_list, unit_wf_list, trialspikes_list = [], [], []
    for unit_id, unit in units_df.iterrows():
        unit_key = {**electrode_group_keys[int(unit.electrode_group.name)], 'unit': unit_id}
        unit_list.append({**unit_key, 'unit_uid': uuid.uuid4(), 'unit_quality': unit.quality,
                          'unit_channel': unit.electrodes.index[0]})
        unit_celltype_list.append({**unit_key, 'cell_type': unit.cell_type})
        unit_spikes_list.append({**unit_key, 'spike_times': unit.spike_times})
        unit_position_list.append({**unit_key, 'cf_annotation_type': 'manipulator',
                                   'hemisphere': insert_location['hemisphere'],
                                   'brain_area': insert_location['brain_area'],
                                   'skull_reference': insert_location['skull_reference'],
                                   'unit_ml_location': unit.unit_ml_location,
                                   'unit_ap_location': unit.unit_ap_location,
                                   'unit_dv_location': unit.unit_dv_location})
        unit_wf_list.append({**unit_key, 'waveform': unit.waveform_mean.flatten(),
                             'spk_width_ms': unit.spk_width_ms,
                             'sampling_fq': unit.sampling_rate,
                             'waveform_amplitude': unit.waveform_amplitude})
        # trial-spikes
        for trial_id, start_time, stop_time in zip(trials_df.index, trials_df.start_time, trials_df.stop_time):
            spks = unit.spike_times[np.logical_and(unit.spike_times >= start_time, unit.spike_times < stop_time)]
            trialspikes_list.append({**unit_key, 'trial': trial_id, 'spike_times': spks - start_time})

    ephys.Unit.insert(unit_list)
    ephys.UnitCellType.insert(unit_celltype_list)
    ephys.Unit.Position.insert(unit_position_list)
    ephys.Unit.Spikes.insert(unit_spikes_list)
    ephys.Unit.Waveform.insert(unit_wf_list)

    # =============================== BEHAVIOR TRIALS ===============================
    # trials
    session_trial_list, behavior_trial_list, trial_name_list, photostim_trial_list = [], [], [], []
    photostim_event_list, behavior_event_list = [], []
    trial_go_times = {}  # times of go-cue for each trial (relative to trial's start)
    for trial_id, trial in trials_df.iterrows():
        # trials
        trial_key = {**session_key, 'trial': trial_id}
        session_trial_list.append({**trial_key, 'start_time': trial.start_time})
        behavior_trial_list.append({**trial_key, 'task': trial.task,
                                    'task_protocol': trial.task_protocol,
                                    'trial_instruction': trial.trial_instruction,
                                    'early_lick': trial.early_lick,
                                    'outcome': trial.outcome})
        trial_name_list.append({**trial_key, 'task': trial.task, 'trial_type_name': trial.trial_type_name})
        if trial.photostim_duration == 'N/A' and trial.photostim_power == 'N/A' and trial.photostim_duration == 'N/A':
            photostim_trial_list.append({**trial_key})

        # trials' events
        event_names = set([re.sub('_start_times|_stop_times', '', e_ts.name)
                           for e_ts in nwbfile.acquisition['BehavioralEvents'].children])
        for event_type in event_names:
            start_times = nwbfile.acquisition['BehavioralEvents'][event_type + '_start_times'].timestamps[()]
            valid_trial_ind = np.where(np.logical_and(start_times >= trial.start_time,
                           start_times < trial.stop_time))[0]
            durations = nwbfile.acquisition['BehavioralEvents'][event_type + '_stop_times'].timestamps[valid_trial_ind] - start_times[valid_trial_ind]

            if event_type == 'photostim':
                powers = nwbfile.acquisition['BehavioralEvents'][event_type + '_start_times'].data[valid_trial_ind]
                stim_ind = nwbfile.acquisition['BehavioralEvents'][event_type + '_start_times'].control[valid_trial_ind]
                photostim_event_list.extend([{**trial_key, 'photostim_event_time': start_time - start_time,
                                              'power': power, **photostim_dict[stim_id]}
                                            for start_time, power, stim_id in zip(start_times[valid_trial_ind], powers, stim_ind)])
            else:
                behavior_event_list.extend([{**trial_key, 'trial_event_type': event_type,
                                             'trial_event_time': start_time - start_time, 'duration': dur}
                                            for start_time, dur in zip(start_times[valid_trial_ind], durations)])
                if event_type == 'go':
                    trial_go_times[trial_id] = start_times[valid_trial_ind][0] - start_time

        # trials' video fiducials data
        if 'BehavioralTimeSeries' in nwbfile.acquisition:
            for behav_ts in nwbfile.acquisition['BehavioralTimeSeries'].children:
                trk_device_id, fiducial_name = behav_ts.name.replace('Camera', '').split('_')
                fiducial_type_key = fiducials_type_dict[(int(trk_device_id), fiducial_name)]
                valid_trial_ind = np.where(np.logical_and(behav_ts.timestamps[()] >= trial.start_time,
                           behav_ts.timestamps[()] < trial.stop_time))[0]

                fiducial_time = behav_ts.timestamps[valid_trial_ind] - trial_go_times[trial_id]
                fiducial_x_position, fiducial_y_position, fiducial_p = behav_ts.data[valid_trial_ind, :].T
                


    experiment.SessionTrial.insert(session_trial_list)
    experiment.BehaviorTrial.insert(behavior_trial_list)
    experiment.TrialName.insert(trial_name_list)
    experiment.PhotostimTrial.insert(photostim_trial_list)
    experiment.BehaviorTrial.Event.insert(behavior_event_list)
    experiment.PhotostimTrial.Event.insert(photostim_event_list)
    ephys.TrialSpikes.insert(trialspikes_list)

    # ============================= BEHAVIOR TRACKING ===============================

    if 'BehavioralTimeSeries' in nwbfile.acquisition:

        for

# ============================== INGEST ALL NWB FILES ==========================================


if __name__ == '__main__':
    if len(sys.argv) > 1:
        nwb_dir = pathlib.Path(sys.argv[1])
    else:
        print('Usage error, please specify the directory to the NWB files')
        sys.exit(0)

    if not nwb_dir.exists():
        raise FileNotFoundError(f'NWB data directory not found: {nwb_dir}')

    for nwb_fp in tqdm(nwb_dir.glob('*.nwb')):
        ingest_to_pipeline(nwb_fp)
