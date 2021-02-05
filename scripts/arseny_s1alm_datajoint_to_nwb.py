import os
import sys
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np
import json
import datajoint as dj
import warnings
import pathlib
from collections import OrderedDict

import pynwb
from pynwb import NWBFile, NWBHDF5IO
from ndx_events import LabeledEvents


warnings.filterwarnings('ignore', module='pynwb')

cf = dj.create_virtual_module('cf', 'arseny_cf')
lab = dj.create_virtual_module('lab', 'map_lab')
experiment = dj.create_virtual_module('experiment', 'arseny_s1alm_experiment')
ephys = dj.create_virtual_module('ephys', 'arseny_s1alm_ephys')
misc = dj.create_virtual_module('misc', 'arseny_s1alm_misc')


# ============================== SET CONSTANTS ==========================================
default_nwb_output_dir = os.path.join('.', 'NWB 2.0')
zero_zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available
hardware_filter = 'Bandpass filtered 300-6K Hz'
institution = 'Janelia Research Campus'

study_description = dict(
    related_publications='n/a',
    experiment_description='Extracellular electrophysiology recordings in anterior lateral motor cortex and in vibrissal sensory cortex in mice trained to detect optogenetic stimulation of the vibrissal sensory cortex',
    keywords=['decision-making', 'motor cortex', 'optogenetic stimulation', 'extracellular electrophysiology', 'attractor'])


def export_to_nwb(session_key, nwb_output_dir=default_nwb_output_dir, save=False, overwrite=False):

    this_session = (experiment.Session * experiment.SessionComment & session_key).fetch1()

    print(f'Exporting to NWB 2.0 for session: {session_key}...')
    # ===============================================================================
    # ============================== META INFORMATION ===============================
    # ===============================================================================

    # -- NWB file - a NWB2.0 file for each session
    nwbfile = NWBFile(identifier=f'{this_session["subject_id"]}_session_{this_session["session"]}',
                      session_description=json.dumps((experiment.SessionTask * experiment.Task & session_key).fetch1()),
                      session_start_time=datetime.combine(this_session['session_date'], zero_zero_time),
                      file_create_date=datetime.now(tzlocal()),
                      experimenter=this_session['username'],
                      data_collection=this_session['session_comment'],
                      institution=institution,
                      experiment_description=study_description['experiment_description'],
                      related_publications=study_description['related_publications'],
                      keywords=study_description['keywords'])

    # -- subject
    subj = (lab.Subject & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subj['subject_id']),
        description=f'source: {subj["animal_source"]}; cage_number: {subj["cage_number"]}',
        genotype=' x '.join((lab.Subject.GeneModification & subj).fetch('gene_modification')),
        sex=subj['sex'],
        species='Mus musculus',
        date_of_birth=datetime.combine(subj['date_of_birth'], zero_zero_time) if subj['date_of_birth'] else None)

    # ===============================================================================
    # ======================== EXTRACELLULAR & CLUSTERING ===========================
    # ===============================================================================

    """
    In the event of multiple probe recording (i.e. multiple probe insertions), the clustering results 
    (and the associated units) are associated with the corresponding probe. 
    Each probe insertion is associated with one ElectrodeConfiguration (which may define multiple electrode groups)
    """

    # retrieve trial start/stop times
    trial_times = []
    no_spikes_trials = []
    for trial_key in (experiment.SessionTrial & session_key).fetch('KEY', order_by='trial'):
        ephys_start = float((experiment.BehaviorTrial.Event & 'trial_event_type = "trigger ephys rec."'
                             & trial_key).fetch1('trial_event_time'))
        # trial_start: previous trial_stop + ephys_start (as inter-trial-interval)
        # so on a per-trial basis, ephys_start IS the trial's start time
        trial_start = trial_times[-1][-1] + ephys_start if len(trial_times) else 0.0
        # trial_stop: last spike
        spks = (ephys.TrialSpikes & trial_key).fetch('spike_times')
        last_spikes = [spk[-1] for spk in spks if len(spk)]
        if len(last_spikes):
            trial_stop = np.max(last_spikes) - ephys_start + trial_start
        else:
            if len(trial_times) == 0:
                # the case of very first trial having no spikes, hard-coding the duration to be 12 seconds
                mean_dur = 12
            else:
                mean_dur = np.mean([stop - start for _, start, stop in trial_times])

            trial_stop = trial_start + mean_dur
            no_spikes_trials.append(trial_key['trial'])
        trial_times.append((trial_key['trial'], trial_start, trial_stop))

    trial_times = {tr: (start, stop) for tr, start, stop in trial_times}
    obs_intervals = [(start, stop) for tr, (start, stop) in trial_times.items() if tr not in no_spikes_trials]

    # add new Units columns
    # --- unit spike times ---
    nwbfile.add_unit_column(name='sampling_rate', description='Sampling rate of the raw voltage traces (Hz)')
    nwbfile.add_unit_column(name='quality', description='unit quality from clustering')
    nwbfile.add_unit_column(name='cell_type', description='cell type')
    nwbfile.add_unit_column(name='waveform_amplitude', description='unit amplitude (peak) in microvolts')
    nwbfile.add_unit_column(name='spk_width_ms', description='unit average spike width, in ms')
    nwbfile.add_unit_column(name='unit_ml_location', description='um from ref; right is positive; based on manipulator coordinates (or histology) & probe config')
    nwbfile.add_unit_column(name='unit_ap_location', description='um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config')
    nwbfile.add_unit_column(name='unit_dv_location', description='um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config')

    for egroup_key in ephys.ElectrodeGroup & session_key:
        egroup = (ephys.Probe * ephys.ElectrodeGroup & egroup_key).aggr(
            ephys.Unit.Position, ..., brain_area='GROUP_CONCAT(DISTINCT brain_area)',
            hemi='GROUP_CONCAT(DISTINCT hemisphere)').fetch1()
        insert_location = {k: str(v) for k, v in (ephys.ElectrodeGroup.Position & egroup_key).fetch1().items()
                           if k not in ephys.ElectrodeGroup.Position.primary_key}
        insert_location = {**insert_location, 'brain_area': egroup['brain_area'], 'hemisphere': egroup['hemi']}

        ephys_device_name = f'{egroup["probe_type"]}_{egroup["probe_part_no"]}'
        ephys_device = (nwbfile.get_device(ephys_device_name)
                        if ephys_device_name in nwbfile.devices
                        else nwbfile.create_device(name=ephys_device_name))
        electrode_group = nwbfile.create_electrode_group(
            name=str(egroup['electrode_group']), description='N/A', device=ephys_device,
            location=json.dumps(insert_location))

        for chn in np.arange(1, 33):
            nwbfile.add_electrode(id=chn, group=electrode_group, filtering=hardware_filter, imp=-1.,
                                  x=np.nan, y=np.nan, z=np.nan,
                                  location=electrode_group.location)

        # add units
        for unit_key in (ephys.Unit & egroup_key).fetch('KEY'):
            unit = (ephys.UnitCellType * ephys.Unit * ephys.Unit.Waveform * ephys.Unit.Position & unit_key).fetch1()
            # concatenate unit's spiketimes from all trials
            trials, ephys_starts, unit_spiketimes = ((experiment.BehaviorTrial.Event
                                                      & 'trial_event_type = "trigger ephys rec."')
                                                     * ephys.TrialSpikes & unit_key).fetch(
                'trial', 'trial_event_time', 'spike_times', order_by='trial')

            unit_spiketimes = [spks.flatten() - ephys_start + trial_times[tr][0]
                               for tr, spks, ephys_start in zip(trials, unit_spiketimes, ephys_starts.astype(float))]

            # make an electrode table region (which electrode(s) is this unit coming from)
            electrode_df = nwbfile.electrodes.to_dataframe()
            electrode_ind = electrode_df.index[electrode_df.group_name == electrode_group.name]
            nwbfile.add_unit(id=unit['unit'],
                             electrodes=np.where(electrode_ind == int(unit['unit_channel']))[0],
                             electrode_group=electrode_group,
                             obs_intervals=obs_intervals,
                             sampling_rate=unit['sampling_fq'],
                             quality=unit['unit_quality'],
                             cell_type=unit['cell_type'],
                             spike_times=np.hstack(unit_spiketimes),
                             waveform_mean=unit['waveform'],
                             waveform_amplitude=unit['waveform_amplitude'],
                             spk_width_ms=unit['spk_width_ms'],
                             waveform_sd=np.full_like(unit['waveform'], np.nan),
                             unit_ml_location=float(unit['unit_ml_location']) if unit['unit_ml_location'] else np.nan,
                             unit_ap_location=float(unit['unit_ap_location']) if unit['unit_ap_location'] else np.nan,
                             unit_dv_location=float(unit['unit_dv_location']) if unit['unit_dv_location'] else np.nan)

    # ===============================================================================
    # ============================= PHOTO-STIMULATION ===============================
    # ===============================================================================
    stim_sites = {}
    for photostim_key in (experiment.Photostim & (experiment.PhotostimTrial.Event & session_key)).fetch('KEY'):
        photostim = (experiment.Photostim * experiment.PhotostimDevice & photostim_key).fetch1()
        stim_device = (nwbfile.get_device(photostim['photostim_device'])
                       if photostim['photostim_device'] in nwbfile.devices
                       else nwbfile.create_device(name=photostim['photostim_device']))

        stim_site = pynwb.ogen.OptogeneticStimulusSite(
            name=f'{photostim["photostim_device"]}_{photostim["photo_stim"]}',
            device=stim_device,
            excitation_lambda=float(photostim['excitation_wavelength']),
            location=json.dumps([{k: v for k, v in stim_locs.items()
                                  if k not in experiment.Photostim.primary_key}
                                 for stim_locs in (experiment.PhotostimLocation
                                                   & photostim_key).fetch(as_dict=True)], default=str),
            description=f'excitation_duration: {photostim["duration"]}')
        nwbfile.add_ogen_site(stim_site)
        stim_sites[photostim['photo_stim']] = stim_site

    # ===============================================================================
    # =============================== BEHAVIOR TRIALS ===============================
    # ===============================================================================

    # =============== TrialSet ====================
    # NWB 'trial' (of type dynamic table) by default comes with three mandatory attributes: 'start_time' and 'stop_time'
    # Other trial-related information needs to be added in to the trial-table as additional columns (with column name
    # and column description)
    q_photostim = ((experiment.BehaviorTrial.Event
                    & 'trial_event_type = "go"').proj(go_time='trial_event_time')
                   * experiment.PhotostimTrial.Event * experiment.Photostim.proj(stim_dur='duration')
                   & session_key).proj('stim_dur', stim_time='ROUND(go_time - photostim_event_time, 2)')
    q_trial = experiment.SessionTrial * experiment.BehaviorTrial * experiment.TrialName & session_key
    q_trial = q_trial.aggr(q_photostim, ..., photostim_onset='IFNULL(GROUP_CONCAT(stim_time SEPARATOR ", "), "N/A")',
                           photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
                           photostim_duration='IFNULL(GROUP_CONCAT(stim_dur SEPARATOR ", "), "N/A")', keep_all_rows=True)

    skip_adding_columns = experiment.Session.primary_key + ['trial_uid', 'trial']

    if q_trial:
        # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {tag: {'name': tag,
                               'description': q_trial.heading.attributes[tag].comment}
                         for tag in q_trial.heading.names
                         if tag not in skip_adding_columns + ['start_time', 'stop_time']}
        # Photostim labels from misc.S1PhotostimTrial
        trial_columns.update({'photostim_' + tag: {'name': 'photostim_' + tag,
                                    'description': misc.S1PhotostimTrial.heading.attributes[tag].comment}
                              for tag in ('onset', 'power', 'duration')})

        # Add new table columns to nwb trial-table for trial-label
        for c in trial_columns.values():
            nwbfile.add_trial_column(**c)

        # Add entry to the trial-table
        for trial in q_trial.fetch(as_dict=True):
            trial['start_time'], trial['stop_time'] = trial_times[trial['trial']]
            trial['id'] = trial['trial']  # rename 'trial_id' to 'id'
            [trial.pop(k) for k in skip_adding_columns]
            nwbfile.add_trial(**trial)

    # ===============================================================================
    # =============================== TRIAL EVENTS ==========================
    # ===============================================================================

    event_times, event_label_ind = [], []
    event_labels = OrderedDict()

    # ---- behavior events ----
    q_ephys_event = (experiment.BehaviorTrial.Event & 'trial_event_type = "trigger ephys rec."').proj(
        ephys_trigger='trial_event_type', ephys_start='trial_event_time')
    q_trial_event = (experiment.BehaviorTrial.Event * q_ephys_event
                     & 'trial_event_type NOT in ("send scheduled wave", "trigger ephys rec.", "trigger imaging")'
                     & session_key).proj(
        event_start='trial_event_time - ephys_start', event_stop='trial_event_time - ephys_start + duration')

    trials, event_types, event_starts, event_stops = q_trial_event.fetch(
        'trial', 'trial_event_type', 'event_start', 'event_stop', order_by='trial')
    trial_starts = [trial_times[tr][0] for tr in trials]
    event_starts = event_starts.astype(float) + trial_starts
    event_stops = event_stops.astype(float) + trial_starts

    for etype in set(event_types):
        event_labels[etype + '_start_times'] = len(event_labels)
        event_labels[etype + '_stop_times'] = len(event_labels)

    event_times.extend(event_starts)
    event_label_ind.extend([event_labels[etype + '_start_times'] for etype in event_types])
    event_times.extend(event_stops)
    event_label_ind.extend([event_labels[etype + '_stop_times'] for etype in event_types])

    # ---- action events ----
    q_action_event = (experiment.ActionEvent * q_ephys_event & session_key).proj(
        event_time='action_event_time - ephys_start')

    trials, event_types, event_starts = q_action_event.fetch(
        'trial', 'action_event_type', 'event_time', order_by='trial')
    trial_starts = [trial_times[tr][0] for tr in trials]
    event_starts = event_starts.astype(float) + trial_starts

    for etype in set(event_types):
        event_labels[etype] = len(event_labels)

    event_times.extend(event_starts)
    event_label_ind.extend([event_labels[etype] for etype in event_types])

    labeled_events = LabeledEvents(name='LabeledEvents',
                                   description='behavioral events of the experimental paradigm',
                                   timestamps=event_times,
                                   data=event_label_ind,
                                   labels=list(event_labels.keys()))
    nwbfile.add_acquisition(labeled_events)

    # ---- photostim events ----
    behav_event = pynwb.behavior.BehavioralEvents(name='PhotostimEvents')
    nwbfile.add_acquisition(behav_event)

    q_photostim_event = (experiment.Photostim.proj('duration') * experiment.PhotostimTrial.Event
                         * q_ephys_event & session_key).proj(
        event_start='photostim_event_time - ephys_start',
        event_stop='photostim_event_time - ephys_start + duration')

    trials, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
        'trial', 'event_start', 'event_stop', 'power', 'photo_stim', order_by='trial')
    trial_starts = [trial_times[tr][0] for tr in trials]

    behav_event.create_timeseries(name='photostim_start_times', unit='mW', conversion=1.0,
                                  description='Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied',
                                  data=powers.astype(float),
                                  timestamps=event_starts.astype(float) + trial_starts,
                                  control=photo_stim.astype('uint8'), control_description=stim_sites.values());
    behav_event.create_timeseries(name='photostim_stop_times', unit='mW', conversion=1.0,
                                  description = 'Timestamps of the photo-stimulation being switched off',
                                  data=np.full_like(event_starts.astype(float), 0),
                                  timestamps=event_stops.astype(float) + trial_starts,
                                  control=photo_stim.astype('uint8'), control_description=stim_sites.values());

    # =============== Write NWB 2.0 file ===============
    if save:
        save_file_name = ''.join([nwbfile.identifier, '.nwb'])
        if not os.path.exists(nwb_output_dir):
            os.makedirs(nwb_output_dir)
        output_fp = (pathlib.Path(nwb_output_dir) / save_file_name).absolute()
        if not overwrite and output_fp.exists():
            return nwbfile
        with NWBHDF5IO(output_fp.as_posix(), mode='w') as io:
            io.write(nwbfile)
            print(f'Write NWB 2.0 file: {save_file_name}')

    return nwbfile


# ============================== EXPORT ALL ==========================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        nwb_outdir = sys.argv[1]
    else:
        nwb_outdir = default_nwb_output_dir

    for skey in (experiment.Session & ephys.Unit).fetch('KEY'):
        save_file_name = f'{skey["subject_id"]}_session_{skey["session"]}.nwb'
        output_fp = (pathlib.Path(nwb_outdir) / save_file_name).absolute()
        if not output_fp.exists():
            try:
                export_to_nwb(skey, nwb_output_dir=nwb_outdir, save=True)
            except Exception as e:
                print(f'Session: {skey}\n{str(e)}')
