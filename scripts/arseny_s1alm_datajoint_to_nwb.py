import os
import sys
from datetime import datetime
from dateutil.tz import tzlocal
import re
import numpy as np
import json
import pandas as pd
import datajoint as dj
import warnings

import pynwb
from pynwb import NWBFile, NWBHDF5IO

warnings.filterwarnings('ignore', module='pynwb')

cf = dj.create_virtual_module('cf', 'arseny_cf')
lab = dj.create_virtual_module('lab', 'map_lab')
experiment = dj.create_virtual_module('experiment', 'arseny_s1alm_experiment')
ephys = dj.create_virtual_module('ephys', 'arseny_s1alm_ephys')
misc = dj.create_virtual_module('misc', 'arseny_s1alm_misc')


# ============================== SET CONSTANTS ==========================================
default_nwb_output_dir = os.path.join('data', 'NWB 2.0')
zero_zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available
hardware_filter = 'Bandpass filtered 300-6K Hz'
ecephys_fs = 19531.25
institution = 'Janelia Research Campus'

study_description = dict(
    related_publications='doi:10.1038/nature17643',
    experiment_description='Extracellular electrophysiology recordings with optogenetic perturbations performed on anterior lateral region of the mouse cortex during object location discrimination task',
    keywords=['motor planning', 'premotor cortex', 'whiskers',
                'optogenetic perturbations', 'extracellular electrophysiology'])

unit_quality_mapper = {0: 'multi-unit', 1: 'likely single-unit', 2: 'single-unit'}


def export_to_nwb(session_key, nwb_output_dir=default_nwb_output_dir, save=False, overwrite=False):

    this_session = (experiment.Session * experiment.SessionComment & session_key).fetch1()

    print(f'Exporting to NWB 2.0 for session: {session_key}...')
    # ===============================================================================
    # ============================== META INFORMATION ===============================
    # ===============================================================================

    # -- NWB file - a NWB2.0 file for each session
    nwbfile = NWBFile(identifier=f'{this_session["animal_id"]}_session_{this_session["session_id"]}',
                      session_description='',
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
        species='mouse',  # TODO: verify
        date_of_birth=datetime.combine(subj['date_of_birth'], zero_zero_time) if subj['date_of_birth'] else None)

    # ===============================================================================
    # ======================== EXTRACELLULAR & CLUSTERING ===========================
    # ===============================================================================

    """
    In the event of multiple probe recording (i.e. multiple probe insertions), the clustering results 
    (and the associated units) are associated with the corresponding probe. 
    Each probe insertion is associated with one ElectrodeConfiguration (which may define multiple electrode groups)
    """

    for egroup_key in ephys.ElectrodeGroup & session_key:
        egroup = (ephys.Probe * ephys.ElectrodeGroup & egroup_key).fetch1()

        electrode_group = nwbfile.create_electrode_group(
            name=egroup['electrode_group'],
            description='N/A',
            device=nwbfile.create_device(name=f'{egroup["probe_type"]}_{egroup["probe_part_no"]}'),
            location=json.dumps({k: str(v) for k, v in (ephys.ElectrodeGroup.Position & egroup_key).fetch1().items()
                                 if k not in ephys.ElectrodeGroup.Position.primary_key}))

        for chn in np.arange(1, 32):
            nwbfile.add_electrode(id=chn, group=electrode_group, filtering=hardware_filter, imp=-1.,
                                  x=np.nan, y=np.nan, z=np.nan,
                                  location=electrode_group.location)

        # --- unit spike times ---
        nwbfile.add_unit_column(name='sampling_rate', description='Sampling rate of the raw voltage traces (Hz)')
        nwbfile.add_unit_column(name='quality', description='unit quality from clustering')
        nwbfile.add_unit_column(name='cell_type', description='cell type')
        nwbfile.add_unit_column(name='waveform_amplitude', description='unit amplitude (peak) in microvolts')
        nwbfile.add_unit_column(name='spk_width_ms', description='unit average spike width, in ms')
        nwbfile.add_unit_column(name='unit_ml_location', description='um from ref; right is positive; based on manipulator coordinates (or histology) & probe config')
        nwbfile.add_unit_column(name='unit_ap_location', description='um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config')
        nwbfile.add_unit_column(name='unit_dv_location', description='um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config')

        for unit_key in (ephys.Unit & egroup_key).fetch('KEY'):
            unit = (ephys.UnitCellType * ephys.Unit * ephys.Unit.Waveform
                    * ephys.Unit.Spikes * ephys.Unit.Position & unit_key).fetch1()

            # concatenate unit's spiketimes from all trials and build observation intervals
            obs_start, unit_spiketimes, go_times = (
                    experiment.SessionTrial.proj('start_time') * ephys.TrialSpikes
                    * (experiment.BehaviorTrial.Event & 'trial_event_type = "go"') & unit_key).fetch(
                'start_time', 'spike_times', 'trial_event_time', order_by='start_time')

            unit_spiketimes = [spks.flatten() - float(go_time) + float(start)
                               for start, spks, go_time in zip(obs_start, unit_spiketimes, go_times)]
            obs_stop = [s[-1] for s in unit_spiketimes]
            obs_intervals = [(float(start), stop) for start, stop in zip(obs_start, obs_stop)]

            # make an electrode table region (which electrode(s) is this unit coming from)
            nwbfile.add_unit(id=unit['unit_id'],
                             electrodes=np.where(np.array(nwbfile.electrodes.id.data) == int(unit['unit_channel']))[0],
                             electrode_group=electrode_group,
                             obs_intervals=obs_intervals,
                             sampling_rate=unit['sampling_fq'],
                             quality=unit['unit_quality'],
                             spike_times=np.hstack(unit_spiketimes),
                             waveform_mean=unit['waveform'],
                             waveform_amplitude=unit['waveform_amplitude'],
                             spk_width_ms=unit['spk_width_ms'],
                             waveform_sd=None,
                             unit_ml_location=unit['unit_ml_location'],
                             unit_ap_location=unit['unit_ap_location'],
                             unit_dv_location=unit['unit_dv_location'])

    # ===============================================================================
    # ============================= BEHAVIOR TRACKING ===============================
    # ===============================================================================

    if experiment.VideoFiducialsTrial & session_key:
        behav_acq = pynwb.behavior.BehavioralTimeSeries(name = 'BehavioralTimeSeries')
        nwbfile.add_acquisition(behav_acq)
        for fiducial_type in experiment.VideoFiducialsType.fetch('KEY'):
            trial_starts, go_times, fiducial_x, fiducial_y, fiducial_p, fiducial_times = (
                    experiment.SessionTrial.proj('start_time') * experiment.VideoFiducialsTrial
                    * (experiment.BehaviorTrial.Event & 'trial_event_type = "go"') & fiducial_type & session_key).fetch(
                'start_time', 'trial_event_time', 'fiducial_x_position', 'fiducial_y_position',
                'fiducial_p', 'fiducial_time', order_by='start_time')

            fiducial_times = [fiducial_t.flatten() - float(go_time) + float(start)
                              for start, fiducial_t, go_time in zip(trial_starts, fiducial_times, go_times)]
            fiducial_x = [d.flatten() for d in fiducial_x]
            fiducial_y = [d.flatten() for d in fiducial_y]
            fiducial_p = [d.flatten() for d in fiducial_p]

            behav_acq.create_timeseries(name=f'{fiducial_type["tracking_device_type"]}'
                                             f'{fiducial_type["tracking_device_id"]}_'
                                             f'{fiducial_type["video_fiducial_name"]}',
                                        unit='a.u.', conversion=1.0,
                                        data=np.vstack([np.hstack(fiducial_x), np.hstack(fiducial_y), np.hstack(fiducial_p)]),
                                        description=f'Time-series of the animal\'s {fiducial_type["video_fiducial_name"]} position (x, y, probability) - shape: (3 x times)',
                                        timestamps=np.hstack(fiducial_times))

    # ===============================================================================
    # ============================= PHOTO-STIMULATION ===============================
    # ===============================================================================
    stim_sites = {}
    for photostim_key in (experiment.Photostim & (experiment.PhotostimTrial & session_key)).fetch('KEY'):
        photostim = (experiment.Photostim * experiment.PhotostimDevice & photostim_key).fetch1()
        stim_device = (nwbfile.get_device(photostim['photostim_device'])
                       if photostim['photostim_device'] in nwbfile.devices
                       else nwbfile.create_device(name=photostim['photostim_device']))

        stim_site = pynwb.ogen.OptogeneticStimulusSite(
            name=photostim['photo_stim'],
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

    dj_trial = pipeline.SessionTrial * pipeline.BehaviorTrial
    skip_adding_columns = pipeline.Session.primary_key + ['trial_uid', 'trial']

    if pipeline.SessionTrial & session_key:
        # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = [{'name': tag,
                          'description': re.sub('\s+:|\s+', ' ', re.search(
                              f'(?<={tag})(.*)', str(dj_trial.heading)).group()).strip()}
                         for tag in dj_trial.heading.names
                         if tag not in skip_adding_columns + ['start_time', 'stop_time']]

        # Add new table columns to nwb trial-table for trial-label
        for c in trial_columns:
            nwbfile.add_trial_column(**c)

        # Add entry to the trial-table
        for trial in (dj_trial & session_key).fetch(as_dict=True):
            trial['start_time'] = float(trial['start_time'])
            trial['stop_time'] = float(trial['stop_time']) if trial['stop_time'] else np.nan
            trial['id'] = trial['trial']  # rename 'trial_id' to 'id'
            [trial.pop(k) for k in skip_adding_columns]
            nwbfile.add_trial(**trial)

    # ===============================================================================
    # =============================== TRIAL EVENTS ==========================
    # ===============================================================================

    behav_event = pynwb.behavior.BehavioralEvents(name='BehavioralEvents')
    nwbfile.add_acquisition(behav_event)

    for trial_event_type in (pipeline.TrialEventType & pipeline.TrialEvent & session_key).fetch('trial_event_type'):
        event_times, trial_starts = (pipeline.TrialEvent * pipeline.SessionTrial
                                     & session_key & {'trial_event_type': trial_event_type}).fetch(
            'trial_event_time', 'start_time')
        if len(event_times) > 0:
            event_times = np.hstack(event_times.astype(float) + trial_starts.astype(float))
            behav_event.create_timeseries(name=trial_event_type, unit='a.u.', conversion=1.0,
                                          data=np.full_like(event_times, 1),
                                          timestamps=event_times)

    photostim_event_time, trial_starts, photo_stim, power, duration = (
            pipeline.PhotostimEvent * pipeline.SessionTrial & session_key).fetch(
        'photostim_event_time', 'start_time', 'photo_stim', 'power', 'duration')

    if len(photostim_event_time) > 0:
        behav_event.create_timeseries(name='photostim_start_time', unit='a.u.', conversion=1.0,
                                      data=power,
                                      timestamps=photostim_event_time.astype(float) + trial_starts.astype(float),
                                      control=photo_stim.astype('uint8'), control_description=stim_sites)
        behav_event.create_timeseries(name='photostim_stop_time', unit='a.u.', conversion=1.0,
                                      data=np.full_like(photostim_event_time, 0),
                                      timestamps=photostim_event_time.astype(float) + duration.astype(float) + trial_starts.astype(float),
                                      control=photo_stim.astype('uint8'), control_description=stim_sites)

    # =============== Write NWB 2.0 file ===============
    if save:
        save_file_name = ''.join([nwbfile.identifier, '.nwb'])
        if not os.path.exists(nwb_output_dir):
            os.makedirs(nwb_output_dir)
        if not overwrite and os.path.exists(os.path.join(nwb_output_dir, save_file_name)):
            return nwbfile
        with NWBHDF5IO(os.path.join(nwb_output_dir, save_file_name), mode='w') as io:
            io.write(nwbfile)
            print(f'Write NWB 2.0 file: {save_file_name}')

    return nwbfile


# ============================== EXPORT ALL ==========================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        nwb_outdir = sys.argv[1]
    else:
        nwb_outdir = default_nwb_output_dir

    for skey in pipeline.Session.fetch('KEY'):
        export_to_nwb(skey, nwb_output_dir=nwb_outdir, save=True)