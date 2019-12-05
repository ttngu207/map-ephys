#! /usr/bin/env python

import os
import logging
import pathlib
from os import path

from glob import glob
import re
from itertools import repeat

import scipy.io as spio
import h5py
import numpy as np
import re

import datajoint as dj
#import pdb

from pipeline import lab
from pipeline import experiment
from pipeline import ephys
from pipeline import InsertBuffer, dict_to_hash
from pipeline.ingest import behavior as behavior_ingest
from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_ephys'))

log = logging.getLogger(__name__)


@schema
class EphysDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    data_path:              varchar(255)                # rig data path
    ---
    search_order:           int                         # rig search order
    """

    @property
    def contents(self):
        if 'ephys_data_paths' in dj.config['custom']:  # for local testing
            return dj.config['custom']['ephys_data_paths']

        return ((r'H:\\data\MAP', 0),)


@schema
class EphysIngest(dj.Imported):
    # subpaths like: \2017-10-21\tw5ap_imec3_opt3_jrc.mat

    definition = """
        -> experiment.Session
    """

    class EphysFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> EphysIngest
        probe_insertion_number:         tinyint         # electrode_group
        ephys_file:                     varchar(255)    # rig file subpath
        """

    key_source = experiment.Session - ephys.ProbeInsertion

    def make(self, key):
        '''
        Ephys .make() function
        '''

        log.info('EphysIngest().make(): key: {k}'.format(k=key))

        #
        # Find Ephys Recording
        #
        key = (experiment.Session & key).fetch1()
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & key).fetch1()

        rigpath = EphysDataPath().fetch1('data_path')
        h2o = sinfo['water_restriction_number']
        date = key['session_date'].strftime('%Y%m%d')

        dpath = pathlib.Path(rigpath, h2o, date)
        dglob = '[0-9]/{}'  # probe directory pattern

        # npx ap.meta: '{}_*.imec.ap.meta'.format(h2o)
        npx_meta_files = list(dpath.glob(dglob.format('{}_*.ap.meta'.format(h2o))))
        if not npx_meta_files:
            log.warning('Error - no ap.meta files in {}. Skipping.'.format(dpath))
            return
        npx_metas = {f.parent.name: NeuropixelsMeta(f) for f in npx_meta_files}
        log.info('Found .ap.meta for probe(s): {}'.format(list(npx_metas.keys())))

        v3spec = '{}_*_jrc.mat'.format(h2o)
        # old v3spec = '{}_g0_*.imec.ap_imec3_opt3_jrc.mat'.format(h2o)
        v3files = {f.parent.name: (f, self._load_v3) for f in dpath.glob(dglob.format(v3spec))}

        v4spec = '{}_*.ap_res.mat'.format(h2o)
        # old v4spec = '{}_g0_*.imec?.ap_res.mat'.format(h2o)  # TODO v4ify
        v4files = {f.parent.name: (f, self._load_v4) for f in dpath.glob(dglob.format(v4spec))}

        overlap_v3v4 = np.intersect1d(list(v3files), list(v4files))
        if len(overlap_v3v4) > 0:
            log.warning('Error - both jrclust V3 and V4 exist for probe(s): {} - Search path: {}. Skipping.'.format(
                overlap_v3v4, dpath))
            return

        jrclust_files = {**v3files, **v4files}

        missing_metas = [p for p in jrclust_files if p not in npx_metas]
        if missing_metas:
            log.warning('Error - missing ap.meta for probe(s): {} - Search path: {}. Skipping.'.format(
                missing_metas, dpath))
            return

        for probe_no, (f, loader) in jrclust_files.items():
            self._load(loader(sinfo, rigpath, dpath, f.relative_to(dpath)), npx_metas[probe_no])

    def _load(self, data, npx_meta):

        sinfo = data['sinfo']
        rigpath = data['rigpath']
        ef_path = data['ef_path']
        probe = data['probe']
        skey = data['skey']
        method = data['method']
        hz = data['hz'] if data['hz'] else npx_meta.meta['imSampRate']
        spikes = data['spikes']
        spike_sites = data['spike_sites']
        units = data['units']
        unit_wav = data['unit_wav']
        unit_notes = data['unit_notes']
        unit_xpos = data['unit_xpos']
        unit_ypos = data['unit_ypos']
        unit_amp = data['unit_amp']
        unit_snr = data['unit_snr']
        vmax_unit_site = data['vmax_unit_site']
        trial_start = data['trial_start']
        trial_go = data['trial_go']
        sync_ephys = data['sync_ephys']
        sync_behav = data['sync_behav']
        trial_fix = data['trial_fix']

        # account for the buffer period before trial_start
        buffer_sample_count = npx_meta.meta['trgTTLMarginS'] * npx_meta.meta['imSampRate']
        trial_start = trial_start - np.round(buffer_sample_count).astype(int)

        # remove noise clusters
        units, spikes, spike_sites = (v[i] for v, i in zip(
            (units, spikes, spike_sites), repeat((units > 0))))

        # Determine trial (re)numbering for ephys:
        #
        # - if ephys & bitcode match, determine ephys-to-behavior trial shift
        #   when needed for different-length recordings
        # - otherwise, use trial number correction array (bf['trialNum'])

        sync_behav_start = np.where(sync_behav == sync_ephys[0])[0][0]
        sync_behav_range = sync_behav[sync_behav_start:][:len(sync_ephys)]

        if not np.all(np.equal(sync_ephys, sync_behav_range)):
            if trial_fix is not None:
                log.info('ephys/bitcode trial mismatch - fix using "trialNum"')
                trials = trial_fix[0]
            else:
                raise Exception('Bitcode Mismatch - Fix with "trialNum" not available')
        else:
            if len(sync_behav) < len(sync_ephys):
                start_behav = np.where(sync_behav[0] == sync_ephys)[0][0]
            elif len(sync_behav) > len(sync_ephys):
                start_behav = - np.where(sync_ephys[0] == sync_behav)[0][0]
            else:
                start_behav = 0
            trial_indices = np.arange(len(sync_behav_range)) - start_behav

            # mapping to the behav-trial numbering
            # "trials" here is just the 0-based indices of the behavioral trials
            behav_trials = (experiment.SessionTrial & skey).fetch('trial', order_by='trial')
            trials = behav_trials[trial_indices]

        # trialize the spikes & subtract go cue
        t, trial_spikes, trial_units = 0, [], []

        while t < len(trial_start) - 1:

            s0, s1 = trial_start[t], trial_start[t+1]

            trial_idx = np.where((spikes > s0) & (spikes < s1))

            trial_spikes.append(spikes[trial_idx] - trial_go[t])
            trial_units.append(units[trial_idx])

            t += 1

        # ... including the last trial
        trial_idx = np.where((spikes > s1))
        trial_spikes.append(spikes[trial_idx] - trial_go[t])
        trial_units.append(units[trial_idx])

        trial_spikes = np.array(trial_spikes)
        trial_units = np.array(trial_units)

        # convert spike data to seconds
        spikes = spikes / hz
        trial_start = trial_start / hz
        trial_spikes = trial_spikes / hz

        # build spike arrays
        unit_spikes = np.array([spikes[np.where(units == u)]
                                for u in set(units)]) - trial_start[0]

        unit_trial_spikes = np.array(
            [[trial_spikes[t][np.where(trial_units[t] == u)]
              for t in range(len(trials))] for u in set(units)])

        # create probe insertion records
        insertion_key = self._gen_probe_insert(sinfo, probe, npx_meta)

        electrode_keys = {c['electrode']: c for c in (lab.ElectrodeConfig.Electrode & insertion_key).fetch('KEY')}

        # insert Unit
        log.info('.. ephys.Unit')

        with InsertBuffer(ephys.Unit, 10, skip_duplicates=True,
                          allow_direct_insert=True) as ib:

            for i, u in enumerate(set(units)):

                ib.insert1({**skey, **insertion_key,
                            **electrode_keys[vmax_unit_site[i]],
                            'clustering_method': method,
                            'unit': u,
                            'unit_uid': u,
                            'unit_quality': unit_notes[i],
                            'unit_posx': unit_xpos[i],
                            'unit_posy': unit_ypos[i],
                            'unit_amp': unit_amp[i],
                            'unit_snr': unit_snr[i],
                            'spike_times': unit_spikes[i],
                            'waveform': unit_wav[i][0]})

                if ib.flush():
                    log.debug('.... {}'.format(u))

        # insert Unit.UnitTrial
        log.info('.. ephys.Unit.UnitTrial')

        with InsertBuffer(ephys.Unit.UnitTrial, 10000, skip_duplicates=True,
                          allow_direct_insert=True) as ib:

            for i, u in enumerate(set(units)):
                for t in range(len(trials)):
                    if len(unit_trial_spikes[i][t]):
                        ib.insert1({**skey,
                                    'insertion_number': probe,
                                    'clustering_method': method,
                                    'unit': u,
                                    'trial': trials[t]})
                        if ib.flush():
                            log.debug('.... (u: {}, t: {})'.format(u, t))

        # insert TrialSpikes
        log.info('.. ephys.Unit.TrialSpikes')
        with InsertBuffer(ephys.Unit.TrialSpikes, 10000, skip_duplicates=True,
                          allow_direct_insert=True) as ib:

            for i, u in enumerate(set(units)):
                for t in range(len(trials)):
                    ib.insert1({**skey,
                                'insertion_number': probe,
                                'clustering_method': method,
                                'unit': u,
                                'trial': trials[t],
                                'spike_times': unit_trial_spikes[i][t]})
                    if ib.flush():
                        log.debug('.... (u: {}, t: {})'.format(u, t))

        log.info('.. inserting file load information')

        self.insert1(skey, skip_duplicates=True)

        EphysIngest.EphysFile().insert1(
            {**skey, 'probe_insertion_number': probe,
             'ephys_file': str(ef_path.relative_to(rigpath))})

        log.info('ephys ingest for {} complete'.format(skey))

    def _gen_probe_insert(self, sinfo, probe, npx_meta):
        '''
        generate probe insertion for session / probe - for neuropixels recording

        Arguments:

          - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
          - probe: probe id

        '''

        part_no = npx_meta.probe_SN

        e_config = self._gen_electrode_config(npx_meta)

        # ------ ProbeInsertion ------
        insertion_key = {'subject_id': sinfo['subject_id'],
                         'session': sinfo['session'],
                         'insertion_number': probe}

        # add probe insertion
        log.info('.. creating probe insertion')

        lab.Probe.insert1({'probe': part_no, 'probe_type': e_config['probe_type']}, skip_duplicates=True)

        ephys.ProbeInsertion.insert1({**insertion_key,  **e_config, 'probe': part_no})

        ephys.ProbeInsertion.RecordingSystemSetup.insert1({**insertion_key, 'sampling_rate': npx_meta.meta['imSampRate']})

        return insertion_key

    def _gen_electrode_config(self, npx_meta):
        """
        Generate and insert (if needed) an ElectrodeConfiguration based on the specified neuropixels meta information
        """

        probe_type = npx_meta.probe_model

        if '1.0' in probe_type:
            eg_members = []
            probe_type = {'probe_type': probe_type}
            q_electrodes = lab.ProbeType.Electrode & probe_type
            for shank, shank_col, shank_row, is_used in npx_meta.shankmap['data']:
                electrode = (q_electrodes & {'shank': shank, 'shank_col': shank_col, 'shank_row': shank_row}).fetch1(
                    'KEY')
                eg_members.append({**electrode, 'is_used': is_used, 'electrode_group': 0})
        else:
            raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(probe_type))

        # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
        ec_hash = dict_to_hash({k['electrode']: k for k in eg_members})

        el_list = sorted([k['electrode'] for k in eg_members])
        el_jumps = [0] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
        ec_name = '; '.join([f'{el_list[s]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

        e_config = {**probe_type, 'electrode_config_name': ec_name}

        # ---- make new ElectrodeConfig if needed ----
        if not (lab.ElectrodeConfig & {'electrode_config_hash': ec_hash}):

            log.info('.. creating lab.ElectrodeConfig: {}'.format(ec_name))

            lab.ElectrodeConfig.insert1({**e_config, 'electrode_config_hash': ec_hash})

            lab.ElectrodeConfig.ElectrodeGroup.insert1({**e_config, 'electrode_group': 0})

            lab.ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

        return e_config

    @staticmethod
    def _decode_notes(fh, notes):
        '''
        dereference and decode unit notes, translate to local labels
        '''
        note_map = {'single': 'good', 'ok': 'ok', 'multi': 'multi',
                    '\x00\x00': 'all'}  # 'all' is default / null label
        decoded_notes = []
        for n in notes:
            note_val = str().join(chr(c) for c in fh[n])
            match = [k for k in note_map if re.match(k, note_val)]
            decoded_notes.append(note_map[match[0]] if len(match) > 0 else 'all')

        return decoded_notes

    def _load_v3(self, sinfo, rigpath, dpath, fpath):
        '''
        Ephys data loader for JRClust v4 files.

        Arguments:

          - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
          - rigpath: rig path root
          - dpath: expanded rig data path (rigpath/h2o/YYYY-MM-DD)
          - fpath: file path under dpath

        Returns:
          - tbd
        '''

        h2o = sinfo['water_restriction_number']
        skey = {k: v for k, v in sinfo.items()
                if k in experiment.Session.primary_key}

        probe = fpath.parts[0]

        ef_path = pathlib.Path(dpath, fpath)
        bf_path = pathlib.Path(dpath, probe, '{}_bitcode.mat'.format(h2o))

        log.info('.. jrclust v3 data load:')
        log.info('.... sinfo: {}'.format(sinfo))
        log.info('.... probe: {}'.format(probe))

        log.info('.... loading ef_path: {}'.format(str(ef_path)))
        ef = h5py.File(str(pathlib.Path(dpath, fpath)))  # ephys file

        log.info('.... loading bf_path: {}'.format(str(bf_path)))
        bf = spio.loadmat(pathlib.Path(
            dpath, probe, '{}_bitcode.mat'.format(h2o)))  # bitcode file

        # extract unit data

        hz = ef['P']['sRateHz'][0][0]                   # sampling rate

        spikes = ef['viTime_spk'][0]                    # spike times
        spike_sites = ef['viSite_spk'][0]               # spike electrode

        units = ef['S_clu']['viClu'][0]                 # spike:unit id
        unit_wav = ef['S_clu']['trWav_raw_clu']         # waveform

        unit_notes = ef['S_clu']['csNote_clu'][0]       # curation notes
        unit_notes = self._decode_notes(ef, unit_notes)

        unit_xpos = ef['S_clu']['vrPosX_clu'][0]        # x position
        unit_ypos = ef['S_clu']['vrPosY_clu'][0]        # y position

        unit_amp = ef['S_clu']['vrVpp_uv_clu'][0]       # amplitude
        unit_snr = ef['S_clu']['vrSnr_clu'][0]          # signal to noise

        vmax_unit_site = ef['S_clu']['viSite_clu']      # max amplitude site
        vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

        trial_start = bf['sTrig'].flatten()           # start of trials
        trial_go = bf['goCue'].flatten()                # go cues

        sync_ephys = bf['bitCodeS'].flatten()           # ephys sync codes
        sync_behav = (experiment.TrialNote()            # behavior sync codes
                      & {**skey, 'trial_note_type': 'bitcode'}).fetch(
                          'trial_note', order_by='trial')

        trial_fix = bf['trialNum'] if 'trialNum' in bf else None

        data = {
            'sinfo': sinfo,
            'rigpath': rigpath,
            'ef_path': ef_path,
            'probe': probe,
            'skey': skey,
            'method': 'jrclust_v3',
            'hz': hz,
            'spikes': spikes,
            'spike_sites': spike_sites,
            'units': units,
            'unit_wav': unit_wav,
            'unit_notes': unit_notes,
            'unit_xpos': unit_xpos,
            'unit_ypos': unit_ypos,
            'unit_amp': unit_amp,
            'unit_snr': unit_snr,
            'vmax_unit_site': vmax_unit_site,
            'trial_start': trial_start,
            'trial_go': trial_go,
            'sync_ephys': sync_ephys,
            'sync_behav': sync_behav,
            'trial_fix': trial_fix,
        }

        return data

    def _load_v4(self, sinfo, rigpath, dpath, fpath):
        '''
        Ephys data loader for JRClust v4 files.
        Arguments:
          - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
          - rigpath: rig path root
          - dpath: expanded rig data path (rigpath/h2o/YYYY-MM-DD)
          - fpath: file path under dpath
        '''

        h2o = sinfo['water_restriction_number']
        skey = {k: v for k, v in sinfo.items()
                if k in experiment.Session.primary_key}

        probe = fpath.parts[0]

        ef_path = pathlib.Path(dpath, fpath)

        log.info('.. jrclust v4 data load:')
        log.info('.... sinfo: {}'.format(sinfo))
        log.info('.... probe: {}'.format(probe))

        log.info('.... loading ef_path: {}'.format(str(ef_path)))
        ef = h5py.File(str(pathlib.Path(dpath, fpath)))  # ephys file

        # bitcode path (ex: 'SC022_030319_Imec3_bitcode.mat')
        bf_path = list(pathlib.Path(dpath, probe).glob(
            '{}_*bitcode.mat'.format(h2o)))[0]
        log.info('.... loading bf_path: {}'.format(str(bf_path)))
        bf = spio.loadmat(str(bf_path))

        # extract unit data
        hz = bf['SR'][0][0] if 'SR' in bf else None    # sampling rate

        spikes = ef['spikeTimes'][0]                    # spikes times
        spike_sites = ef['spikeSites'][0]               # spike electrode

        units = ef['spikeClusters'][0]                  # spike:unit id
        unit_wav = ef['meanWfLocalRaw']                 # waveform

        unit_notes = ef['clusterNotes']                 # curation notes
        unit_notes = self._decode_notes(ef, unit_notes[:].flatten())

        unit_xpos = ef['clusterCentroids'][0]           # x position
        unit_ypos = ef['clusterCentroids'][1]           # y position

        unit_amp = ef['unitVppRaw'][0]                  # amplitude
        unit_snr = ef['unitSNR'][0]                     # signal to noise

        vmax_unit_site = ef['clusterSites']             # max amplitude site
        vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

        trial_start = bf['sTrig'].flatten()             # trial start
        trial_go = bf['goCue'].flatten()                 # trial go cues

        sync_ephys = bf['bitCodeS']                     # ephys sync codes
        sync_behav = (experiment.TrialNote()            # behavior sync codes
                      & {**skey, 'trial_note_type': 'bitcode'}).fetch(
                          'trial_note', order_by='trial')

        trial_fix = bf['trialNum'] if 'trialNum' in bf else None

        data = {
            'sinfo': sinfo,
            'rigpath': rigpath,
            'ef_path': ef_path,
            'probe': probe,
            'skey': skey,
            'method': 'jrclust_v4',
            'hz': hz,
            'spikes': spikes,
            'spike_sites': spike_sites,
            'units': units,
            'unit_wav': unit_wav,
            'unit_notes': unit_notes,
            'unit_xpos': unit_xpos,
            'unit_ypos': unit_ypos,
            'unit_amp': unit_amp,
            'unit_snr': unit_snr,
            'vmax_unit_site': vmax_unit_site,
            'trial_start': trial_start,
            'trial_go': trial_go,
            'sync_ephys': sync_ephys,
            'sync_behav': sync_behav,
            'trial_fix': trial_fix,
        }

        return data


class NeuropixelsMeta:

    def __init__(self, meta_filepath):
        # a good processing reference: https://github.com/jenniferColonell/Neuropixels_evaluation_tools/blob/master/SGLXMetaToCoords.m

        self.fname = meta_filepath
        self.meta = self._read_meta()

        # Infer npx probe model (e.g. 1.0 (3A, 3B) or 2.0)
        probe_model = self.meta.get('imDatPrb_type', 1)
        if probe_model <= 1:
            if 'typeEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3A'
            elif 'typeImEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3B'
        else:
            self.probe_model = str(probe_model)

        # Get probe serial number - 'imProbeSN' for 3A and 'imDatPrb_sn' for 3B
        try:
            self.probe_SN = self.meta.get('imProbeSN', self.meta.get('imDatPrb_sn'))
        except KeyError:
            raise KeyError('Probe Serial Number not found in either "imProbeSN" or "imDatPrb_sn"')

        self.chanmap = self._parse_chanmap(self.meta['~snsChanMap']) if '~snsChanMap' in self.meta else None
        self.shankmap = self._parse_shankmap(self.meta['~snsShankMap']) if '~snsShankMap' in self.meta else None
        self.imroTbl = self._parse_imrotbl(self.meta['~imroTbl']) if '~imroTbl' in self.meta else None

    def _read_meta(self):
        '''
        Read metadata in 'k = v' format.

        The fields '~snsChanMap' and '~snsShankMap' are further parsed into
        'snsChanMap' and 'snsShankMap' dictionaries via calls to
        Neuropixels._parse_chanmap and Neuropixels._parse_shankmap.
        '''

        fname = self.fname

        res = {}
        with open(fname) as f:
            for l in (l.rstrip() for l in f):
                if '=' in l:
                    try:
                        k, v = l.split('=')
                        v = handle_string(v)
                        res[k] = v
                    except ValueError:
                        pass
        return res

    @staticmethod
    def _parse_chanmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#channel-map
        Parse channel map header structure. Converts:

            '(x,y,z)(c0,x:y)...(cI,x:y),(sy0;x:y)'

        e.g:

            '(384,384,1)(AP0;0:0)...(AP383;383:383)(SY0;768:768)'

        into dict of form:

            {'shape': [x,y,z], 'c0': [x,y], ... }
        '''

        res = {}
        for u in (i.rstrip(')').split(';') for i in raw.split('(') if i != ''):
            if (len(u)) == 1:
                res['shape'] = u[0].split(',')
            else:
                res[u[0]] = u[1].split(':')

        return res

    @staticmethod
    def _parse_shankmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#shank-map
        Parse shank map header structure. Converts:

            '(x,y,z)(a:b:c:d)...(a:b:c:d)'

        e.g:

            '(1,2,480)(0:0:192:1)...(0:1:191:1)'

        into dict of form:

            {'shape': [x,y,z], 'data': [[a,b,c,d],...]}

        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(':')])

        return res

    @staticmethod
    def _parse_imrotbl(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#imro-per-channel-settings
        Parse imro tbl structure. Converts:

            '(X,Y,Z)(A B C D E)...(A B C D E)'

        e.g.:

            '(641251209,3,384)(0 1 0 500 250)...(383 0 0 500 250)'

        into dict of form:

            {'shape': (x,y,z), 'data': []}
        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(' ')])

        return res


def handle_string(value):
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    return value
