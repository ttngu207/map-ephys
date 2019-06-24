import os
import logging

import numpy as np
import scipy.io as scio
import datajoint as dj
import pathlib

from pipeline import lab
from pipeline import ephys
from pipeline import experiment
from pipeline import ccf
from pipeline import histology
from pipeline.ingest import ephys as ephys_ingest

from code import interact
from collections import ChainMap

from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_histology'))

log = logging.getLogger(__name__)


@schema
class HistologyIngest(dj.Imported):
    definition = """
    -> ephys_ingest.EphysIngest
    """

    class HistologyFile(dj.Part):
        definition = """
        -> master
        probe_insertion_number:        tinyint         # electrode group
        histology_file:         varchar(255)    # rig file subpath
        """

    def make(self, key):
        '''
        HistologyIngest .make() function
        '''
        # todo: check the length of the `site.ont.name` variable,
        #   and only ingest the sites with an ontology associated to it.
        log.info('HistologyIngest().make(): key: {}'.format(key))

        session = (experiment.Session & key).fetch1()

        rigpath = ephys_ingest.EphysDataPath().fetch1('data_path')
        subject_id = session['subject_id']
        session_date = session['session_date']
        water = (
            lab.WaterRestriction() & {'subject_id': subject_id}
        ).fetch1('water_restriction_number')

        egmap = {e['insertion_number']: e
                 for e in (ephys.ProbeInsertion & session).fetch('KEY')}

        errlabel = ccf.CCFLabel.CCF_R3_20UM_ERROR

        for probe in range(1, 3):

            log.info('... probe {} position ingest.'.format(probe))

            if probe not in egmap:
                log.info('... probe {} ephys N/A, skipping.'.format(probe))
                continue

            # subpaths like:
            # directory: {h2o}/{yyyy-mm-dd}/histology
            # file: landmarks_{h2o}_{YYYYMMDD}_{probe}_siteInfo.mat

            directory = pathlib.Path(water, session_date.strftime('%Y-%m-%d'), 'histology')
            file = 'landmarks_{}_{}_{}_siteInfo.mat'.format(
                water, session['session_date'].strftime('%Y%m%d'), probe)
            subpath = directory / file
            fullpath = rigpath / subpath

            if not fullpath.exists():
                log.info('... probe {} histology file {} N/A, skipping.'
                         .format(probe, fullpath))
                continue

            log.info('... found probe {} histology file {}'.format(
                probe, fullpath))

            hist = scio.loadmat(fullpath, struct_as_record=False, squeeze_me=True)['site']
            # probe CCF 3D positions
            pos_xyz = np.vstack([hist.pos.x, hist.pos.y, hist.pos.z]).T * 20  # 20um spacing (y-wise) between channels

            # probe CCF regions
            names = hist.ont.name  # there are 1000 names here, 1-960 represents the names for each electrode sites (960 sites for npx probe), ignore the last 40
            valid = [isinstance(n, (str,)) for n in names]
            goodn = np.where(np.array(valid))[0]

            # XXX: to verify - electrode off-by-one in ingest (e.g. mat->py)??
            # note, hard-code for neuropixels probe -> only one ElectrodeGroup (electrode_group=0)
            electrodes = np.array((ephys.ProbeInsertion * lab.ElectrodeConfig.Electrode
                                   & egmap[probe] & 'electrode_group=0').fetch(
                'KEY', order_by='electrode asc'))

            # interact('histoloading', local=dict(ChainMap(locals(), globals())))

            # XXX: off by one in ephys.ElectrodeGroup 'builder' routine?
            #   .. relatedly, we index pos_xyz by 'goodn' directly,
            #   .. rather than via electrodes[goodn]['electrode']
            recs = ({**p_i, 'ccf_label_id': ccf.CCFLabel.CCF_R3_20UM_ID, 'x': x, 'y': y, 'z': z}
                    for p_i, (x, y, z) in zip(electrodes[goodn], pos_xyz[goodn]))

            # ideally:
            # ephys.ElectrodeGroup.ElectrodePosition.insert(
            #     recs, allow_direct_insert=True)
            # but hitting ccf coordinate issues..:

            log.info('inserting channel ccf position')
            histology.ElectrodeCCFPosition.insert1(egmap[probe])

            for r in recs:
                log.debug('... adding probe/position: {}'.format(r))
                try:
                    histology.ElectrodeCCFPosition.ElectrodePosition.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)
                except Exception as e:
                    log.warning('... ERROR!')
                    histology.ElectrodeCCFPosition.ElectrodePositionError.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)

            log.info('... ok.')

        log.info('HistologyIngest().make(): {} complete.'.format(key))
        self.insert1(key)
