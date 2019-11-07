# map-ephys interative shell module

import os
import sys
import logging
from code import interact
import time

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from pipeline import ccf
from pipeline import ephys
from pipeline import histology
from pipeline import tracking
from pipeline import psth
from pipeline import publication
from pipeline import export
from pipeline import report


pipeline_modules = [lab, ccf, experiment, ephys, histology, tracking, psth,
                    publication]

log = logging.getLogger(__name__)


def usage_exit():
    print("usage: {p} [{c}] <args>"
          .format(p=os.path.basename(sys.argv[0]),
                  c='|'.join(list(actions.keys()))))
    sys.exit(0)


def logsetup(*args):
    level_map = {
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'DEBUG': logging.DEBUG,
    }
    level = level_map[args[0]] if args else logging.INFO

    logging.basicConfig(level=logging.ERROR)
    log.setLevel(level)
    logging.getLogger('pipeline.ingest.behavior').setLevel(level)
    logging.getLogger('pipeline.ingest.ephys').setLevel(level)
    logging.getLogger('pipeline.ingest.tracking').setLevel(level)
    logging.getLogger('pipeline.ingest.histology').setLevel(level)
    logging.getLogger('pipeline.psth').setLevel(level)
    logging.getLogger('pipeline.ccf').setLevel(level)
    logging.getLogger('pipeline.publication').setLevel(level)


def ingest_behavior(*args):
    from pipeline.ingest import behavior as behavior_ingest
    behavior_ingest.BehaviorIngest().populate(display_progress=True)


def ingest_ephys(*args):
    from pipeline.ingest import ephys as ephys_ingest
    ephys_ingest.EphysIngest().populate(display_progress=True)


def ingest_tracking(*args):
    from pipeline.ingest import tracking as tracking_ingest
    tracking_ingest.TrackingIngest().populate(display_progress=True)


def ingest_histology(*args):
    from pipeline.ingest import histology as histology_ingest
    histology_ingest.HistologyIngest().populate(display_progress=True)


def populate_psth(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('ephys.UnitStat.populate()')
    ephys.UnitStat.populate(**populate_settings)

    log.info('ephys.UnitCellType.populate()')
    ephys.UnitCellType.populate(**populate_settings)

    log.info('psth.UnitPsth.populate()')
    psth.UnitPsth.populate(**populate_settings)

    log.info('psth.PeriodSelectivity.populate()')
    psth.PeriodSelectivity.populate(**populate_settings)

    log.info('psth.UnitSelectivity.populate()')
    psth.UnitSelectivity.populate(**populate_settings)


def generate_report(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('report.SessionLevelReport.populate()')
    report.SessionLevelReport.populate(**populate_settings)

    log.info('report.ProbeLevelReport.populate()')
    report.ProbeLevelReport.populate(**populate_settings)

    log.info('report.ProbeLevelPhotostimEffectReport.populate()')
    report.ProbeLevelPhotostimEffectReport.populate(**populate_settings)

    log.info('report.UnitLevelReport.populate()')
    report.UnitLevelReport.populate(**populate_settings)

    log.info('report.SessionLevelCDReport.populate()')
    report.SessionLevelCDReport.populate(**populate_settings)


def sync_report():
    stage = dj.config['stores']['report_store']['stage']

    log.info(f'Sync report.SessionLevelReport from {stage}')
    report.SessionLevelReport.fetch()

    log.info(f'Sync report.ProbeLevelReport from {stage}')
    report.ProbeLevelReport.fetch()

    log.info(f'Sync report.ProbeLevelPhotostimEffectReport from {stage}')
    report.ProbeLevelPhotostimEffectReport.fetch()

    log.info(f'Sync report.UnitLevelReport from {stage}')
    report.UnitLevelReport.fetch()

    log.info(f'Sync report.SessionLevelCDReport from {stage}')
    report.SessionLevelCDReport.fetch()


def nuke_all():
    if 'nuclear_option' not in dj.config:
        raise RuntimeError('nuke_all() function not enabled')

    from pipeline.ingest import behavior as behavior_ingest
    from pipeline.ingest import ephys as ephys_ingest
    from pipeline.ingest import tracking as tracking_ingest
    from pipeline.ingest import histology as histology_ingest

    ingest_modules = [behavior_ingest, ephys_ingest, tracking_ingest,
                      histology_ingest]

    for m in reversed(ingest_modules):
        m.schema.drop()

    # production lab schema is not map project specific, so keep it.
    for m in reversed([m for m in pipeline_modules if m is not lab]):
        m.schema.drop()


def publish(*args):
    publication.ArchivedRawEphys.populate()
    publication.ArchivedTrackingVideo.populate()


def export_recording(*args):
    if not args:
        print("usage: {} export-recording \"probe key\"\n"
              "  where \"probe key\" specifies a ProbeInsertion")
        return

    ik = eval(args[0])  # "{k: v}" -> {k: v}
    fn = args[1] if len(args) > 1 else None
    export.export_recording(ik, fn)


def shell(*args):
    interact('map shell.\n\nschema modules:\n\n  - {m}\n'
             .format(m='\n  - '.join(
                 '.'.join(m.__name__.split('.')[1:])
                 for m in pipeline_modules)),
             local=globals())


def ccfload(*args):
    ccf.CCFAnnotation.load_ccf_r3_20um()


def erd(*args):
    for mod in (ephys, lab, experiment, tracking, psth, ccf, publication):
        modname = str().join(mod.__name__.split('.')[1:])
        fname = os.path.join('pipeline', './images/{}.png'.format(modname))
        print('saving', fname)
        dj.ERD(mod, context={modname: mod}).save(fname)


def automate_computation():
    populate_settings = {'reserve_jobs': True, 'suppress_errors': True, 'display_progress': True}
    while True:
        populate_psth(populate_settings)
        generate_report(populate_settings)

        time.sleep(1)


actions = {
    'ingest-behavior': ingest_behavior,
    'ingest-ephys': ingest_ephys,
    'ingest-tracking': ingest_tracking,
    'ingest-histology': ingest_histology,
    'populate-psth': populate_psth,
    'publish': publish,
    'export-recording': export_recording,
    'generate-report': generate_report,
    'sync-report': sync_report,
    'shell': shell,
    'erd': erd,
    'ccfload': ccfload,
    'automate-computation': automate_computation
}
