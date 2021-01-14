import pynwb
import pathlib
import numpy as np
from pynwb import NWBHDF5IO
from tqdm import tqdm
import sys
import datajoint as dj
import re
import json

db_prefix = dj.config['custom']['database.prefix']

cf = dj.create_virtual_module('cf', db_prefix + 'cf')
lab = dj.create_virtual_module('lab', db_prefix + 'lab')
experiment = dj.create_virtual_module('experiment', db_prefix + 'experiment')
ephys = dj.create_virtual_module('ephys', db_prefix + 'ephys')
misc = dj.create_virtual_module('misc', db_prefix + 'misc')

# Insert some meta information
lab.Person.insert1(('ars', 'ArsenyFinkelstein'), skip_duplicates=True)
lab.ModifiedGene.insert([{'gene_modification': 'Ai94',
                          'gene_modification_description': 'TITL-GCaMP6s Cre/Tet-dependent, fluorescent calcium indicator GCaMP6s inserted into the Igs7 locus (TIGRE)'},
                         {'gene_modification': 'CamK2a-tTA',
                          'gene_modification_description': 'When these Camk2a-tTA transgenic mice are mated to strain carrying a gene of interest under the regulatory control of a tetracycline-responsive promoter element (TRE; tetO), expression of the target gene can be blocked by administration of the tetracycline analog, doxycycline'},
                         {'gene_modification': 'slc17a7 IRES Cre 1D12',
                          'gene_modification_description': 'Cre recombinase expression directed to Vglut1-expressing cells'}],
                        skip_duplicates=True)
lab.Rig.insert1({'rig': 'ephys', 'room': '2W.330', 'rig_description': 'Ephys Behavior Optogenetics'},
                skip_duplicates=True)


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
    for egroup_no, egroup in nwbfile.electrode_groups.items():
        probe_part_no = egroup.device.name.split('_')[-1]
        ephys.ElectrodeGroup.insert1({**session_key,
                                      'electrode_group': int(egroup_no),
                                      'probe_part_no': probe_part_no})
        insert_location = {k: None if v == 'None' else v
                           for k, v in json.loads(egroup.location).items()}


# ============================== INGEST ALL ==========================================


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
