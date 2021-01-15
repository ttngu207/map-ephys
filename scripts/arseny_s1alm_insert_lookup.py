import datajoint as dj
import pickle

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

photostim_meta_fp = './scripts/s1alm_photostim_meta.pickle'
with open(photostim_meta_fp, 'rb') as f:
    photostims = pickle.load(f)
experiment.Photostim.insert(photostims, skip_duplicates=True, allow_direct_insert=True)

experiment.VideoFiducialsType.insert([{'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'jaw',
                                       'video_fiducial_description': 'anterior edge of the jaw/mouth'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'left_port',
                                       'video_fiducial_description': 'tip of the left lick port'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'nose_tip',
                                       'video_fiducial_description': 'anterior edge of the nose at the midline'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'right_port',
                                       'video_fiducial_description': 'tip of the right lick port'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'tongue_left',
                                       'video_fiducial_description': 'left edge of the tongue, at the point of minimal curvuture close to the tip'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'tongue_right',
                                       'video_fiducial_description': 'right edge of the tongue, at the point of minimal curvuture close to the tip'},
                                      {'rig': 'ephys',
                                       'tracking_device_type': 'Camera',
                                       'tracking_device_id': 1,
                                       'video_fiducial_name': 'tongue_tip',
                                       'video_fiducial_description': 'tip of the tongue'}],
                                     skip_duplicates=True, allow_direct_insert=True)
