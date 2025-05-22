#Code to run after a measurement to analyse alignment and to extract the experiment info
#Raw data must lay in data/raw_input/subject

from eeg_measurement import EegMeasurement
from eeg_attention_pytorch_dataset import EegAttentionDataset
import numpy as np
import git
import os


def debug_stim_writing():
    #create dataset with ica cleaned data

    #Specify newly measured subjects
    subjects = [116,120]
    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'test_stim.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)

        #write stimulus data only once
        #if subject == 120:
        eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        
        if subject > 116:
            #for new data only
            eeg_measurement.extract_experiment_info()
            eeg_measurement.analyse_alignment()
            eeg_measurement.analyse_drift()
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)
        print('Subject {} done'.format(subject))

def check_data_set():
    # subjects = list(range(116,117))
    # subjects = subjects + list(range(118,123))
    subjects = [116,120]
    subjects = [str(s) for s in subjects]
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    dir_h5 = os.path.join(base_dir, 'data', 'processed', 'ci_l_1,1_h_32,50_out_125_125_incl_ica.hdf5')
    trials = list(range(0,20))
    torch_datasets = []
    for sub in subjects:
        torch_datasets.append(EegAttentionDataset(dir_h5, sub, trials, window_size_training=60))
    pass

def debug_subject_summaries():
    #leave out 115 and 117
    subjects = [127, 128, 130]

    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct)
        eeg_measurement.extract_experiment_info()
        print(f'subject {subject} processed')

def create_intermediate_database():
    #create dataset with ica cleaned data, now ica is calculated on re-referenced data, which led to cleaner TRFs

    #Specify newly measured subjects

    #leave out 115 and 117
    subjects = [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    data_filename = 'intermediate_127_128_130_noreref.hdf5'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference=None)

        #write stimulus data only once
        if subject == 127:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=False)

def create_hdf5_dataset():
    #create dataset with ica cleaned data, now ica is calculated on re-referenced data, which led to cleaner TRFs
    #This is the dataset for final evaluation used in the paper

    #Specify newly measured subjects

    #leave out 101, 115 and 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]


    aux_channels_correct = np.ones(len(subjects), dtype=int).tolist()
    #subject 103 has switched aux channels
    aux_channels_correct[1] = 0

    l_freq_env = 1.0
    h_freq_env = 50.0

    l_freq_eeg = 1.0
    h_freq_eeg = 32

    output_freq = 125
    #data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    data_filename='test'
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    for subject, aux_correct in zip(subjects, aux_channels_correct):
        eeg_measurement = EegMeasurement(subject, base_dir, aux_correct, eeg_reference='avg')

        #write stimulus data only once
        if subject == 102:
            eeg_measurement.write_stimulus_data(data_filename = data_filename, l_freq_env=l_freq_env, h_freq_env = h_freq_env, output_freq = output_freq)
        eeg_measurement.write_subjects_eeg_data(data_filename = data_filename, l_freq_eeg = l_freq_eeg, h_freq_eeg = h_freq_eeg, output_freq = output_freq, use_ica=True)

if __name__ == '__main__':
    create_hdf5_dataset()