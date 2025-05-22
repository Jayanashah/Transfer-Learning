from src.data.eeg_measurement import EegMeasurement
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from src.evaluation.eval_vanilla_ridge import RidgeEvaluator
import numpy as np
import git
import os
import pandas
import pickle

import argparse

parser = argparse.ArgumentParser(description='Train CNN model on EEG attention dataset.')

parser.add_argument('-model_id', type=str, help='model id')
args = parser.parse_args()

def cross_val_models_complete_dataset():
    """
    Crossval over all competing trials
    """
    data_filename = 'ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5'
    accuracy_window_size = 60

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices), 1).reshape(-1).tolist()
    val_indices = [np.array([i]) for i in val_indices]

    freq_range = '1-32Hz'

    #reference model raw data on envelope
    if args.model_id == '020':
        model_id = args.model_id
        eval_raw_env = RidgeEvaluator(0, 1000, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=False, speech_feature= 'env', model_id=model_id)
        eval_raw_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using ica cleaned eeg data
    elif args.model_id == '021':
        model_id = args.model_id
        eval_ica_env = RidgeEvaluator(0, 1000, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=True, speech_feature= 'env', model_id=model_id)
        eval_ica_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

    #model using raw data and onset stimulus
    elif args.model_id == '022':
        model_id = args.model_id
        eval_raw_onset_env = RidgeEvaluator(0, 1000, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=False, speech_feature= 'onset_env', model_id=model_id)
        eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)
    
    elif args.model_id == '023':
        model_id = args.model_id
        eval_raw_onset_env = RidgeEvaluator(0, 1000, database_filename= data_filename, training_scheme='concat', 
                                            freq_range= freq_range, use_ica_data=True, speech_feature= 'onset_env', model_id=model_id)
        eval_raw_onset_env.eval_competing_speaker_cross_val(test_indices_list = test_indices, val_indices_list = val_indices, accuracy_window_size = accuracy_window_size)

if __name__ == '__main__':
    #create_dataset_19_10()
    cross_val_models_complete_dataset()