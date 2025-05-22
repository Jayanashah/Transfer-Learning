from src.evaluation.training_functions import train_dnn
import os
import git
import pickle
import torch
import numpy as np
import h5py
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from scipy.stats import pearsonr
import optuna
from src.models.dnn import CNN
import tomllib
import argparse
from torch.optim import NAdam
from collections.abc import Iterable
from torch.utils.data import DataLoader

from torch.optim import NAdam, Adam

from optuna.storages import JournalStorage, JournalFileStorage
import copy

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = argparse.ArgumentParser(description='Train CNN model on EEG attention dataset.')

parser.add_argument('-subj_0', type=int, help='index of first subject to train')
parser.add_argument('-subj_1', type=int, help='index of last subject to train')
parser.add_argument('-model_id', type=str, help='model id')
parser.add_argument('-job_nr', type=int, help='learning rate')
parser.add_argument('-activation_fct', type=str, help='Activation function either ELU, ReLU or LeakyReLU')
parser.add_argument('-conv_bias', type=int, help='Whether to use bias in convolutional layer')
args = parser.parse_args()

base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
data_dir = os.path.join(base_dir, "data/processed/ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5")

def generate_test_and_val_indices():
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    #generate pseudo random test and validation indices - where test indices are balanced between the books
    random_generator = np.random.default_rng(seed=random_state)

    book_0_test = random_generator.choice(competing_indices_book_0, size=3, replace=False)
    book_1_test = random_generator.choice(competing_indices_book_1, size=3, replace=False)
    test_indices = np.concatenate((book_0_test, book_1_test), axis=0)

    possible_val_indices = np.setdiff1d(competing_indcies, test_indices)
    val_indices = random_generator.choice(possible_val_indices, size=6, replace=False)

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in test_indices.tolist()]
    val_indices = [np.array([i]) for i in val_indices.tolist()]

    return test_indices, val_indices


def load_pretrained(model_id):
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    model_dir = os.path.join(base_dir, 'models', 'cnn', 'checkpoints', model_id, 'basemodel')
    model_kwargs = pickle.load(open(os.path.join(model_dir, 'model_kwargs.pkl'), 'rb'))

    state_dict = torch.load(os.path.join(model_dir, 'best_model.ckpt'), map_location=device)
    model = CNN(**model_kwargs)
    model.load_state_dict(state_dict)

    return model

def create_train_val_test_indices():
    random_state = 672
    competing_indices_book_0 = np.arange(8,19,2, dtype=int)
    competing_indices_book_1 = np.arange(9,20,2, dtype=int)
    competing_indcies = np.arange(8,20,1,dtype=int)
    #generate pseudo random test and validation indices - where test indices are balanced between the books
    random_generator = np.random.default_rng(seed=random_state)

    book_0_test = random_generator.choice(competing_indices_book_0, size=3, replace=False)
    book_1_test = random_generator.choice(competing_indices_book_1, size=3, replace=False)
    test_indices = np.concatenate((book_0_test, book_1_test), axis=0)

    possible_val_indices = np.setdiff1d(competing_indcies, test_indices)
    val_indices = random_generator.choice(possible_val_indices, size=6, replace=False)

    #convert to list of arrays - needed this way by evaluator class
    test_indices = [np.array([i]) for i in test_indices.tolist()]
    val_indices = [np.array([i]) for i in val_indices.tolist()]
    train_indices = [np.delete(np.arange(0,20), np.hstack((test_indices[i],val_indices[i]))) for i in range(0,len(test_indices))]

    return train_indices, val_indices, test_indices

def compare_pretraining(subjects, pretrain_id, ica=False, feature='env'):
    """
    Compare the performance of pretrained models to models trained without pretraining.
    For each pretrained model, a subject specific model is trained on the same subjects as the pretrained model with model id increased by one.
    args.model_id is used to give id to the finetuned model, subject specific models get id increased by one.
    watch out when using the method to increase model_ids by two

    Args:
        subjects (list): subjects to compare
        pretrain_id (str): model_id of pretrained model to load for the corresponding subjects
        ica (bool, optional): whether to use ica data. Defaults to False.
        feature (str, optional): feature to use. Defaults to 'env'. Must be either 'env' or 'onset_env'.
    """

    assert feature in ['env', 'onset_env'], 'feature must be either env or onset_env'

    window_size_training = 128

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 10}
    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #remaining parameters are predifined by pretrained model
    cnn_hyperparameters_pretrained = {"input_length": 100}

    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

    pretrained_model = load_pretrained(pretrain_id)
    print(f'Pretrained model loaded')


    #model id where to drop the finetuned models
    model_id_pretrained = args.model_id
    #subjects left out for finetuning
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_pretrained, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True,use_pretrained=True,
                                        pretrained_model=pretrained_model, **cnn_train_params, **cnn_hyperparameters_pretrained)
    
    #model id for models without pretraining
    model_id_subj = format(int(args.model_id) + 1, '03d')
    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)

def train_basemodels_complete_dataset():
    """
    Training baseline models leaving several subjects out for finetuning.
    Now on the complete dataset.
    """

    #define hyperparameters
    window_size_training = 128
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}
    cnn_train_params = {"data_dir": data_dir, "lr": 0.00003, "batch_size": 512, "weight_decay": 1e-08, "epochs": 35, "early_stopping_patience": 10}

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    held_out_subjects = np.array_split(subjects, 6)
    #subjects that can be used for pretraining
    training_subjects_list = [np.setdiff1d(subjects, held_out) for held_out in held_out_subjects]

    #setting index for subject to be held out --> can train in parallel
    if args.model_id == '130':
        subj_index = 0
    elif args.model_id == '131':
        subj_index = 1
    elif args.model_id == '132':
        subj_index = 2
    elif args.model_id == '133':
        subj_index = 3
    elif args.model_id == '134':
        subj_index = 4
    elif args.model_id == '135':
        subj_index = 5
    else:
        raise ValueError('Invalid model id')

    #to make results reproducible
    np.random.seed(672)

    #choose subjects for training for current execution
    training_subjects = training_subjects_list[subj_index]
    training_subject_strings = [str(subject) for subject in training_subjects]
    #choose three random subjects for validation
    validation_subjects = np.random.choice(training_subjects, size=4, replace=False)
    validation_subject_strings = [str(subject) for subject in validation_subjects]

    #choose four indices between 8 and 19 for validation
    validation_indices = np.random.choice(np.arange(8,19,dtype=int), size=4, replace=False)
    validation_indices = validation_indices.tolist()
    
    #create training indices specific to each subject (validation indices are removed from validation subjects)
    train_indices = []
    for subj, ind in zip(training_subjects, range(0,training_subjects.shape[0])):
        subj_train_indices = np.arange(0,20,1,dtype=int)
        #remove validation trials
        if subj in validation_subjects:
            subj_train_indices = np.delete(subj_train_indices, validation_indices)
        train_indices.append(subj_train_indices.tolist())

    print(f'Entering model {args.model_id} training')

    check_point_path = os.path.join(base_dir,'models','cnn', 'checkpoints', args.model_id, 'basemodel')
    #check if path exists and create it if not
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)

    best_correlation, best_state_dict = train_dnn(subject_string_train=training_subject_strings, checkpoint_path = check_point_path, train_indices = train_indices, 
                                            subject_string_val = validation_subject_strings, val_indices = validation_indices, workers=0, mdl_checkpointing=True, **cnn_train_params, **cnn_hyperparameters)

def eval_pretrain_effectiveness_complete_dataset():
    """
    Experiment to evaluate the effectiveness of pretraining on the competing trials.
    Subject specific models are trained using same models and training parameters as in the fine tuning experiments, just without pretraining.
    """

    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]
    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]


    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    evaluation_subjects = np.array_split(subjects, 6)
    evaluation_subjects = [x.tolist() for x in evaluation_subjects]
    #convert all elements to string
    evaluation_subjects = [[str(x) for x in y] for y in evaluation_subjects]

    #setting pretrained id matching to evaluation subjects

    if args.model_id == '140':
        pretrained_id = '130'
        subj_index = 0
    elif args.model_id == '142':
        pretrained_id = '131'
        subj_index = 1
    elif args.model_id == '144':
        pretrained_id = '132'
        subj_index = 2
    elif args.model_id == '146':
        pretrained_id = '133'
        subj_index = 3
    elif args.model_id == '148':
        pretrained_id = '134'
        subj_index = 4
    elif args.model_id == '150':
        pretrained_id = '135'
        subj_index = 5
    else:
        raise ValueError('Invalid model id')
    
    subjects = evaluation_subjects[subj_index]

    compare_pretraining(subjects=subjects, pretrain_id=pretrained_id)

def eval_ica_onset_cnn():
    """
    Evaluates the performance of the subject specific cnn models on ica-cleaned eeg data.
    Also considering onset_envelope as feature.
    """
    base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

    #subjects in dataset
    #leave out 101, 115, 117, 126, 129
    subjects = list(range(102,115))
    subjects = subjects + list(range(116,117))
    subjects = subjects + list(range(118,126))
    subjects = subjects + [127, 128, 130]

    #create sub lists for pretraining
    evaluation_subjects = np.array_split(subjects, 6)
    evaluation_subjects = [x.tolist() for x in evaluation_subjects]
    #convert all elements to string
    evaluation_subjects = [[str(x) for x in y] for y in evaluation_subjects]

    window_size_training = 128

    cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 15, "early_stopping_patience": 7}
    #parameters for subject specific models (identical to pretrained model)
    cnn_hyperparameters = {"dropout_rate": 0.35,"F1": 8,"D": 8,"F2": 64, "input_length": window_size_training, "num_input_channels": 31}

    #test on all sets
    test_indices_list = [np.array([i]) for i in range(8,20)]

    val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
    val_indices_list = [np.array([i]) for i in val_indices]

    train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_list[i]))) for i in range(0,len(test_indices_list))]

    #model id for fine tuning
    model_id_subj = format(int(args.model_id), '03d')

    if args.model_id in ['170', '171', '172', '173', '174', '175']:
        ica = True
        feature = 'env'
    elif args.model_id in ['180', '181', '182', '183', '184', '185']:
        ica = False
        feature = 'onset_env'
    elif args.model_id in ['190', '191', '192', '193', '194', '195']:
        ica = True
        feature = 'onset_env'
    else:
        raise ValueError('Invalid model id')
    
    if args.model_id in ['170', '180', '190']:
        subj_index = 0
    elif args.model_id in ['171', '181', '191']:
        subj_index = 1
    elif args.model_id in ['172', '182', '192']:
        subj_index = 2
    elif args.model_id in ['173', '183', '193']:
        subj_index = 3
    elif args.model_id in ['174', '184', '194']:
        subj_index = 4
    elif args.model_id in ['175', '185', '195']:
        subj_index = 5

    subjects = evaluation_subjects[subj_index]

    #training loop without pretraining
    for subject in subjects:
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

            checkpoint_path = os.path.join(base_dir,'models','cnn', 'checkpoints', model_id_subj, subject, 'test_ind_' + str(test_indices[0].item()))
            #check if path exists and create it if not
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            
            _, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
                                        subject_string_val = [subject], val_indices = val_indices, ica=ica, feature=feature, workers=0, mdl_checkpointing=True, 
                                        use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)

if __name__ == '__main__':
    #1. Train baseline models on complete dataset
    #train_basemodels_complete_dataset()
    #2. Evaluate pretraining effectiveness compare to models without pretraining
    eval_pretrain_effectiveness_complete_dataset()
    #3. Evaluate ica and onset envelope on subject specific models
    #eval_ica_onset_cnn()
