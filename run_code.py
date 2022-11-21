import os

model_folders = []
models_pre_trained = []
nr_models = 15

# train to identify all positives or just ccc:s
ccc_train = False

# train data paths
samitrop_data_path  = '../data/samitrop/samitrop_all_records.h5'
if ccc_train:
    samitrop_chagas_path = '../data/chagas_samitrop2054_ccc.csv'
    folder_name = 'ccc'
else:
    samitrop_chagas_path = '../data/chagas_samitrop2054.csv'
    folder_name = 'all'
code_data_path = '/home/caran948/datasets/ecg-traces/preprocessed/traces.hdf5'
code_chagas_path = '../data/chagas_no_samitrop.csv'

# test data sets
red_data_path = '../data/red/REDS2.h5'
red_chagas_path = '../data/chagas_red.csv'
red_chagas_ccc_path = '../data/chagas_red_ccc.csv'
red_chagas_noccc_path = '../data/chagas_red_noccc.csv'

elsa_data_path = '../data/elsa/elsa.hdf5'
elsa_chagas_path = '../data/chagas_elsa.csv'
elsa_chagas_ccc_path = '../data/chagas_elsa_ccc.csv'
elsa_chagas_noccc_path = '../data/chagas_elsa_noccc.csv'


# result paths
for q in range(nr_models):
    model_folders.append(os.path.join('./models/model' + str(q), folder_name))


# train and evaluate
for q, model_folder in enumerate(model_folders):
    print(model_folder)

    ######################### train ############################################
    os.system('python3 train.py --folder ' + model_folder + ' --epochs 200 '
              '--path_to_data ' + samitrop_data_path + ' --traces_dset "tracings" '
              '--examid_dset "id_exam" --path_to_chagas ' + samitrop_chagas_path + ' '
              '--path_to_data2 ' + code_data_path + ' --traces_dset2 "signal" '
              '--examid_dset2 "id_exam" --path_to_chagas2 ' + code_chagas_path + ' '
              '--data_tot 1.0 --dropout_rate 0.5 --valid_split 0.3 --patience 10 '
              '--cuda --weight_decay 0.001 --batch_size 32 --lr 0.001 '
              '--seed ' + str(q))


    ######################### evaluation RED ###################################
    red_output_both = os.path.join(model_folder, 'evaluation_red.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + red_data_path + ' --traces_dset "tracings" '
              '--examid_dset "id_exam" --path_to_chagas ' + red_chagas_path + ' '
              '--output ' + red_output_both)

    red_output_ccc = os.path.join(model_folder, 'evaluation_red_ccc.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + red_data_path + ' --traces_dset "tracings" '
              '--examid_dset "id_exam" --path_to_chagas ' + red_chagas_ccc_path + ' '
              '--output ' + red_output_ccc)

    red_output_noccc = os.path.join(model_folder, 'evaluation_red_noccc.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + red_data_path + ' --traces_dset "tracings" '
              '--examid_dset "id_exam" --path_to_chagas ' + red_chagas_noccc_path + ' '
              '--output ' + red_output_noccc)
    
    
    ######################### evaluation ELSA-BRASIL ###########################          
    elsa_output_both = os.path.join(model_folder, 'evaluation_elsa.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + elsa_data_path + ' --traces_dset "signal" '
              '--examid_dset "ref_id" --path_to_chagas ' + elsa_chagas_path + ' '
              '--chagas_id "ref_id" --output ' + elsa_output_both)

    elsa_output_ccc = os.path.join(model_folder, 'evaluation_elsa_ccc.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + elsa_data_path + ' --traces_dset "signal" '
              '--examid_dset "ref_id" --path_to_chagas ' + elsa_chagas_ccc_path + ' '
              '--chagas_id "ref_id" --output ' + elsa_output_ccc)

    elsa_output_noccc = os.path.join(model_folder, 'evaluation_elsa_noccc.csv')
    os.system('python3 evaluate.py --mdl ' + model_folder + ' ' +
              '--path_to_traces ' + elsa_data_path + ' --traces_dset "signal" '
              '--examid_dset "ref_id" --path_to_chagas ' + elsa_chagas_noccc_path + ' '
              '--chagas_id "ref_id" --output ' + elsa_output_noccc)


# merge to ensemble models
import numpy as np
from ensemble_merge import ensemble_merge
mdl_nrs = np.arange(nr_models)
adders = ['_elsa','_elsa_ccc','_elsa_noccc','_red','_red_ccc','_red_noccc']
out_path = os.path.join('./models/ensemble', folder_name)
ensemble_merge(model_folders, out_path, adders, mdl_nrs)


# stratify
for adder in ['_red', '_elsa']:
    os.system('python3 stratification.py --adder ' + adder + ' --path_to_chagas "../data/" '
              '--model ' + os.path.join('./models/ensemble/', folder_name))


# plots
print(os.path.join('./models/ensemble/', folder_name))
for adder in adders:
    os.system('python3 generate_figs.py --path_to_data ' + os.path.join('./models/ensemble/', folder_name)
              + ' --test_config ' + adder)


# grad-cam
for chagas_path in [elsa_chagas_ccc_path, elsa_chagas_noccc_path]:
    os.system('python3 generate_visualisations.py --path_to_traces ' + elsa_data_path + ' ' +
              '--adder "_elsa" --path_to_chagas ' + chagas_path + ' '
              '--model ' + model_folders[0] + ' --traces_dset "signal" '
              '--examid_dset "ref_id" --chagas_id "ref_id"')
