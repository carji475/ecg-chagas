import os

model_name = './model/model12/'

# train
os.system('python3 train.py --folder ' + model_name + ' --epochs 70 '
          '--path_to_data "/scratch/code/date_order.hdf5" --traces_dset "signal" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas_no_samitrop.csv" '
          '--data_tot 1.0 --dropout_rate 0.5 --valid_split 0.15 --patience 5 '
          '--cuda --weight_decay 0.001 --pos_weight --batch_size 64')

# evaluate
os.system('python3 evaluate.py --mdl ' + model_name + ' ' +
          '--path_to_traces "/scratch/code/samitrop1631.hdf5" --traces_dset "exam_id" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas_samitrop.csv" ')

