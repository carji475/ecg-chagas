import os

model_name = './model/model8'

# train
os.system('python3 train.py --folder ' + model_name + ' --epochs 70 '
          '--path_to_data "/scratch/code/date_order.hdf5" --traces_dset "signal" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas.csv" '
          '--data_tot 1.0 --dropout_rate 0.8 --valid_split 0.15 --patience 7 '
          '--cuda --weight_decay 0.01')

# evaluate
os.system('python3 evaluate.py --mdl ' + model_name + ' ' +
          '--path_to_traces "/scratch/code/samitrop.hdf5" --traces_dset "signal" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas.csv" ')

# loss fig
os.system('python3 plot_learning_curves.py --history_file ' + os.path.join(model_name, 'history.csv') +
          ' --save ' + os.path.join(model_name, 'res.pdf'))
