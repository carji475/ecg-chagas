import os

model_name = './model/model2'

os.system('python3 train.py --folder ' +model_name+ ' --epochs 10 '
          '--path_to_data "/scratch/code/date_order.hdf5" --traces_dset "signal" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas.csv" '
          '--cuda --weight_decay 0.01')
