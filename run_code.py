import os

model_name = './model/model3'

os.system('python3 train.py --folder ' +model_name+ ' --epochs 70 '
          '--path_to_data "/scratch/code/date_order.hdf5" --traces_dset "signal" '
          '--examid_dset "id_exam" --path_to_chagas "/scratch/code/chagas.csv" '
          '--cuda --weight_decay 0.001')
