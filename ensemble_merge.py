import os
import numpy as np
import pandas as pd
import json


def test_valid_frames(base_path, adder):
    dft = pd.read_csv(os.path.join(base_path, 'evaluation'+adder+'.csv'))
    dfv = pd.read_csv(os.path.join(base_path, 'best_valid_output.csv'))
    return dft, dfv


def ensemble_merge(base_paths, out_path, adders, mdl_nrs):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get config
    config = os.path.join(base_paths[0], 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)

    for adder in adders:
        df_test, df_valid = test_valid_frames(base_paths[0], adder)
        ensemble_test = df_test.drop('test_output', axis=1)
        ensemble_valid = df_valid.drop('valid_output', axis=1)
        ensemble_hist = pd.DataFrame({'epoch': np.arange(config_dict['epochs'])})

        eht = np.zeros((config_dict['epochs'],len(base_paths)))
        ehv = np.zeros((config_dict['epochs'],len(base_paths)))

        for ind, (mdlnr, base_path) in enumerate(zip(mdl_nrs, base_paths)):
            df_test, df_valid = test_valid_frames(base_path, adder)
            train_hist = pd.read_csv(os.path.join(base_path, 'history.csv'))

            ensemble_test['test_output_'+str(mdlnr)] = df_test['test_output']
            ensemble_valid['valid_output_'+str(mdlnr)] = df_valid['valid_output']

            ensemble_hist['train_loss_'+str(mdlnr)] = train_hist['train_loss']
            eht[:,ind] = np.array(ensemble_hist['train_loss_'+str(mdlnr)])
            ensemble_hist['valid_loss_'+str(mdlnr)] = train_hist['valid_loss']
            ehv[:,ind] = np.array(ensemble_hist['valid_loss_'+str(mdlnr)])

        ensemble_hist['train_loss_mean'] = np.nanmean(eht,1)
        ensemble_hist['valid_loss_mean'] = np.nanmean(ehv,1)

        # average over models
        ensemble_test['test_output_mean'] = np.mean(np.array(ensemble_test)[:, 3::], 1)
        ensemble_valid['valid_output_mean'] = np.mean(np.array(ensemble_valid)[:, 1::], 1)

        # save data frame
        ensemble_test.to_csv(os.path.join(out_path, 'evaluation'+adder+'.csv'), index=False)
        ensemble_valid.to_csv(os.path.join(out_path, 'best_valid_output.csv'), index=False)
        ensemble_hist.to_csv(os.path.join(out_path, 'history.csv'), index=False)
