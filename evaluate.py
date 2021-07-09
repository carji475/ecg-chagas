# Imports
from resnet import ResNet1d
from tqdm import tqdm
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
from dataloader import ECGDatasetH5, ECGDataloaderH5
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl', default='model/', type=str,
                        help='folder containing model.')
    parser.add_argument('--path_to_traces', type=str, default='../data/sami-trop/exams.hdf5',
                        help='path to hdf5 containing ECG traces')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default=parser.parse_known_args()[0].mdl+'predicted_diagnoses.csv',
                        help='output file.')
    parser.add_argument('--traces_dset', default='tracings',
                         help='traces dataset in the hdf5 file.')
    parser.add_argument('--examid_dset', default='exam_id',
                     help='exam id dataset in the hdf5 file.')
    parser.add_argument('--path_to_chagas', default='../data/chagas.csv',
                        help='path to csv file containing chagas .')
    args, unk = parser.parse_known_args()

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get checkpoint
    ckpt = torch.load(os.path.join(args.mdl, 'model.pth'), map_location=lambda storage, loc: storage)

    # Get config
    config = os.path.join(args.mdl, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)

    # Get model
    N_LEADS = 12
    N_CLASSES = 1  # two classes, but just need one output
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=N_CLASSES,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])

    # load model checkpoint
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    # optimal threshold
    opt_threshold = ckpt['opt_threshold']

    # test dataloader
    test_set = ECGDatasetH5(
        path=args.path_to_traces,
        traces_dset=args.traces_dset,
        exam_id_dset=args.examid_dset,
        ids_dset=None,
        path_to_chagas=args.path_to_chagas
        )
    test_start = 0
    test_end = None
    test_set_in_chagas = np.where(test_set.in_chagas[test_start:test_end])[0]
    test_loader = ECGDataloaderH5(test_set, args.batch_size, test_start, test_end)

    # evaluate on test set
    model.eval()
    pred_diagnoses = np.zeros_like(test_set_in_chagas)  # allocate space
    eval_bar = tqdm(initial=0, leave=True, total=math.ceil(len(test_loader)/args.batch_size), position=0)
    end = test_start
    for traces, _ in test_loader:
        traces = traces.to(device)
        start = end
        with torch.no_grad():
            # Forward pass
            mod_out = model(traces)  # classify predictions

            end = min(start + len(traces), pred_diagnoses.size)
            pred_diagnoses[start:end] = (torch.nn.Sigmoid()(mod_out)-opt_threshold+0.5)\
                .round().detach().cpu().numpy().flatten()

        # Print result
        eval_bar.update(1)
    eval_bar.close()

    # Save predictions
    df = pd.DataFrame({'ids': test_set_in_chagas,
                       'exam_id': test_set.exams[test_set_in_chagas],
                       'predicted_chagas': pred_diagnoses.astype(np.bool)})
    df = df.set_index('ids')  # else we get two index columns ...
    df.to_csv(args.output)

    # true diagnoses
    true_diagnoses = test_loader.getfullbatch(attr_only=True).cpu().numpy()

    # metrics
    P = true_diagnoses.sum()
    N = (1-true_diagnoses).sum()
    TP = ((true_diagnoses+pred_diagnoses)==2).sum()
    FP = ((true_diagnoses-pred_diagnoses)==-1).sum()
    FN = ((true_diagnoses-pred_diagnoses)==1).sum()
    TN = ((true_diagnoses+pred_diagnoses)==0).sum()

    TPR = TP/P  # true positive rate
    TNR = TN/N  # true negative rate

    accuracy_balanced = (TPR+TNR)/2
    f1_score = 2*TP/(2*TP+FP+FN)
    matthew = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    accuracy = (TP+TN)/(P+N)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print('Balanced accuracy: {}\nF1 score: {}\nMatthew: {}\nAccuracy: {}\nPrecision: '
          '{}\nRecall: {}'.format(accuracy_balanced, f1_score, matthew,
                                  accuracy, precision, recall))
