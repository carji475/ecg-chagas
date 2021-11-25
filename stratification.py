# Generate visualizations of saliency maps for ecgs
# Parts of the implementation bellow is adapted from that of:
# http://kazuto1011.github.io from Kazuto Nakashima.
# Which is made available under MIT license.

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import ECGDatasetH5
from compute_metrics import compute_metrics

import os
import numpy as np
import sklearn.metrics as sklm

from resnet import ResNet1d
import h5py
import json

import matplotlib.pyplot as plt

def add_cm_bars(ax, cntrs, cms):
        cm_labels = ['TN', 'FN', 'FP', 'TP']
        bar_width = 0.05
        bar_atc = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
        for w, cntr in enumerate(cntrs):
                bar_pos = cntr + bar_atc
                bar_heights = cms[w].flatten(order='F')/cms[w].sum()
                bars = ax.bar(bar_pos, bar_heights, width=bar_width, color=['b','r','g','k'])
        ax.add_artist(ax.legend(bars,cm_labels, loc='lower center', bbox_to_anchor=(0.5,1), ncol=4))

if __name__ == "__main__":
        import argparse
        import ecg_plot
        from tqdm import tqdm
        import pandas as pd
        import matplotlib.pyplot as plt



        parser = argparse.ArgumentParser()
        parser.add_argument('--path_to_traces', default='../data/samitrop/samitrop1631.hdf5',
                        help='path to data.')
        parser.add_argument('--model', default='./model/model12',
                        help='path to data.')
        parser.add_argument('--save', type=str, default=os.path.join(parser.parse_args().model, 'stratify'),
                        help='file to save the plot.')
        args = parser.parse_args()

        # Youden's threshold from validation data
        valid_file = os.path.join(args.model, 'best_valid_output.csv')
        df_valid = pd.read_csv(valid_file)
        valid_true = df_valid['valid_true'].to_numpy()
        valid_output = df_valid['valid_output'].to_numpy()
        fpr, tpr, threshold_roc = sklm.roc_curve(valid_true, valid_output)
        J = tpr - fpr  # Youden's J statistics
        maxJ_index = np.argmax(J)
        optY = threshold_roc[maxJ_index]

        # test data
        samitrop = h5py.File(args.path_to_traces, 'r')

        test_file = os.path.join(args.model, 'evaluation.csv')
        df_test = pd.read_csv(test_file)
        df_test = df_test.set_index('exam_id')
        df_test = df_test.reindex(samitrop['exam_id'])
        df_test['age'] = samitrop['age']
        df_test['normal_ecg'] = samitrop['normal_ecg']
        df_test['is_male'] = samitrop['is_male']

        ## age stratification
        ages = np.append(np.arange(40, 81, 10), 200)
        lower = 0
        res = []
        label_str = []
        cms = []
        for age in ages:
            df_red = df_test[(df_test['age']>lower) & (df_test['age']<age)]
            metrics = compute_metrics(df_red['test_true'].to_numpy(),
                                  df_red['test_output'].to_numpy(), optY, cm=True)
            res.append( metrics[1:] )
            cms.append( metrics[0] )
            label_str.append(str(lower)+'-'+str(age))
            lower = age
        label_str[-1] = str(ages[-2])+'+'

        all_res = np.array(res)
        fig, ax = plt.subplots(1,1)
        cntrs = np.arange(ages.size)
        ax.plot(cntrs, all_res[:,-2], '-o', label='ROC')
        ax.plot(cntrs, all_res[:,-1], '-<', label='Avg. precision')
        ax.plot(cntrs, all_res[:,-5], '-d', label='Precision')
        ax.plot(cntrs, all_res[:,-4], '-v', label='Recall')
        ax.plot(cntrs, all_res[:,-3], '-x', label='Specificity')
        ax.set_xticks(cntrs)
        add_cm_bars(ax, cntrs, cms)
        ax.set_xticklabels(label_str)
        ax.set_xlabel('Age span')
        ax.legend()


        ## sex stratification
        res = []
        cms = []
        for is_male in [True, False]:
            df_red = df_test[df_test['is_male']==is_male]
            metrics = compute_metrics(df_red['test_true'].to_numpy(),
                                  df_red['test_output'].to_numpy(), optY, cm=True)
            res.append( metrics[1:] )
            cms.append( metrics[0] )

        all_res = np.array(res)
        fig, ax = plt.subplots(1,1)
        cntrs = np.arange(2)
        ax.plot(cntrs, all_res[:,-2], '-o', label='ROC')
        ax.plot(cntrs, all_res[:,-1], '-<', label='Avg. precision')
        ax.plot(cntrs, all_res[:,-5], '-d', label='Precision')
        ax.plot(cntrs, all_res[:,-4], '-v', label='Recall')
        ax.plot(cntrs, all_res[:,-3], '-x', label='Specificity')
        add_cm_bars(ax, cntrs, cms)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Male', 'Female'])
        ax.set_xlabel('Sex')
        ax.legend()


        ## normal ecg stratification
        cms = []
        res = []
        outputs = []
        for is_normal in [True, False]:
            df_red = df_test[df_test['normal_ecg']==is_normal]
            ground_truth = df_red['test_true'].to_numpy()
            output = df_red['test_output'].to_numpy()
            metrics = compute_metrics(ground_truth, output, optY, cm=True)
            res.append( metrics[1:] )
            cms.append( metrics[0] )
            outputs.append(output)

        all_res = np.array(res)
        fig, ax = plt.subplots(1,1)
        cntrs = np.arange(2)
        ax.plot(cntrs, all_res[:,-2], '-o', label='ROC')
        ax.plot(cntrs, all_res[:,-1], '-<', label='Avg. precision')
        ax.plot(cntrs, all_res[:,-5], '-d', label='Precision')
        ax.plot(cntrs, all_res[:,-4], '-v', label='Recall')
        ax.plot(cntrs, all_res[:,-3], '-x', label='Specificity')
        add_cm_bars(ax, cntrs, cms)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Abnormal'])
        ax.set_xlabel('ECG status')
        ax.legend()
