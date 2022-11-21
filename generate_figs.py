import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import os
from scipy.interpolate import interp1d


def fill_in(ax, alist, blist):
    a_all = np.concatenate(alist)
    a_all = np.sort(a_all)  # sort
    a_all = np.unique(a_all)  # remove repeated entries
    b_all = np.zeros((len(blist),a_all.size))
    for k,(r,p) in enumerate(zip(alist,blist)):
        interpolated = interp1d(r, p)
        b_all[k,:] = interpolated(a_all)
    b_max = np.nanmax(b_all,0)
    b_min = np.nanmin(b_all,0)
    ax.plot(a_all, b_min, color='orange', alpha=0.3)
    ax.plot(a_all, b_max, color='orange', alpha=0.3)
    ax.fill_between(a_all, b_min, b_max,
                    facecolor='orange', alpha=0.2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', default='',
                        help='path to data')
    parser.add_argument('--test_config', default='_red',
                        help='test data configuration')
    parser.add_argument('--fig_path', default=os.path.join(parser.parse_known_args()[0].path_to_data, 'figs'),
                        help='path to save figures')
    parser.add_argument('--metric_box', action='store_true',
                        help='include metric in roc/avg prec plots')
    args = parser.parse_args()
        
    base_path = args.path_to_data
    fig_path = args.fig_path
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    adder = args.test_config
    metric_box = args.metric_box

    ##### roc/avg. prec. plots
    types=['valid','test']
    files = [os.path.join(base_path, 'best_valid_output.csv'),
             os.path.join(base_path, 'evaluation'+adder+'.csv')]  # must match with 'types'
    for q, _ in enumerate(types):
        dset = types[q]

        plt.style.use(['./mystyle_roc_pc.mplsty'])
        fig1, ax1 = plt.subplots(figsize=(4.5,4))
        fig2, ax2 = plt.subplots(figsize=(4.5,4))
        figs = [fig1, fig2]
        axarr = [ax1, ax2]

        # data frames
        df = pd.read_csv(files[q])  # model output
        data_true = df[dset+'_true']  # ground truth

        # allocate result lists
        recall_list = []
        precision_list = []
        fpr_list = []
        tpr_list = []

        # loop over all sub-models
        for dfkey in df.keys():
            if dset in dfkey and not('true' in dfkey):
                name = dfkey.split(dset)[1]

                data_output = df[dset+name]

                fpr, tpr, threshold_roc = sklm.roc_curve(data_true, data_output)
                precision, recall, threshold_pc = sklm.precision_recall_curve(data_true, data_output)
                if not('mean' in name):
                    # roc curve
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)

                    # recall precision curve
                    recall_list.append(recall)
                    precision_list.append(precision)
                else:
                    roc_auc = sklm.roc_auc_score(data_true, data_output)
                    prec_avg = sklm.average_precision_score(data_true, data_output)
                    textstr_roc = r'AUC-ROC = %.2f' % (roc_auc, )
                    textstr_avp = r'Avg prec. = %.2f' % (prec_avg, )
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                    ax = axarr[0]
                    ax.plot(fpr, tpr, color='darkred')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    if metric_box:
                        ax.text(0.5, 0.08, textstr_roc, transform=ax.transAxes, fontsize=14,
                                verticalalignment='top', bbox=props)

                    ax = axarr[1]
                    ax.plot(recall, precision, color='darkred')
                    if metric_box:
                        ax.text(0.03, 0.08, textstr_avp, transform=ax.transAxes, fontsize=14,
                                verticalalignment='top', bbox=props)

        plt.figure(figs[0].number)
        ax = axarr[0]
        fill_in(ax,fpr_list,tpr_list)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel('Recall (sensitivity)')
        ax.set_xlabel('1 - specificity')
        plt.savefig(os.path.join(fig_path, 'roc_'+types[q]+adder+'.pdf'))

        plt.figure(figs[1].number)
        ax = axarr[1]
        fill_in(ax,recall_list,precision_list)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        plt.savefig(os.path.join(fig_path, 'pc_'+types[q]+adder+'.pdf'))
        
        ########################################################################
        ################# histogram plots ######################################
        ########################################################################
        plt.style.use(['./mystyle_hist.mplsty'])
        plt.figure(figsize=(4.5,4))
        nbins=50
        plt.hist(data_output[data_true==0],density=False,bins=nbins, alpha=1, fill=False, label='Negative')
        plt.hist(data_output[data_true==1],density=False,bins=nbins, alpha=0.6, label='Positive')
        plt.yscale('log')
        plt.xlabel('Network output')
        plt.ylabel('Number of samples')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'neg_pos_dens_'+types[q]+adder+'.pdf'))
        plt.savefig(os.path.join(fig_path, 'neg_pos_dens_'+types[q]+adder+'.pdf'))


    ##### history plot
    hf = os.path.join(base_path, 'history.csv')
    if os.path.exists(hf):
        df = pd.read_csv(hf)

        # figure object
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(1, 1, hspace=0.2, wspace=0.6)
        ax = gs.subplots(sharex=True)

        train_losses = []
        valid_losses = []
        for dfkey in df.keys():
            if 'train_loss_' in dfkey and not('mean' in dfkey):
                name = dfkey.split('train')[1]

                train_losses.append(df['train'+name].to_numpy())
                valid_losses.append(df['valid'+name].to_numpy())

        train_losses = np.array(train_losses)
        train_losses_max = np.nanmax(train_losses,0)
        train_losses_min = np.nanmin(train_losses,0)
        valid_losses_max = np.nanmax(valid_losses,0)
        valid_losses_min = np.nanmin(valid_losses,0)
        valid_losses = np.array(valid_losses)
        ax.plot(np.arange(1, 1+len(df['epoch'])), train_losses_max, alpha=0.3, color='blue')
        ax.plot(np.arange(1, 1+len(df['epoch'])), train_losses_min, alpha=0.3, color='blue')
        ax.fill_between(np.arange(1, 1+len(df['epoch'])), train_losses_min, train_losses_max,
                    facecolor='blue', alpha=0.2)
        ax.plot(np.arange(1, 1+len(df['epoch'])), valid_losses_max, alpha=0.3, color='orange')
        ax.plot(np.arange(1, 1+len(df['epoch'])), valid_losses_min, alpha=0.3, color='orange')
        ax.fill_between(np.arange(1, 1+len(df['epoch'])), valid_losses_min, valid_losses_max,
                    facecolor='orange', alpha=0.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Binary cross entropy')

        try:
            ax.plot(np.arange(1, 1+len(df['epoch'])), df['train_loss_mean'], label='Train data', color='blue')
            ax.plot(np.arange(1, 1+len(df['epoch'])), df['valid_loss_mean'], label='Validation data', color='orange')
            ax.legend()
        except:
            pass

        plt.savefig(os.path.join(fig_path, 'learning.pdf'))
