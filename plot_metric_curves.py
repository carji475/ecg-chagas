import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as sklm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot metric curve.')
    parser.add_argument('--evaluation', default='model/tmp/best_valid_output.csv', type=str,
                        help="path to evaluation file.")
    parser.add_argument('--dset', default='valid', type=str,
                        help="data set (train/valid/test)")
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(args.dset+' data set', fontsize=20)
    gs = fig.add_gridspec(1, 4, hspace=0.5, wspace=0.4)
    axarr = gs.subplots()

    df = pd.read_csv(args.evaluation)
    data_true = df[args.dset+'_true']
    data_output = df[args.dset+'_output']

    # roc curve
    ax = axarr[0]
    linewidth = 2
    fpr, tpr, threshold_roc = sklm.roc_curve(data_true, data_output)
    ax.plot(fpr, tpr, color='darkorange')
    ax.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    J = tpr - fpr  # Youden's J statistics
    maxJ_index = np.argmax(J)
    ax.plot(fpr[maxJ_index], tpr[maxJ_index], 'og', label='opt J')
    ax.legend()

    # J vs threshold
    ax = axarr[1]
    ax.plot(threshold_roc, J, color='darkorange')
    ax.plot(threshold_roc[maxJ_index]*np.ones(2), J[maxJ_index]*np.arange(2),
            '--', color='grey', label=str(threshold_roc[maxJ_index].round(2)))
    ax.plot(threshold_roc[maxJ_index], J[maxJ_index], 'og')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Threshold')
    ax.set_ylabel("Youden's J")
    ax.legend()

    # f1 vs threshold
    ax = axarr[3]
    precision, recall, threshold_pc = sklm.precision_recall_curve(data_true, data_output)
    f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    ax.plot(threshold_pc, f1_score[:-1], color='darkorange')
    maxf1_index = np.argmax(f1_score)
    ax.plot(threshold_pc[maxf1_index]*np.ones(2), f1_score[maxf1_index]*np.arange(2),
        '--', color='grey', label=str(threshold_pc[maxf1_index].round(2)))
    ax.plot(threshold_pc[maxf1_index], f1_score[maxf1_index], 'o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 score')
    ax.legend()

    # recall precision curve
    ax = axarr[2]
    ax.plot(recall, precision, color='darkorange')
    ax.plot(recall[maxf1_index], precision[maxf1_index], 'o', label='opt F1')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    ax.legend()
    
    if args.save:
        plt.savefig(args.save)
    plt.show(block=False)
