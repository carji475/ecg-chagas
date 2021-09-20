import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as sklm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot metric curve.')
    parser.add_argument('--evaluation', default='model/best_valid_outputs.csv', type=str,
                        help="path to evaluation file.")
    parser.add_argument('--dset', default='valid', type=str,
                        help="data set (train/valid/test)")
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(args.dset+' data set', fontsize=20)
    gs = fig.add_gridspec(1, 3, hspace=0.5, wspace=0.2)
    axarr = gs.subplots()

    df = pd.read_csv(args.evaluation)
    data_true = df[args.dset+'_true']
    data_output = df[args.dset+'_output']

    # roc curve
    ax = axarr[0]
    linewidth = 2
    fpr, tpr, thresholds = sklm.roc_curve(data_true, data_output)
    ax.plot(fpr, tpr, color='darkorange')
    ax.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')

    # f1 vs threshold
    ax = axarr[1]
    precision, recall, threshold = sklm.precision_recall_curve(data_true, data_output)
    f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    ax.plot(threshold, f1_score[:-1], color='darkorange')
    max_index = np.argmax(f1_score)
    ax.plot(threshold[max_index], f1_score[max_index], 'o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 score')

    # recall precision curve
    ax = axarr[2]
    ax.plot(recall, precision, color='darkorange')
    ax.plot(recall[max_index], precision[max_index], 'o', label='opt f1')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    ax.legend()

    plt.show()
    if args.save:
        plt.savefig(args.save)
