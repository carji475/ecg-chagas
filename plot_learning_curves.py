import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot learning curve.')
    parser.add_argument('--history_file', default='model/tmp/history.csv', type=str,
                        help="path to history file.")
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    df = pd.read_csv(args.history_file)

    # figure object
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.6)
    axarr = gs.subplots(sharex=True)

    w = -1
    for q in range(df.keys().size):
        if 'train' in df.keys()[q]:
            w += 1

            ax = axarr.flatten()[w]
            metr = df.keys()[q].split('_', 1)[1]

            # plot losses
            ax.plot(df['epoch']+1, df[df.keys()[q]], label='train', color='blue')
            ax.plot(df['epoch']+1, df['valid_'+metr], label='valid', color='orange')
            ax.set_xlabel('epoch')
            ax.set_ylabel(metr, color='red')
            ax.legend()

    # Plot learning rate
    axt = axarr[0, 0].twinx()
    axt.step(df['epoch']+1, df['lr'], label='train', alpha=0.4, color='purple')
    axt.set_yscale('log')
    axt.set_ylabel('learning rate', alpha=1, color='purple')
    axt.set_ylim((1e-8, 1e-2))

    if args.save:
        plt.savefig(args.save)
    plt.show(block=False)
