import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot learning curve.')
    parser.add_argument('--history_file', default='model/history.csv', type=str,
                        help="path to history file.")
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.history_file)

    # figure object
    fig = plt.figure(figsize=(30, 30))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.5)
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
    else:
        plt.show()
