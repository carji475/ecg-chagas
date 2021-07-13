import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='plot learning curve.')
    parser.add_argument('--history_file', default='model/model3/history.csv', type=str,
                        help="path to history file.")
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--save', default='',
                        help='save the plot in the given file')
    args = parser.parse_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.history_file)

    fig, ax = plt.subplots()
    ax.plot(df['epoch']+1, df['train_loss'], label='train', color='blue')
    ax.plot(df['epoch']+1, df['valid_loss'], label='valid', color='orange')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss', color='red')
    # ax.set_ylim((8, 14))
    ax.legend()
    axt = ax.twinx()

    # Plot learning rate
    axt.step(df['epoch']+1, df['lr'], label='train', alpha=0.4, color='purple')
    axt.set_yscale('log')
    axt.set_ylabel('learning rate', alpha=1, color='purple')
    axt.set_ylim((1e-8, 1e-2))

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
