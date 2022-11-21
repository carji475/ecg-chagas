import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import sklearn.metrics as sklm
import seaborn as sns


def get_scores(y_true, y_pred, score_fun):
    scores = {name: fun(y_true, y_pred) for name, fun in score_fun.items()}
    return scores


def youdensJ_score(y_true, y_pred):
    return sklm.balanced_accuracy_score(y_true, y_pred, adjusted=True)


def specificity_score(y_true, y_pred):
    return sklm.recall_score(1-y_true, 1-y_pred)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--adder', default='_red',
                            help='test specification')
        parser.add_argument('--path_to_chagas', default='../data/',
                            help='path to chagas *folder*')
        parser.add_argument('--model', default='./model/ensemble/all/',
                            help='path to data.')
        parser.add_argument('--save', type=str, default=os.path.join(parser.parse_known_args()[0].model, 'stratify'),
                            help='file to save the plot.')
        args = parser.parse_args()

        # Generate figure folder if needed
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        # F1 threshold from validation data
        valid_file = os.path.join(args.model, 'best_valid_output.csv')
        df_valid = pd.read_csv(valid_file)
        valid_true = df_valid['valid_true'].to_numpy()
        if 'valid_output_mean' in df_valid.keys():
            valid_output = df_valid['valid_output_mean'].to_numpy()
        else:
            valid_output = df_valid['valid_output'].to_numpy()

        ## 90 % specificity
        fpr, tpr, threshold_roc = sklm.roc_curve(valid_true, valid_output)
        max90_index = np.argmin(np.abs(fpr-(1-0.90)))
        opt90 = threshold_roc[max90_index]
        ## f1
        precision, recall, threshold_pc = sklm.precision_recall_curve(valid_true, valid_output)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        maxf1_index = np.argmax(f1_score)
        optF1 = threshold_pc[maxf1_index]

        # data frames
        df_chagas = pd.read_csv(os.path.join(args.path_to_chagas,'chagas'+args.adder+'.csv'))
        df_test = pd.read_csv(os.path.join(args.model, 'evaluation'+args.adder+'.csv'))

        if 'elsa' in args.adder:
            df_chagas = df_chagas.set_index('ref_id')
            df_chagas = df_chagas.reindex(df_test['ref_id'])
            opt_thres = opt90
        else:
            df_chagas = df_chagas.set_index('exam_id')
            df_chagas = df_chagas.reindex(df_test['exam_id'])
            opt_thres = optF1

        df_test['age'] = df_chagas['age'].to_numpy()
        tmp = df_chagas['gender'].to_numpy()
        tmp[tmp=='M']=1; tmp[tmp=='F']=0; tmp=np.array(tmp, dtype=bool)
        df_test['is_male'] = tmp

        # Compute scores and bootstraped version of these scores
        bootstrap_nsamples = 1000

        if 'test_output_mean' in df_test.keys():
            y_pred = np.array(df_test['test_output_mean'] > opt_thres, dtype=int)
        else:
            y_pred = np.array(df_test['test_output'] > opt_thres, dtype=int)
        y_true = np.array(df_test['test_true'], dtype=int)

        strat_dicts = [{
            "feature": "age",
            "categories": ['0-40', '40-50', '50-60', '60-70', '70+'],
            "label": "Age",
            "ncols": 3
        },
        {
            "feature": "is_male",
            "categories": ['male', 'female'],
            "label": "Sex"
        }
        ]

        score_fun = {'Recall': sklm.recall_score,
                     'Specificity': specificity_score,
                     'Precision': sklm.precision_score,
                     'F1 score': sklm.f1_score}

        all_scores = []
        for _, strat_dict in enumerate(strat_dicts):

            condition = np.array(df_test[strat_dict["feature"]])

            # Compute bootstraped samples
            np.random.seed(0)
            n = y_true.size
            samples = np.random.randint(n, size=n * bootstrap_nsamples)

            # Get samples
            y_true_resampled = np.reshape(y_true[samples], (bootstrap_nsamples, n))
            y_pred_resampled = np.reshape(y_pred[samples], (bootstrap_nsamples, n))
            condition_resampled = np.reshape(condition[samples], (bootstrap_nsamples, n))
            # Apply functions
            scores_resampled = []

            masks = []
            if strat_dict["feature"] == 'age':
                for w, group in enumerate(strat_dict["categories"]):
                    if w == len(strat_dict["categories"])-1:
                        lower = int(group.split('+')[0])
                        upper = 1000
                    else:
                        lower, upper = tuple(np.array(group.split('-'), dtype=int))
                    masks.append((condition_resampled>lower) & (condition_resampled<upper))
            else:
                masks.append(condition_resampled)
                masks.append(~condition_resampled)
            masks = np.array(masks)

            for i in range(bootstrap_nsamples):
                for strat_cat, mask in zip(strat_dict["categories"], masks[:, i, :]):
                    s = get_scores(y_true_resampled[i, mask], y_pred_resampled[i, mask], score_fun)
                    scores_resampled += [{"score": n, "value": v,
                                          strat_dict["label"]: strat_cat}
                                          for n, v in s.items()]
            all_scores.append(scores_resampled)


# plot
plt.style.use(['./mystyle_boxplot.mplsty'])  # style file for box plot
for strat_dict, scores_resampled in zip(strat_dicts, all_scores):
    df_scores = pd.DataFrame(scores_resampled)
    plt.figure()
    ax = sns.boxplot(y="value", x="score", hue=strat_dict["label"], data=df_scores)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.legend(ncol=strat_dict['ncols'] if 'ncols' in strat_dict.keys() else 1, loc='best')
    plt.savefig(os.path.join(args.save, 'stratif' + args.adder + '_' + strat_dict["feature"] + '.pdf'))
