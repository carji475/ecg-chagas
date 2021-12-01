import os
import numpy as np
import h5py
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
    m = sklm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_to_traces', default='../data/samitrop/samitrop1631.hdf5',
                        help='path to data.')
        parser.add_argument('--model', default='./model/model12',
                        help='path to data.')
        parser.add_argument('--save', type=str, default=os.path.join(parser.parse_args().model, 'stratify'),
                        help='file to save the plot.')
        args = parser.parse_args()

        # Generate figure folder if needed
        if not os.path.exists(args.save):
            os.makedirs(args.save)

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



        #%% Compute scores and bootstraped version of these scores
        bootstrap_nsamples = 1000
        percentiles = [2.5, 97.5]
        scores_resampled_list = []
        scores_percentiles_list = []

        y_pred = np.array(df_test['test_output'] > optY, dtype=int)
        y_true = np.array(df_test['test_true'], dtype=int)

        strat_dicts = [{
            "feature": "age",
            "categories": ['0-40', '40-50', '50-60', '60-70', '70-80', '80+'],
            "label": "Age"
        },
        {
            "feature": "is_male",
            "categories": ['male', 'female'],
            "label": "Gender"
        },
        {
            "feature": "normal_ecg",
            "categories": ['normal', 'abnormal'],
            "label": "ECG status"
        }
        ]

        score_fun = {'Recall': sklm.recall_score, 'Specificity': specificity_score,
                     'Precision': sklm.precision_score,
                     'F1 score': sklm.f1_score, 'Youdens J': youdensJ_score}

        for w, strat_dict in enumerate(strat_dicts):

            condition = np.array(df_test[strat_dict["feature"]])

            # Compute bootstraped samples
            np.random.seed(123)  # NEVER change this =P
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
                        lower, upper = tuple(np.array(group.split('-'),dtype=int))
                    masks.append((condition_resampled>lower) & (condition_resampled<upper))
            else:
                masks.append(condition_resampled)
                masks.append(~condition_resampled)
            masks = np.array(masks)

            for i in range(bootstrap_nsamples):
                for strat_cat, mask in zip(strat_dict["categories"], masks[:, i, :]):
                    s = get_scores(y_true_resampled[i, mask], y_pred_resampled[i, mask], score_fun)
                    scores_resampled += [{"score": n, "value": v, strat_dict["label"]: strat_cat} for n, v in s.items()]

            df_scores = pd.DataFrame(scores_resampled)
            plt.figure(figsize=(16,4))
            ax = sns.boxplot(y="value", x="score", hue=strat_dict["label"], data=df_scores)
            ax.set_ylabel('')
            ax.set_xlabel('Metric')
            plt.savefig(os.path.join(args.save, strat_dict["feature"] + '.pdf'))
            plt.show()
