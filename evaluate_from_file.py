import numpy as np
import pandas as pd
import sklearn.metrics as sklm
from compute_metrics import compute_metrics
import os

## compute metrics and confidence intervals

# fix the seed
np.random.seed(0)

########### edit here after choice #############################################
model_path = 'model/ensemble/all/'
adder = '_elsa_ccc'  # test set specification
ci_valid = [False, False, False]  # Youden, F1, 90%
ci_test = [False, False, True]  # Youden, F1, 90%
################################################################################

def bootstrap_ci(y_true, y_pred, thres, bootstrap_nsamples=1000, lb=2.5, ub=97.5):
    # Compute bootstraped samples
    n = y_true.size
    samples = np.random.randint(n, size=n * bootstrap_nsamples)

    # Get samples
    y_true_resampled = np.reshape(y_true[samples], (bootstrap_nsamples, n))
    y_pred_resampled = np.reshape(y_pred[samples], (bootstrap_nsamples, n))

    # collect all bootstrapped metrics
    all_metrs_bs= np.zeros((bootstrap_nsamples,
                            len(compute_metrics(y_true, y_pred, thres))))

    for i in range(bootstrap_nsamples):
        print(i)
        metrs = compute_metrics(y_true_resampled[i], y_pred_resampled[i], thres)
        all_metrs_bs[i,:] = np.array(metrs)

    return np.percentile(all_metrs_bs, (lb, ub), axis=0).T


# file reading
# validation
infile_valid = os.path.join(model_path, 'best_valid_output.csv')
df_valid = pd.read_csv(infile_valid)
valid_true = df_valid['valid_true'].to_numpy()
if 'valid_output_mean' in df_valid.keys():  # case: ensemble model
    valid_output = df_valid['valid_output_mean'].to_numpy()
else:
    valid_output = df_valid['valid_output'].to_numpy()

# test
infile_test = os.path.join(model_path, 'evaluation'+adder+'.csv')
df_test = pd.read_csv(infile_test)
test_true = df_test['test_true'].to_numpy()
if 'test_output_mean' in df_test.keys():  # case: ensemble model
    test_output = df_test['test_output_mean'].to_numpy()
else:
    test_output = df_test['test_output'].to_numpy()


# allocate result lists
res_strings = []
optYs = []
resY_latex_list = []
cmsY = []
cisY = []

optF1s = []
resF1_latex_list = []
cisF1 = []
cmsF1 = []

opt90s = []
res90_latex_list = []
cms90 = []
cis90 = []
for ind, (data_true, data_output) in enumerate(zip((valid_true, test_true),
                                                   (valid_output, test_output))):
    # optimal threshold
    # Youden's J statistics
    fpr, tpr, threshold_roc = sklm.roc_curve(data_true, data_output)
    J = tpr - fpr
    maxJ_index = np.argmax(J)
    optYs.append(threshold_roc[maxJ_index])

    # F1
    precision, recall, threshold_pc = sklm.precision_recall_curve(data_true, data_output)
    f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    maxf1_index = np.argmax(f1_score)
    optF1s.append(threshold_pc[maxf1_index])

    # 90% specificity
    max90_index = np.argmin(np.abs(fpr-(1-0.90)))
    opt90s.append(threshold_roc[max90_index])

    # metrics
    res_strings.append( compute_metrics(data_true, data_output, optYs[0],
                                 path=None, string=True) )  # Youden
    res_strings.append( compute_metrics(data_true, data_output, optF1s[0],
                                  path=None, string=True) )  # F1
    res_strings.append( compute_metrics(data_true, data_output, opt90s[0],
                                  path=None, string=True) )  # 90% specificity

    # for latex table
    cm, _, f1, J, _, _, precision, recall, specificity, roc_auc, prec_avg = \
        compute_metrics(data_true, data_output, optYs[0], cm=True)
    resY_latex_list.append( [recall, specificity, precision, f1, J, roc_auc, prec_avg] )
    cmsY.append(cm)

    cm, _, f1, J, _, _, precision, recall, specificity, roc_auc, prec_avg = \
        compute_metrics(data_true, data_output, optF1s[0], cm=True)
    cmsF1.append(cm)
    resF1_latex_list.append( [recall, specificity, precision, f1, J, roc_auc, prec_avg] )

    cm, _, f1, J, _, _, precision, recall, specificity, roc_auc, prec_avg = \
        compute_metrics(data_true, data_output, opt90s[0], cm=True)
    cms90.append(cm)
    res90_latex_list.append( [recall, specificity, precision, f1, J, roc_auc, prec_avg] )


    # bootstraps
    if ind==0:
        if ci_valid[0]:
            cisY.append(bootstrap_ci(data_true, data_output, optYs[0]))
        if ci_valid[1]:
            cisF1.append(bootstrap_ci(data_true, data_output, optF1s[0]))
        if ci_valid[2]:
            cis90.append(bootstrap_ci(data_true, data_output, opt90s[0]))
    if ind==1:
        if ci_test[0]:
            cisY.append(bootstrap_ci(data_true, data_output, optYs[0]))
        if ci_test[1]:
            cisF1.append(bootstrap_ci(data_true, data_output, optF1s[0]))
        if ci_test[2]:
            cis90.append(bootstrap_ci(data_true, data_output, opt90s[0]))


all_metrics = '========== Valid Youden ==========\n'+res_strings[0]+\
          '\n\n========== Test Youden ==========\n'+res_strings[3]+\
          '\n\n========== Valid F1 ==========\n'+res_strings[1]+\
          '\n\n========== Test F1 ==========\n'+res_strings[4]+\
          '\n\n========== Valid 90% ==========\n'+res_strings[2]+\
          '\n\n========== Test 90% ==========\n'+res_strings[5]
print(all_metrics)
file = open(os.path.join(model_path, 'res'+adder+'.txt'), 'w+')
file.write(all_metrics)
file.close()

# data frame with bootstrapped cis
headers = ['Balanced accuracy',
             'F1 score',
             "Youden's J statistic",
             'Matthew',
             'Accuracy',
             'Precision',
             'Recall',
             'Specificity',
             'AUC/ROC',
             'Average precision']
cols=('lb','ub')
if ci_valid[0]:
    pd.DataFrame(cisY[0], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cisY_valid.csv'))
if ci_valid[1]:
    pd.DataFrame(cisF1[0], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cisF1_valid.csv'))
if ci_valid[2]:
    pd.DataFrame(cis90[0], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cis90_valid.csv'))
if ci_test[0]:
    pd.DataFrame(cisY[-1], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cisY'+adder+'.csv'))
if ci_test[1]:
    pd.DataFrame(cisF1[-1], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cisF1'+adder+'.csv'))
if ci_test[2]:
    pd.DataFrame(cis90[-1], index=headers, columns=cols).round(2)\
        .to_csv(os.path.join(model_path, 'cis90'+adder+'.csv'))


# create latex tables to copy-paste into the paper
file_latex = open(os.path.join(model_path, 'res_latex'+adder+'.txt'), 'w+')
for label, mylist in zip(['===== Youdens J =====\n', 5*'\n'+'===== F1 =====\n',
                          5*'\n'+'===== 90% =====\n'],
                         [resY_latex_list, resF1_latex_list, res90_latex_list]):
    file_latex.write(label +
        '\\begin{tabular}{l|c|c}\n' \
                'Metric \hfill & Validation & Test \\\ \n' \
                '\midrule\n' \
                '\midrule\n' \
                f"{'Recall (sensitivity)'} & ${mylist[0][0]:.2f}$ & ${mylist[1][0]:.2f}$ \\\ \n" \
                '\midrule\n' \
                f"{'Specificity'} & ${mylist[0][1]:.2f}$ & ${mylist[1][1]:.2f}$ \\\ \n" \
                '\midrule\n' \
                f"{'Precision'} & ${mylist[0][2]:.2g}$ & ${mylist[1][2]:.2f}$  \\\ \n" \
                '\midrule\n' \
                f"{'F1 score'} & ${mylist[0][3]:.2f}$ & ${mylist[1][3]:.2f}$ \\\ \n" \
                '\midrule\n' +\
                '{}'.format("Youden's J statistic") + f" & ${mylist[0][4]:.2f}$ & ${mylist[1][4]:.2f}$ \\\ \n" \
                '\midrule\n' \
                f"{'AUC-ROC'} & ${mylist[0][5]:.2f}$ & ${mylist[1][5]:.2f}$ \\\ \n" \
                '\midrule\n' \
                f"{'Avg. prec.'} & ${mylist[0][6]:.2f}$ & ${mylist[1][6]:.2f}$ \\\ \n" \
                '\midrule\n' \
        '\end{tabular}')
file_latex.close()





