import sklearn.metrics as sklm

def compute_metrics(ytrue, yscore, threshold=0.5, path=None, string=False, cm=False):
    ypred = (yscore > threshold).astype(int)

    accuracy_balanced = sklm.balanced_accuracy_score(ytrue, ypred)
    f1_score = sklm.f1_score(ytrue, ypred)
    J = sklm.balanced_accuracy_score(ytrue, ypred, adjusted=True)
    matthew = sklm.matthews_corrcoef(ytrue, ypred)
    accuracy = sklm.accuracy_score(ytrue, ypred)
    precision = sklm.precision_score(ytrue, ypred)
    recall = sklm.recall_score(ytrue, ypred)
    roc_auc = sklm.roc_auc_score(ytrue, yscore)
    prec_avg = sklm.average_precision_score(ytrue, yscore)
    specificity = sklm.recall_score(1-ytrue, 1-ypred)

    CM = sklm.confusion_matrix(ytrue, ypred)

    res_string = "Bal. accuracy:\t{}\nF1 score:\t{}\nYouden's J:\t{}\nMatthew:\t{}\nAccuracy:\t" \
     "{}\nPrecision:\t{}\nRecall:\t\t{}\nSpecificity:\t{}\nRoc_auc:\t{}\nAvg prec:\t{}".format(accuracy_balanced,
                    f1_score, J, matthew, accuracy, precision, recall, specificity, roc_auc, prec_avg)
    if path is not(None):
        file = open(path, 'w+')
        file.write(res_string)
        file.close()

    if string:
        return res_string
    elif cm:
        return CM, accuracy_balanced, f1_score, J, matthew, accuracy, precision, recall, specificity, roc_auc, prec_avg
    else:
        return accuracy_balanced, f1_score, J, matthew, accuracy, precision, recall, specificity, roc_auc, prec_avg
