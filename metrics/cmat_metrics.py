import numpy as np
from typing import NamedTuple

CMResults = NamedTuple(
    "CMResults",
    [
        ('overall_accuracy', float),
        #
        ("sensitivity_micro", float),
        ("sensitivity_macro", float),
        ("sensitivity_class", list),
        #
        ("specificity_micro", float),
        ("specificity_macro", float),
        ("specificity_class", float),
        ("precision_micro", float),
        ("precision_macro", float),
        ("precision_class", float),
        #
        ("recall_micro", float),
        ("recall_macro", float),
        ("recall_class", float),
        #
        ("f1_micro", float),
        ("f1_macro", float),
        ("f1_class", float),
        #
        ("accuracy_micro", float),
        ("accuracy_macro", float),
        ("accuracy_class", float),
        #
        ('true_positive_rate_micro', float),
        ('true_positive_rate_macro', float),
        ('true_positive_rate_class', float),
        #
        ('true_negative_rate_micro', float),
        ('true_negative_rate_macro', float),
        ('true_negative_rate_class', float),
        #
        ('false_positive_rate_micro', float),
        ('false_positive_rate_macro', float),
        ('false_positive_rate_class', float),
        #
        ('false_negative_rate_micro', float),
        ('false_negative_rate_macro', float),
        ('false_negative_rate_class', float),
        #
        ('positive_pred_value_micro', float),
        ('positive_pred_value_macro', float),
        ('positive_pred_value_class', float),
        #
        ('negative_pred_value_micro', float),
        ('negative_pred_value_macro', float),
        ('negative_pred_value_class', float),
        #
        ('negative_likelihood_ratio_micro', float),
        ('negative_likelihood_ratio_macro', float),
        ('negative_likelihood_ratio_class', float),
        #
        ('positive_likelihood_ratio_micro', float),
        ('positive_likelihood_ratio_macro', float),
        ('positive_likelihood_ratio_class', float),
        #
        ('diagnostic_odd_ratio_micro', float),
        ('diagnostic_odd_ratio_macro', float),
        ('diagnostic_odd_ratio_class', float),
        #
        ('younden_index_micro', float),
        ('younden_index_macro', float),
        ('younden_index_class', float)
    ],
)


def compute_micro_stats(values_a, values_b, eps=1e-8):
    sum_a = np.sum(values_a)
    sum_b = np.sum(values_b)

    micro_sc = sum_a / (sum_a + sum_b + eps)

    return micro_sc


def compute_macro_stats(values):
    return np.mean(values)


class CMMetrics(object):
    '''
    Metrics defined here: https://www.sciencedirect.com/science/article/pii/S2210832718301546
    '''

    def __init__(self):
        super(CMMetrics, self).__init__()
        self.eps = 1e-8

    def compute_precision(self, tp, fp):
        '''
        Precision = TP/(TP + FP)
        '''
        class_wise = tp / (tp + fp + self.eps)

        micro = compute_micro_stats(tp, fp)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_senstivity(self, tp, fn):
        '''
            Sensitivity = TP/(TP + FN)
        '''
        class_wise = tp / (tp + fn + self.eps)
        micro = compute_micro_stats(tp, fn)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_specificity(self, tn, fp):
        class_wise = (tn / (tn + fp + self.eps))
        micro = compute_micro_stats(tn, fp)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_recall(self, tp, fn):
        # same as sensitivity
        class_wise = (tp / (tp + fn + self.eps))
        micro = compute_micro_stats(tp, fn)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_f1(self, precision, recall):
        return (2.0 * precision * recall) / (precision + recall)

    def compute_acc(self, tp, tn, fp, fn):
        class_wise = ((tp + tn) / (tp + tn + fp + fn + self.eps))
        micro = compute_micro_stats(tp + tn, tp + tn + fp + fn)
        macro = compute_macro_stats(class_wise)
        return micro, macro, class_wise

    def compute_overall_acc(self, tp, N):
        return tp.sum() / (N + self.eps)

    def compute_tpr(self, tp, fn):
        # True positive rate
        # same as senstivity and recall
        class_wise = (tp / (tp + fn + self.eps))

        micro = compute_micro_stats(tp, fn)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_tnr(self, tn, fp):
        # True negative rate
        # same as specificity
        class_wise = (tn / (tn + fp + self.eps))
        micro = compute_micro_stats(tn, fp)
        macro = compute_macro_stats(class_wise)
        return micro, macro, class_wise

    def compute_fpr(self, fp, tn):
        # False posistive rate
        class_wise = (fp / (fp + tn + self.eps))
        micro = compute_micro_stats(fp, tn)

        macro = compute_macro_stats(class_wise)
        return micro, macro, class_wise

    def compute_fnr(self, fn, tp):
        # false negative rate
        # fnr_micro
        class_wise = (fn / (fn + tp + self.eps))

        micro = compute_micro_stats(fn, tp)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_ppv(self, tp, fp):
        # Positive prediction value
        return self.compute_precision(tp=tp, fp=fp)

    def compute_npv(self, tn, fn):
        # Negative predictive value
        class_wise = (tn / (tn + fn + self.eps))
        micro = compute_micro_stats(tn, fn)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_neg_lr(self, tpr, tnr):
        # negative likelihood ratio
        return (1.0 - tpr) / (tnr + self.eps)

    def compute_pos_lr(self, tpr, tnr):
        # positive likelihood ratio
        return tpr / (1.0 - tnr + self.eps)

    def compute_dor(self, tp, tn, fp, fn):
        # Diagnostic odds ratio
        class_wise = ((tp * tn) / (fp * fn + self.eps))

        micro = compute_micro_stats(tp * tn, fp * fn)
        macro = compute_macro_stats(class_wise)

        return micro, macro, class_wise

    def compute_younden_index(self, tpr, tnr):
        # Youden's index
        return tpr + tnr - 1.0

    def compute_metrics(self, conf_mat):
        num_samples = conf_mat.sum()
        if conf_mat.shape[0] > 2:
            true_positives = np.diag(conf_mat)
            false_positives = conf_mat.sum(axis=0) - true_positives
            false_negatives = conf_mat.sum(axis=1) - true_positives
            true_negatives = conf_mat.sum() - (false_positives + false_negatives + true_positives)
        else:
            true_negatives, false_positives, false_negatives, true_positives = conf_mat.ravel()

        false_positives = false_positives.astype(float)
        false_negatives = false_negatives.astype(float)
        true_positives = true_positives.astype(float)
        true_negatives = true_negatives.astype(float)

        #print(true_positives, true_negatives, false_positives, false_negatives)

        sensitivity_micro, sensitivity_macro, sensitivity_class = self.compute_senstivity(tp=true_positives,
                                                                                          fn=false_negatives)
        specificity_micro, specificity_macro, specificity_class = self.compute_specificity(tn=true_negatives,
                                                                                           fp=false_positives)
        precision_micro, precision_macro, precision_class = self.compute_precision(tp=true_positives,
                                                                                   fp=false_positives)
        recall_micro, recall_macro, recall_class = self.compute_recall(tp=true_positives, fn=false_negatives)
        f1_micro = self.compute_f1(precision=precision_micro, recall=recall_micro)
        f1_macro = compute_macro_stats(self.compute_f1(precision=precision_class, recall=recall_class))
        f1_class = self.compute_f1(precision=precision_class, recall=recall_class)

        acc_micro, acc_macro, acc_class = self.compute_acc(tp=true_positives, tn=true_negatives, fp=false_positives,
                                                           fn=false_negatives)
        overall_acc = self.compute_overall_acc(tp=true_positives, N=num_samples)

        tpr_micro, tpr_macro, tpr_class = self.compute_tpr(tp=true_positives, fn=false_negatives)
        tnr_micro, tnr_macro, tnr_class = self.compute_tnr(tn=true_negatives, fp=false_positives)
        fpr_micro, fpr_macro, fpr_class = self.compute_fpr(fp=false_positives, tn=true_negatives)
        fnr_micro, fnr_macro, fnr_class = self.compute_fnr(fn=false_negatives, tp=true_positives)

        ppv_micro, ppv_macro, ppv_class = self.compute_ppv(tp=true_positives, fp=false_positives)
        npv_micro, npv_macro, npv_class = self.compute_npv(tn=true_negatives, fn=false_negatives)
        neg_lr_micro = self.compute_neg_lr(tpr=tpr_micro, tnr=tnr_micro)
        neg_lr_class = self.compute_neg_lr(tpr=tpr_class, tnr=tnr_class)
        neg_lr_macro = compute_macro_stats(self.compute_neg_lr(tpr=tpr_class, tnr=tnr_class))

        pos_lr_micro = self.compute_pos_lr(tpr=tpr_micro, tnr=tnr_micro)
        pos_lr_class = self.compute_pos_lr(tpr=tpr_class, tnr=tnr_class)
        pos_lr_macro = compute_macro_stats(self.compute_pos_lr(tpr=tpr_class, tnr=tnr_class))

        dor_micro, dor_macro, dor_class = self.compute_dor(tp=true_positives, tn=true_negatives, fp=false_positives,
                                                           fn=false_negatives)

        yi_micro = self.compute_younden_index(tpr=tpr_micro, tnr=tnr_micro)
        yi_class = self.compute_younden_index(tpr=tpr_class, tnr=tnr_class)
        yi_macro = compute_macro_stats(self.compute_younden_index(tpr=tpr_class, tnr=tnr_class))

        return CMResults(
            overall_accuracy=overall_acc,
            sensitivity_micro=sensitivity_micro,
            sensitivity_macro=sensitivity_macro,
            sensitivity_class=sensitivity_class,
            #
            specificity_micro=specificity_micro,
            specificity_macro=specificity_macro,
            specificity_class=specificity_class,
            #
            precision_micro=precision_micro,
            precision_macro=precision_macro,
            precision_class=precision_class,
            #
            recall_micro=recall_micro,
            recall_macro=recall_macro,
            recall_class=recall_class,
            #
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            f1_class=f1_class,
            #
            accuracy_micro=acc_micro,
            accuracy_macro=acc_macro,
            accuracy_class=acc_class,
            #
            true_positive_rate_micro=tpr_micro,
            true_positive_rate_macro=tpr_macro,
            true_positive_rate_class=tpr_class,
            #
            true_negative_rate_micro=tnr_micro,
            true_negative_rate_macro=tnr_macro,
            true_negative_rate_class=tnr_class,
            #
            false_positive_rate_micro=fpr_micro,
            false_positive_rate_macro=fpr_macro,
            false_positive_rate_class=fpr_class,
            #
            false_negative_rate_micro=fnr_micro,
            false_negative_rate_macro=fnr_macro,
            false_negative_rate_class=fnr_class,
            #
            positive_pred_value_micro=ppv_micro,
            positive_pred_value_macro=ppv_macro,
            positive_pred_value_class=ppv_class,
            #
            negative_pred_value_micro=npv_micro,
            negative_pred_value_macro=npv_macro,
            negative_pred_value_class=npv_class,
            #
            negative_likelihood_ratio_micro=neg_lr_micro,
            negative_likelihood_ratio_macro=neg_lr_macro,
            negative_likelihood_ratio_class=neg_lr_class,
            #
            positive_likelihood_ratio_micro=pos_lr_micro,
            positive_likelihood_ratio_macro=pos_lr_macro,
            positive_likelihood_ratio_class=pos_lr_class,
            #
            diagnostic_odd_ratio_micro=dor_micro,
            diagnostic_odd_ratio_macro=dor_macro,
            diagnostic_odd_ratio_class=dor_class,
            #
            younden_index_micro=yi_micro,
            younden_index_macro=yi_macro,
            younden_index_class=yi_class
        )


if __name__ == '__main__':
    y_true = [0, 0, 2, 2, 2, 1, 2, 2, 1, 1, 2, 3, 1, 3, 2, 2, 0, 0, 2, 0, 1, 0, 2, 2, 1, 2, 0, 0, 0, 1, 0, 3, 2, 0, 2,
              0, 2, 1, 0, 1, 0, 3, 3, 0, 0, 0, 1, 2, 2, 2, 1, 1, 0, 1, 0, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 0, 0, 2, 3, 0,
              2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 2, 1, 1, 2, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 3, 1, 2, 2,
              0, 2, 2, 0, 1, 2, 1, 1, 2, 1, 0, 1, 2, 3, 2, 3, 2, 2, 3, 0, 1, 2, 2, 2, 0, 1, 0, 2, 1, 0, 2, 2, 0, 0, 2,
              3, 0, 1, 0, 0, 2, 2, 2, 0, 2, 2, 1, 0, 2, 0, 2, 1, 1, 0, 3, 1, 1, 1, 2, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2,
              3, 0, 1, 1, 2, 2, 1, 2, 2, 1, 2, 0, 1, 0, 2, 2, 2, 2, 3, 2, 2, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0,
              1, 2, 2, 2, 2, 0]
    y_pred = [0, 0, 3, 2, 0, 1, 2, 2, 1, 0, 2, 3, 3, 3, 2, 2, 0, 0, 2, 0, 1, 0, 2, 2, 0, 2, 0, 1, 1, 0, 2, 3, 2, 0, 2,
              0, 2, 1, 0, 0, 0, 3, 3, 0, 0, 0, 3, 2, 2, 3, 0, 1, 0, 1, 0, 2, 1, 2, 2, 2, 2, 0, 2, 1, 1, 3, 0, 2, 3, 0,
              2, 0, 2, 2, 2, 2, 1, 2, 3, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 0, 0, 0, 1, 2, 0, 3, 0, 3, 2, 2, 0,
              1, 2, 3, 1, 0, 2, 1, 2, 1, 3, 2, 1, 2, 2, 1, 3, 2, 2, 3, 0, 0, 2, 2, 0, 0, 3, 0, 2, 2, 0, 2, 3, 0, 2, 2,
              3, 0, 3, 0, 0, 2, 0, 2, 0, 0, 2, 3, 2, 2, 0, 2, 0, 2, 0, 3, 0, 0, 0, 3, 3, 1, 0, 0, 0, 0, 1, 1, 2, 1, 0,
              3, 0, 0, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 0, 3, 2, 2, 1, 1, 1, 0, 2, 2, 1, 3, 0, 2, 0, 0, 0, 2,
              1, 2, 0, 2, 2, 2]

    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    cmat = confusion_matrix(y_true, y_pred)

    stats_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    stats_micro =precision_recall_fscore_support(y_true, y_pred, average='micro')
    print('Micro F1 using sklearn: ', round(stats_micro[-2], 3))
    print('Macro F1 using sklearn: ', round(stats_macro[-2], 3))

    cmat_array = np.array(cmat)

    results_summary = dict()
    conf_mat_eval = CMMetrics()
    cmat_results: CMResults = conf_mat_eval.compute_metrics(conf_mat=cmat_array)

    print('Micro F1 using our code: ', round(cmat_results.f1_micro, 3))
    print('Micro F1 using our code: ', round(cmat_results.f1_macro, 3))