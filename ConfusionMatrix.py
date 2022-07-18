# Python script for confusion matrix creation.

import warnings
from itertools import chain

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics._classification import (
    _check_zero_division,
    _check_set_wise_labels,
    multilabel_confusion_matrix,
    _prf_divide,
    _warn_prf,
)
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target, _FN_UNIQUE_LABELS
from sklearn.utils.validation import _num_samples


class ConfusionMatrix:
    def __init__(self, arrActual, arrPredicted):
        self.arrActual = arrActual
        self.arrPredicted = arrPredicted

    # Compute confusion matrix to evaluate the accuracy of a classification
    def confusion_matrix(
            self, y_true, y_pred, labels=None, sample_weight=None, normalize=None
    ):
        y_type, y_true, y_pred = self._check_targets(y_true, y_pred)
        if y_type not in ("binary", "multiclass"):
            raise ValueError("%s is not supported" % y_type)

        if labels is None:
            labels = self.unique_labels(y_true, y_pred)
        else:
            labels = np.asarray(labels)
            if np.all([l not in y_true for l in labels]):
                raise ValueError("At least one label specified must be in y_true")

        if sample_weight is None:
            sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
        else:
            sample_weight = np.asarray(sample_weight)

        self.check_consistent_length(y_true, y_pred, sample_weight)

        if normalize not in ["true", "pred", "all", None]:
            raise ValueError(
                "normalize must be one of {'true', 'pred', " "'all', None}"
            )

        n_labels = labels.size
        label_to_ind = {y: x for x, y in enumerate(labels)}
        # convert yt, yp into index
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

        # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

        # Choose the accumulator dtype to always have high precision
        if sample_weight.dtype.kind in {"i", "u", "b"}:
            dtype = np.int64
        else:
            dtype = np.float64

        cm = coo_matrix(
            (sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype,
        ).toarray()

        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
            cm = np.nan_to_num(cm)

        return cm

        # Check that y_true and y_pred belong to the same classification task

    #
    #     This converts multiclass or binary types to a common shape, and raises a
    #     ValueError for a mix of multilabel and multiclass targets, a mix of
    #     multilabel formats, for the presence of continuous-valued or multioutput
    #     targets, or for targets of different lengths
    def _check_targets(self, y_true, y_pred):
        self.check_consistent_length(y_true, y_pred)
        type_true = type_of_target(y_true)
        type_pred = type_of_target(y_pred)

        y_type = {type_true, type_pred}
        if y_type == {"binary", "multiclass"}:
            y_type = {"multiclass"}

        if len(y_type) > 1:
            raise ValueError(
                "Classification metrics can't handle a mix of {0} "
                "and {1} targets".format(type_true, type_pred)
            )

        # We can't have more than one value on y_type => The set is no more needed
        y_type = y_type.pop()

        # No metrics support "multiclass-multioutput" format
        if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
            raise ValueError("{0} is not supported".format(y_type))

        if y_type in ["binary", "multiclass"]:
            y_true = column_or_1d(y_true)
            y_pred = column_or_1d(y_pred)
            if y_type == "binary":
                unique_values = np.union1d(y_true, y_pred)
                if len(unique_values) > 2:
                    y_type = "multiclass"

        if y_type.startswith("multilabel"):
            y_true = csr_matrix(y_true)
            y_pred = csr_matrix(y_pred)
            y_type = "multilabel-indicator"

        return y_type, y_true, y_pred

    # Extract an ordered array of unique labels
    def unique_labels(self, *ys):
        if not ys:
            raise ValueError("No argument has been passed.")
        # Check that we don't mix label format

        ys_types = set(type_of_target(x) for x in ys)
        if ys_types == {"binary", "multiclass"}:
            ys_types = {"multiclass"}

        if len(ys_types) > 1:
            raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

        label_type = ys_types.pop()

        # Check consistency for the indicator format
        if (
                label_type == "multilabel-indicator"
                and len(set(check_array(y, ["csr", "csc", "coo"]).shape[1] for y in ys)) > 1
        ):
            raise ValueError(
                "Multi-label binary indicator input with " "different numbers of labels"
            )

        # Get the unique set of labels
        _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
        if not _unique_labels:
            raise ValueError("Unknown label type: %s" % repr(ys))

        ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

        # Check that we don't mix string type with number type
        if len(set(isinstance(label, str) for label in ys_labels)) > 1:
            raise ValueError("Mix of label input types (string and number)")

        return np.array(sorted(ys_labels))

    # Check that all arrays have consistent first dimensions.
    # Checks whether all objects in arrays have the same shape or length.
    def check_consistent_length(self, *arrays):
        lengths = [_num_samples(X) for X in arrays if X is not None]
        uniques = np.unique(lengths)
        if len(uniques) > 1:
            raise ValueError(
                "Found input variables with inconsistent numbers of"
                " samples: %r" % [int(l) for l in lengths]
            )

    # Accuracy classification score.
    # In multilabel classification, this function computes subset accuracy:
    # the set of labels predicted for a sample must *exactly* match the
    # corresponding set of labels in y_true.
    def accuracy_score(self, y_true, y_pred, normalize=True, sample_weight=None):
        # Compute accuracy for each possible representation
        y_type, y_true, y_pred = self._check_targets(y_true, y_pred)
        self.check_consistent_length(y_true, y_pred, sample_weight)
        if y_type.startswith("multilabel"):
            differing_labels = self.count_nonzero(y_true - y_pred, axis=1)
            score = differing_labels == 0
        else:
            score = y_true == y_pred

        return self._weighted_sum(score, sample_weight, normalize)

    # A variant of X.getnnz() with extension to weighting on axis 0
    # Useful in efficiently calculating multilabel metrics
    def count_nonzero(self, X, axis=None, sample_weight=None):
        if axis == -1:
            axis = 1
        elif axis == -2:
            axis = 0
        elif X.format != "csr":
            raise TypeError("Expected CSR sparse format, got {0}".format(X.format))

        # We rely here on the fact that np.diff(Y.indptr) for a CSR
        # will return the number of nonzero entries in each row.
        # A bincount over Y.indices will return the number of nonzeros
        # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
        if axis is None:
            if sample_weight is None:
                return X.nnz
            else:
                return np.dot(np.diff(X.indptr), sample_weight)
        elif axis == 1:
            out = np.diff(X.indptr)
            if sample_weight is None:
                # astype here is for consistency with axis=0 dtype
                return out.astype("intp")
            return out * sample_weight
        elif axis == 0:
            if sample_weight is None:
                return np.bincount(X.indices, minlength=X.shape[1])
            else:
                weights = np.repeat(sample_weight, np.diff(X.indptr))
                return np.bincount(X.indices, minlength=X.shape[1], weights=weights)
        else:
            raise ValueError("Unsupported axis: {0}".format(axis))

    def _weighted_sum(self, sample_score, sample_weight, normalize=False):
        if normalize:
            return np.average(sample_score, weights=sample_weight)
        elif sample_weight is not None:
            return np.dot(sample_score, sample_weight)
        else:
            return sample_score.sum()

    # Build a text report showing the main classification metrics
    def classification_report(
            self,
            y_true,
            y_pred,
            labels=None,
            target_names=None,
            sample_weight=None,
            digits=2,
            output_dict=False,
            zero_division="warn",
    ):
        y_type, y_true, y_pred = self._check_targets(y_true, y_pred)

        labels_given = True
        if labels is None:
            labels = self.unique_labels(y_true, y_pred)
            labels_given = False
        else:
            labels = np.asarray(labels)

        # labelled micro average
        micro_is_accuracy = (y_type == "multiclass" or y_type == "binary") and (
                not labels_given or (set(labels) == set(self.unique_labels(y_true, y_pred)))
        )

        if target_names is not None and len(labels) != len(target_names):
            if labels_given:
                warnings.warn(
                    "labels size, {0}, does not match size of target_names, {1}".format(
                        len(labels), len(target_names)
                    )
                )
            else:
                raise ValueError(
                    "Number of classes, {0}, does not match size of "
                    "target_names, {1}. Try specifying the labels "
                    "parameter".format(len(labels), len(target_names))
                )
        if target_names is None:
            target_names = ["%s" % l for l in labels]

        headers = ["precision", "recall", "f1-score", "support"]
        # compute per-class results without averaging
        p, r, f1, s = self.precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        rows = zip(target_names, p, r, f1, s)

        if y_type.startswith("multilabel"):
            average_options = ("micro", "macro", "weighted", "samples")
        else:
            average_options = ("micro", "macro", "weighted")

        if output_dict:
            report_dict = {label[0]: label[1:] for label in rows}
            for label, scores in report_dict.items():
                report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
        else:
            longest_last_line_heading = "weighted avg"
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, len(longest_last_line_heading), digits)
            head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
            report = head_fmt.format("", *headers, width=width)
            report += "\n\n"
            row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
            for row in rows:
                report += row_fmt.format(*row, width=width, digits=digits)
            report += "\n"

        # compute all applicable averages
        for average in average_options:
            if average.startswith("micro") and micro_is_accuracy:
                line_heading = "accuracy"
            else:
                line_heading = average + " avg"

            # compute averages with specified averaging method
            avg_p, avg_r, avg_f1, _ = self.precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average=average,
                sample_weight=sample_weight,
                zero_division=zero_division,
            )
            avg = [avg_p, avg_r, avg_f1, np.sum(s)]

            if output_dict:
                report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))
            else:
                if line_heading == "accuracy":
                    row_fmt_accuracy = (
                            "{:>{width}s} "
                            + " {:>9.{digits}}" * 2
                            + " {:>9.{digits}f}"
                            + " {:>9}\n"
                    )
                    report += row_fmt_accuracy.format(
                        line_heading, "", "", *avg[2:], width=width, digits=digits
                    )
                else:
                    report += row_fmt.format(
                        line_heading, *avg, width=width, digits=digits
                    )

        if output_dict:
            if "accuracy" in report_dict.keys():
                report_dict["accuracy"] = report_dict["accuracy"]["precision"]
            return report_dict
        else:
            return report

    # Compute precision, recall, F-measure and support for each class
    #
    #     The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    #     true positives and ``fp`` the number of false positives. The precision is
    #     intuitively the ability of the classifier not to label as positive a sample
    #     that is negative.
    #
    #     The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    #     true positives and ``fn`` the number of false negatives. The recall is
    #     intuitively the ability of the classifier to find all the positive samples.
    #
    #     The F-beta score can be interpreted as a weighted harmonic mean of
    #     the precision and recall, where an F-beta score reaches its best
    #     value at 1 and worst score at 0.
    #
    #     The F-beta score weights recall more than precision by a factor of
    #     ``beta``. ``beta == 1.0`` means recall and precision are equally important.
    #
    #     The support is the number of occurrences of each class in ``y_true``
    def precision_recall_fscore_support(
            self,
            y_true,
            y_pred,
            beta=1.0,
            labels=None,
            pos_label=1,
            average=None,
            warn_for=("precision", "recall", "f-score"),
            sample_weight=None,
            zero_division="warn",
    ):

        _check_zero_division(zero_division)
        if beta < 0:
            raise ValueError("beta should be >=0 in the F-beta score")
        labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

        # Calculate tp_sum, pred_sum, true_sum ###
        samplewise = average == "samples"
        MCM = multilabel_confusion_matrix(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            labels=labels,
            samplewise=samplewise,
        )
        tp_sum = MCM[:, 1, 1]
        pred_sum = tp_sum + MCM[:, 0, 1]
        true_sum = tp_sum + MCM[:, 1, 0]

        if average == "micro":
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        # Finally, we have all our sufficient statistics. Divide! #
        beta2 = beta ** 2

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        precision = _prf_divide(
            tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
        )
        recall = _prf_divide(
            tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
        )

        # warn for f-score only if zero_division is warn, it is in warn_for
        # and BOTH prec and rec are ill-defined
        if zero_division == "warn" and ("f-score",) == warn_for:
            if (pred_sum[true_sum == 0] == 0).any():
                _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

        # if tp == 0 F will be 1 only if all predictions are zero, all labels are
        # zero, and zero_division=1. In all other case, 0
        if np.isposinf(beta):
            f_score = recall
        else:
            denom = beta2 * precision + recall

            denom[denom == 0.0] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

        # Average the results
        if average == "weighted":
            weights = true_sum
            if weights.sum() == 0:
                zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
                # precision is zero_division if there are no positive predictions
                # recall is zero_division if there are no positive labels
                # fscore is zero_division if all labels AND predictions are
                # negative
                return (
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    None,
                )

        elif average == "samples":
            weights = sample_weight
        else:
            weights = None

        if average is not None:
            assert average != "binary" or len(precision) == 1
            precision = np.average(precision, weights=weights)
            recall = np.average(recall, weights=weights)
            f_score = np.average(f_score, weights=weights)
            true_sum = None  # return no support

        return precision, recall, f_score, true_sum

    def getAccurecyScore(self):
        results = self.confusion_matrix(self.arrActual, self.arrPredicted)
        accuracyScore = self.accuracy_score(self.arrActual, self.arrPredicted)
        classifyReport = self.classification_report(self.arrActual, self.arrPredicted)
        # print("Confusion Matrix :")
        # print(results)
        print("Accuracy Score :")
        print(accuracyScore)
        # print("Report :")
        # print(classifyReport)

    def getClassificationReport(self):
        classifyReport = self.classification_report(self.arrActual, self.arrPredicted,target_names=['positive','neutral','negative'])
        print(classifyReport)
