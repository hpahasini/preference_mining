import csv
import json
import numbers
import warnings
from inspect import isclass

import numpy as np
import pandas as pd
from joblib.parallel import get_active_backend, delayed, Parallel
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from \
    sklearn.linear_model._base import SparseCoefMixin, LinearClassifierMixin
from sklearn.linear_model._logistic import _logistic_regression_path
from sklearn.metrics._classification import _check_targets, precision_recall_fscore_support, \
    _check_zero_division, _check_set_wise_labels, multilabel_confusion_matrix, _prf_divide, _warn_prf
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm._base import _fit_liblinear
from sklearn.utils.extmath import safe_sparse_dot, row_norms, softmax
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import _num_samples, check_X_y, check_non_negative

global df

# Read the training dataset
df = pd.read_csv('corpus.csv')

col = ['label', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.text).toarray()

Train_X, Test_X, Train_Y, Test_Y = train_test_split(df['text'],
                                                    df['label'],
                                                    test_size=0.2,random_state=None)
Train_X, Val_X, Train_Y, Val_Y = train_test_split(Train_X,
                                                    Train_Y,
                                                    test_size=0.2,random_state=None)

Encoder = LabelEncoder()
Train_Y_En = Encoder.fit_transform(Train_Y)
Test_Y_En = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(Train_X)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the
# dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is
# always the same as the original input sample size but the samples are drawn with replacement if `bootstrap=True` (
# default)


# Linear Support Vector Classification.

# Similar to SVC with parameter kernel='linear', but implemented in terms of
# liblinear rather than libsvm, so it has more flexibility in the choice of
# penalties and loss functions and should scale better to large numbers of
# samples.
class LinearSVC(BaseEstimator, LinearClassifierMixin,
                SparseCoefMixin):

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss

    # Fit the model according to the given training data.
    def fit(self, X, y, sample_weight=None):

        # FIXME Remove l1/l2 support in 0.23 ----------------------------------
        msg = ("loss='%s' has been deprecated in favor of "
               "loss='%s' as of 0.16. Backward compatibility"
               " for the loss='%s' will be removed in %s")

        if self.loss in ('l1', 'l2'):
            old_loss = self.loss
            self.loss = {'l1': 'hinge', 'l2': 'squared_hinge'}.get(self.loss)
            warnings.warn(msg % (old_loss, self.loss, old_loss, '0.23'),
                          FutureWarning)
        # ---------------------------------------------------------------------

        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)

        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64, order="C",
                         accept_large_sparse=False)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
            X, y, self.C, self.fit_intercept, self.intercept_scaling,
            self.class_weight, self.penalty, self.dual, self.verbose,
            self.max_iter, self.tol, self.random_state, self.multi_class,
            self.loss, sample_weight=sample_weight)

        if self.multi_class == "crammer_singer" and len(self.classes_) == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0]).reshape(1, -1)
            if self.fit_intercept:
                intercept = self.intercept_[1] - self.intercept_[0]
                self.intercept_ = np.array([intercept])

        return self


# >>>>>>>>>>>>>>>>>>>>>----------done
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 


# Ensure that target y is of a non-regression type
def check_classification_targets(y):
    y_type = type_of_target(y)
    if y_type not in ['binary', 'multiclass', 'multiclass-multioutput',
                      'multilabel-indicator', 'multilabel-sequences']:
        raise ValueError("Unknown label type: %r" % y_type)


#    Naive Bayes classifier for multinomial models
#
#     The multinomial Naive Bayes classifier is suitable for classification with
#     discrete features (e.g., word counts for text classification). The
#     multinomial distribution normally requires integer feature counts. However,
#     in practice, fractional counts such as tf-idf may also work.

class MultinomialNB(_BaseDiscreteNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _more_tags(self):
        return {'requires_positive_X': True}

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)


# >>>>>>>>>>>>>>>>>>>>>----------done
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 

#  Logistic Regression (aka logit, MaxEnt) classifier.
#
#     In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
#     scheme if the 'multi_class' option is set to 'ovr', and uses the
#     cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
#     (Currently the 'multinomial' option is supported only by the 'lbfgs',
#     'sag', 'saga' and 'newton-cg' solvers.)
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    # Fit the model according to the given training data.
    def fit(self, X, y, sample_weight=None):

        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if self.penalty == 'elasticnet':
            if (not isinstance(self.l1_ratio, numbers.Number) or
                    self.l1_ratio < 0 or self.l1_ratio > 1):
                raise ValueError("l1_ratio must be between 0 and 1;"
                                 " got (l1_ratio=%r)" % self.l1_ratio)
        elif self.l1_ratio is not None:
            warnings.warn("l1_ratio parameter is only used when penalty is "
                          "'elasticnet'. Got "
                          "(penalty={})".format(self.penalty))
        if self.penalty == 'none':
            if self.C != 1.0:  # default values
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio "
                    "parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = 'l2'
        else:
            C_ = self.C
            penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        if solver == 'lbfgs':
            _dtype = np.float64
        else:
            _dtype = [np.float64, np.float32]

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order="C",
                         accept_large_sparse=solver != 'liblinear')
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        multi_class = _check_multi_class(self.multi_class, solver,
                                         len(self.classes_))

        if solver == 'liblinear':
            if effective_n_jobs(self.n_jobs) != 1:
                warnings.warn("'n_jobs' > 1 does not have any effect when"
                              " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                              " = {}.".format(effective_n_jobs(self.n_jobs)))
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol, self.random_state,
                sample_weight=sample_weight)
            self.n_iter_ = np.array([n_iter_])
            return self

        if solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ['sag', 'saga']:
            prefer = 'threads'
        else:
            prefer = 'processes'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer=prefer))(
            path_func(X, y, pos_class=class_, Cs=[C_],
                      l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
                      tol=self.tol, verbose=self.verbose, solver=solver,
                      multi_class=multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      penalty=penalty, max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self

    # Probability estimates
    def predict_proba(self, X):

        check_is_fitted(self)

        ovr = (self.multi_class in ["ovr", "warn"] or
               (self.multi_class == 'auto' and (self.classes_.size <= 2 or
                                                self.solver == 'liblinear')))
        if ovr:
            return super()._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)

    # Predict logarithm of probability estimates
    def predict_log_proba(self, X):

        return np.log(self.predict_proba(X))


# >>>>>>>>>>>>>>>>>>>>>----------done
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 
# >>>>>>>>>>>>>>>>>>>>>----------done 

def _check_solver(solver, penalty, dual):
    all_solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    if solver not in all_solvers:
        raise ValueError("Logistic Regression supports only solvers in %s, got"
                         " %s." % (all_solvers, solver))

    all_penalties = ['l1', 'l2', 'elasticnet', 'none']
    if penalty not in all_penalties:
        raise ValueError("Logistic Regression supports only penalties in %s,"
                         " got %s." % (all_penalties, penalty))

    if solver not in ['liblinear', 'saga'] and penalty not in ('l2', 'none'):
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
                         "got %s penalty." % (solver, penalty))
    if solver != 'liblinear' and dual:
        raise ValueError("Solver %s supports only "
                         "dual=False, got dual=%s" % (solver, dual))

    if penalty == 'elasticnet' and solver != 'saga':
        raise ValueError("Only 'saga' solver supports elasticnet penalty,"
                         " got solver={}.".format(solver))

    if solver == 'liblinear' and penalty == 'none':
        raise ValueError(
            "penalty='none' is not supported for the liblinear solver"
        )

    return solver


def _check_multi_class(multi_class, solver, n_classes):
    if multi_class == 'auto':
        if solver == 'liblinear':
            multi_class = 'ovr'
        elif n_classes > 2:
            multi_class = 'multinomial'
        else:
            multi_class = 'ovr'
    if multi_class not in ('multinomial', 'ovr'):
        raise ValueError("multi_class should be 'multinomial', 'ovr' or "
                         "'auto'. Got %s." % multi_class)
    if multi_class == 'multinomial' and solver == 'liblinear':
        raise ValueError("Solver %s does not support "
                         "a multinomial backend." % solver)
    return multi_class


# Checks if the estimator is fitted by verifying the presence of
# fitted attributes (ending with a trailing underscore) and otherwise
# raises a NotFittedError with the given message
def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})


# Determine the number of jobs that can actually run in parallel
def effective_n_jobs(n_jobs=-1):
    backend, _ = get_active_backend()
    return backend.effective_n_jobs(n_jobs=n_jobs)








models = [

    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(verbose=1,
                       solver='liblinear',
                       random_state=0,
                       C=5,
                       penalty='l2',
                       max_iter=1000),

]


# for model in models:
#     model_name = model.__class__.__name__
#     model_fit = model.fit(X_train_tfidf, Train_Y)
#     # print("Testing with: ", model_name)
#     with open("@AmilDilshan_1X.csv", 'r') as data_file:
#         reader = csv.reader(data_file, delimiter='\t')
#         for line in reader:
#             for field in line:
#                 print(field, ":")
#                 print(model_fit.predict(count_vect.transform([field])))

#         print("---------------")
#         # print("---------------")


# In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must
# *exactly* match the corresponding set of labels in y_true.

def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)


# Check that y_true and y_pred belong to the same classification task

# This converts multiclass or binary types to a common shape, and raises a ValueError for a mix of multilabel and
# multiclass targets, a mix of multilabel formats, for the presence of continuous-valued or multioutput targets,
# or for targets of different lengths
def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


# A variant of X.getnnz() with extension to weighting on axis 0 Useful in efficiently calculating multilabel metrics.
def count_nonzero(X, axis=None, sample_weight=None):
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != 'csr':
        raise TypeError('Expected CSR sparse format, got {0}'.format(X.format))

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
            return out.astype('intp')
        return out * sample_weight
    elif axis == 0:
        if sample_weight is None:
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            weights = np.repeat(sample_weight, np.diff(X.indptr))
            return np.bincount(X.indices, minlength=X.shape[1],
                               weights=weights)
    else:
        raise ValueError('Unsupported axis: {0}'.format(axis))


def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=2, output_dict=False,
                          zero_division="warn"):
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    labels_given = True
    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)

    # labelled micro average
    micro_is_accuracy = ((y_type == 'multiclass' or y_type == 'binary') and
                         (not labels_given or
                          (set(labels) == set(unique_labels(y_true, y_pred)))))

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}"
                    .format(len(labels), len(target_names))
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ['%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight,
                                                  zero_division=zero_division)
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith('multilabel'):
        average_options = ('micro', 'macro', 'weighted', 'samples')
    else:
        average_options = ('micro', 'macro', 'weighted')

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers,
                                          [i.item() for i in scores]))
    else:
        longest_last_line_heading = 'weighted avg'
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += '\n'

    # compute all applicable averages
    for average in average_options:
        if average.startswith('micro') and micro_is_accuracy:
            line_heading = 'accuracy'
        else:
            line_heading = average + ' avg'

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels,
            average=average, sample_weight=sample_weight,
            zero_division=zero_division)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(
                zip(headers, [i.item() for i in avg]))
        else:
            if line_heading == 'accuracy':
                row_fmt_accuracy = '{:>{width}s} ' + \
                                   ' {:>9.{digits}}' * 2 + ' {:>9.{digits}f}' + \
                                   ' {:>9}\n'
                report += row_fmt_accuracy.format(line_heading, '', '',
                                                  *avg[2:], width=width,
                                                  digits=digits)
            else:
                report += row_fmt.format(line_heading, *avg,
                                         width=width, digits=digits)

    if output_dict:
        if 'accuracy' in report_dict.keys():
            report_dict['accuracy'] = report_dict['accuracy']['precision']
        return report_dict
    else:
        return report


def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None,
                                    pos_label=1, average=None,
                                    warn_for=('precision', 'recall',
                                              'f-score'),
                                    sample_weight=None,
                                    zero_division="warn"):
    _check_zero_division(zero_division)
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    labels = _check_set_wise_labels(y_true, y_pred, average, labels,
                                    pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == 'samples'
    MCM = multilabel_confusion_matrix(y_true, y_pred,
                                      sample_weight=sample_weight,
                                      labels=labels, samplewise=samplewise)
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(tp_sum, pred_sum, 'precision',
                            'predicted', average, warn_for, zero_division)
    recall = _prf_divide(tp_sum, true_sum, 'recall',
                         'true', average, warn_for, zero_division)

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, "true nor predicted", 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    None)

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def writeJson(modelIndex):
    model = models[modelIndex]
    model_name = model.__class__.__name__
    model_fit = model.fit(X_train_tfidf, Train_Y)
    #print("Selected Model: ", model_name)
    categorized = {}
    with open("@AmilDilshan_1X.csv", 'r') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        count = 0

        # write_file = model_name +".json"
        for line in reader:
            for field in line:
                print(field, ":")
                print(model_fit.predict(count_vect.transform([field])))

                # ___________________________________________________________________________________________________
                cat = model_fit.predict(count_vect.transform([field]))
                if cat[0] not in categorized.keys():
                    categorized[cat[0]] = [field]
                else:
                    categorized[cat[0]].append(field)
                count = count + 1
        with open('data2222.json', 'w') as f:
            json.dump(categorized, f, ensure_ascii=False)
        # print(categorized)
        print("+++++++++++++++++++++++++++++++")
        print("number of rows... ", count)
        print("+++++++++++++++++++++++++++++++")


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

mdname=[]
acscore=[]
# Calculate accuracy
selectedScore = 0
selectedModel = -1
for model in models:
    print("_____________________________________", models.index(model))
    model_name = model.__class__.__name__
    mdname.append(model_name)
    model_predict = model.fit(Train_X_Tfidf, Train_Y_En)
    predictions = model_predict.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    accuracyScore = accuracy_score(predictions, Test_Y_En)
    acscore.append(accuracyScore)
    classification = classification_report(predictions, Test_Y_En)
    confusionMatrix = ConfusionMatrix(predictions, Test_Y_En)
    #print(model_name, " -> Accuracy Score -> ", accuracyScore)
    #print(model_name, " -> Classification Report -> ")
    #print(classification)

    if accuracyScore > selectedScore:
        selectedScore = accuracyScore
        selectedModel = models.index(model)
writeJson(selectedModel)

