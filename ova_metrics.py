from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import numpy as np


def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report showing the main classification metrics

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.67      1.00      0.80         2
        class 1       0.00      0.00      0.00         1
        class 2       1.00      1.00      1.00         2
    <BLANKLINE>
    avg / total       0.67      0.80      0.72         5
    <BLANKLINE>

    """
    labels = unique_labels(y_true, y_pred)
    # if labels is None:
        # labels = unique_labels(y_true, y_pred)
    # else:
        # labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['{}'.format(l) for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        values += ["%d" % int(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["%0.2f" % float(v)]
    values += ['%d' % np.sum(s)]
    report += fmt % tuple(values)
    return report