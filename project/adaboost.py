from haar_features import *


def stump_classify(data, column_idx, threshold, thresh_ineq):
    labels = np.ones((np.shape(data)[0], 1))
    if thresh_ineq == 'lt':
        labels[data[:, column_idx] <= threshold] = -1.0
    else:
        labels[data[:, column_idx] >= threshold] = -1.0
    return labels


def build_stump(data, labels, d, num_steps):
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).T
    m, n = np.shape(data_matrix)
    best_classifier = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        column_min = data_matrix[:, i].min()
        column_max = data_matrix[:, i].max()
        step_size = (column_max - column_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = column_min + float(j) * step_size
                prediction = stump_classify(data_matrix, i, threshold, inequal)
                errors = np.mat(np.ones((m, 1)))
                errors[prediction == label_matrix] = 0
                weighted_error = d.T * errors
                if weighted_error <= 1 - weighted_error:
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_class_est = prediction.copy()
                        best_classifier['index'] = i
                        best_classifier['thresh'] = threshold
                        best_classifier['ineq'] = inequal
                    break

    return best_classifier, min_error, best_class_est


def build_stump_false_positive(data, labels, d, num_steps):
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).T
    m, n = np.shape(data_matrix)
    best_classifier = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        column_min = data_matrix[:, i].min()
        column_max = data_matrix[:, i].max()
        step_size = (column_max - column_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = column_min + float(j) * step_size
                prediction = stump_classify(data_matrix, i, threshold, inequal)
                errors = np.mat(np.ones((m, 1)))
                errors[prediction == label_matrix] = 0
                weighted_error = d.T * errors
                if weighted_error <= 1 - weighted_error:
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_class_est = prediction.copy()
                        best_classifier['index'] = i
                        best_classifier['thresh'] = threshold
                        best_classifier['ineq'] = inequal
                    break

    return best_classifier, min_error, best_class_est


def build_stump_false_negative(data, labels, d, num_steps):
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).T
    m, n = np.shape(data_matrix)
    best_classifier = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        column_min = data_matrix[:, i].min()
        column_max = data_matrix[:, i].max()
        step_size = (column_max - column_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = column_min + float(j) * step_size
                prediction = stump_classify(data_matrix, i, threshold, inequal)
                errors = np.mat(np.ones((m, 1)))
                errors[prediction == label_matrix] = 0
                weighted_error = d.T * errors
                if weighted_error <= 1 - weighted_error:
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_class_est = prediction.copy()
                        best_classifier['index'] = i
                        best_classifier['thresh'] = threshold
                        best_classifier['ineq'] = inequal
                    break

    return best_classifier, min_error, best_class_est


def train(data, labels, classifier_number_max, num_steps=10.0):
    weak_classifier_arr = []
    train_err_arr = []
    m = np.shape(data)[0]
    d = np.mat(np.ones((m, 1)) / m)
    pre_integration_labels_mat = np.mat(np.zeros((m, 1)))
    for i in range(classifier_number_max):
        best_classifier, error, pre_labels_mat = build_stump(data, labels, d, num_steps)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_classifier['alpha'] = alpha
        weak_classifier_arr.append(best_classifier)
        expon = np.multiply(-1 * alpha * np.mat(labels).T, pre_labels_mat)
        d = np.multiply(d, np.exp(expon))
        d = d / d.sum()
        pre_integration_labels_mat += alpha * pre_labels_mat
        integration_errors = np.multiply(np.sign(pre_integration_labels_mat) != np.mat(labels).T, np.ones((m, 1)))
        error_rate = integration_errors.sum() / m
        train_err_arr.append(error_rate)
        best_classifier['accuracy'] = 1 - error_rate
        print("Round # %d: error =  %f" % (i+1, error_rate))
        if error_rate == 0.0:
            break

    return weak_classifier_arr


def classify(test_data, classifiers):
    data_mat = np.mat(test_data)
    m = np.shape(data_mat)[0]
    pre_integration_labels_mat = np.mat(np.zeros((m, 1)))
    for i in range(len(classifiers)):
        class_est = stump_classify(data_mat, classifiers[i]['index'],
                                   classifiers[i]['thresh'],
                                   classifiers[i]['ineq'])  # call stump classify
        pre_integration_labels_mat += classifiers[i]['alpha'] * class_est
    # print("The estimate prediction value after integrating the weak classifiers: %s" % str(pre_integration_labels_mat))
    return np.sign(pre_integration_labels_mat), pre_integration_labels_mat
