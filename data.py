from sklearn import tree
import numpy as np


def get_sample(point, means, stds):
    k = np.random.rand()
    k_stds = k*stds
    return np.add(point, k_stds)


def get_points_around(point, predictor, samples, samples_size=5, classes_num=2):
    # todo for now assume that all features are continuous
    means = np.average(samples, axis=0)
    stds = np.std(samples, axis=0)
    res_samples = []
    for i in range(samples_size):
        current_sample = get_sample(point, means, stds)
        res_samples.append(current_sample)
    return res_samples, predictor(res_samples)


def predict_tree(x, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    return clf


def visualize_tree(clf):
    tree.plot_tree(clf)