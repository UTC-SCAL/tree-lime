from sklearn import tree
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus


def get_sample(point, means, stds):
    k = np.random.rand(len(stds))
    k_stds = np.multiply(k, stds)
    return np.add(point, k_stds)


def get_points_around(point, predictor, samples, samples_size=5, classes_num=2):
    # TODO apply a different criteria for categorical data
    means = np.average(samples, axis=0)
    stds = np.std(samples, axis=0)
    res_samples = []
    for _ in range(samples_size):
        current_sample = get_sample(point, means, stds)
        res_samples.append(current_sample)
    return res_samples, predictor(res_samples)


def predict_tree(x, y):
    clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=20)
    clf = clf.fit(x, y)
    return clf


def visualize_tree(tree_clf, feature_names):
    dot_data = StringIO()
    export_graphviz(tree_clf,
                    out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=False,
                    feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph
