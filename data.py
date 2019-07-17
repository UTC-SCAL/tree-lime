from sklearn import tree
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# TODO convert this code into python classes


def divide_samples(samples, clusters_range=range(2, 11)):
    """
       use a clustering technique in order to divide the data first
       :param clusters_range:
       :param samples:
       :return:
       """
    best_score = 0
    best_labels = None
    best_n = 0
    clf = None
    for n_clusters in clusters_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(samples)
        silhouette_avg = silhouette_score(samples, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            clf = clusterer
            best_labels = cluster_labels
            best_n = n_clusters
    return clf, best_labels, best_n


def divide_samples_by_labels(samples, labels, clusters_number):
    clusters = dict()
    for i in range(clusters_number):
        clusters[i] = []
    for i, sample in enumerate(samples):
        current_cluster = labels[i]
        clusters[current_cluster].append(sample)
    return clusters


def get_best_trees():
    pass


def get_sample(point, means, stds):
    k = 2 * np.random.rand(len(stds))
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
    clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=20)
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
