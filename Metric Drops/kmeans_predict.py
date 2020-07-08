# An import that allows kMeans to make predictions.
# The 'predict' function makes use of distances_to_centroid.
#
# William King
# 950178

import numpy as np

#Used to calculate the distance between a datapoint and its centroid.
def distances_to_centroid(data, model):
    distances = []
    for i in range(len(data)):
        datapoint = data[i]
        centroid = model.cluster_centers_[model.labels_[i]]
        euclid_distance = euclidean_distance(centroid, datapoint)
        distances.append(euclid_distance)
    return distances

#Calculates euclidean distance between 2 points (any dimension).
def euclidean_distance(centroid, datapoint):
    distance = 0
    for i in range(len(datapoint)):
        squared_difference = (centroid[i] - datapoint[i]) ** 2
        distance = distance + squared_difference
    return distance ** 0.5

#Return 'prediction' labels of kMeans based on the threshold.
#Points with the greatest distances from their centroids
#are more likely to be marked as outliers.
def predict(model, data, contamination):
    #Find distances of each datapoint to its allocated centroid.
    distances = np.array(distances_to_centroid(data, model))
    sortedList = np.sort(distances)

    #Set threshold based on percentage of 'outliers'.
    outlier_fraction = contamination
    number_of_outliers = int(len(data) * outlier_fraction)
    threshold = (sortedList[-number_of_outliers:][::-1]).min()

    #Set 'predicted' anomaly labels. 1 = anomaly.
    anomaly_labels = [0 if L < threshold else 1 for L in distances]

    return anomaly_labels
