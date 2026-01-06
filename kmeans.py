import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=',')

def initialise_centroids(data, k):
    # Randomly select k instances of the dataset, do not replace so we get k different rows
    return data[np.random.choice(data.shape[0], size=k, replace=False)]

def get_distance(query_data, query, p=2):
    # Calculate the distances between a single query point in feature space and a collection of other data points in
    # the same feature space (training instances).

    # query_data - numpy array containing feature data
    # query - numpy array containing query
    # returns numpy array containing the calculated distances from the query point to each of the individual feature
    # instances

    # Minkowski - (sum (abs (xi - yi) ^ p)) ^ (1/p)
    # https://en.wikipedia.org/wiki/Minkowski_distance

    distance = np.sum(np.abs(query_data - query) ** p, axis=1) ** (1/p)
    return distance

def assign_centroids(query_data, current_centroids):
    # Calculate the distance between each centroid and all feature data instances
    # Return centroid index that is closest to each individual feature instance
    # return an array with integer index values such that
    # e.g. 3 centroids, 10 features, [2, 0, 1. . . ]
    # first feature instance is closest to centroid 2, second feature closest to centroid 0 etc
    number_of_data_points = query_data.shape[0]
    number_of_centroids = current_centroids.shape[0]

    distances = np.zeros((number_of_data_points, number_of_centroids))

    for i in range(number_of_centroids):
        distances[:, i] = get_distance(query_data, current_centroids[i])

    # Get the indices of the minimum values along axis 1
    centroids = np.argmin(distances, axis=1)

    return centroids

def move_centroids(feature_data, centroid_indices, current_centroids):
    # Generate the new position of the centroids by calculating the mean of the datapoints assigned to each specific
    # centroid.
    # Return an array containing the new centroids
    number_of_data_points = current_centroids.shape[1] # dimension of our data
    number_of_centroids = current_centroids.shape[0]

    new_centroids = np.zeros((number_of_centroids, number_of_data_points))

    # For each centroid, calculate the mean of all assigned points
    for i in range(number_of_centroids):
        # Find all points assigned to centroid i
        assigned_points = feature_data[centroid_indices == i]
        new_centroids[i] = np.mean(assigned_points, axis=0)

    return new_centroids

def get_inertia_cost(feature_data, centroid_indices, current_centroids):
    # Basically - average distance from points to centroids

    # Take in as arguments the feature data, an array containing the centroid indices assigned to each feature instance,
    # and the current array of centroids.
    # Calculate and return the current inertia (distortion cost) function
    inertia_cost = 0.0

    for i in range(feature_data.shape[0]):
        # Get the centroid this point is assigned to
        assigned_centroid = current_centroids[centroid_indices[i]]
        inertia_cost += get_distance(feature_data[i:i + 1], assigned_centroid)[0] ** 2

    return inertia_cost

def calculate_ch_cost(feature_data, centroid_indices, current_centroids):
    # centroid_indices - array matching centroid number to feature data
    # current centroids - k number of current centroids

    # k clusters
    # CH numerator = BCSS/(k-1)
    # n data points
    # CH denominator = WCSS/(n-k)

    k = current_centroids.shape[0]
    # We need to divide by k-1 - so k=1 means we would be dividing be zero
    # We need to divide by n-k so k cannot equal n, or we will have another divide by zero
    if k == 1 or k == feature_data.shape[0]:
        return 0.0

    numerator = bcss(feature_data, centroid_indices, current_centroids) / (k - 1)
    denominator = wcss(feature_data, centroid_indices, current_centroids) / (feature_data.shape[0] - k)

    cost = numerator/denominator
    return cost

def bcss(feature_data, centroid_indices, current_centroids):

    # BCSS - Between-Cluster Sum of Squares - "the weighted sum of squared Euclidean distances between each cluster
    # centroid (mean) and the overall data centroid (mean)" https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
    # n_i number of points in a cluster
    # c_i is the current centroid
    # u - overall centroid / mean
    # BCSS = Sum n_i (abs(c_i - u))^2

    dataset_mean = np.mean(feature_data, axis=0)
    # How many points are assigned to each cluster
    clusters, cluster_counts = np.unique(centroid_indices, return_counts=True)

    # squared distances from each centroid to overall mean
    distances = get_distance(current_centroids, dataset_mean) ** 2
    # weighted sum of squared distances
    return np.sum(cluster_counts * distances)


def wcss(feature_data, centroid_indices, current_centroids):
    # Within Cluster Sum of Squares
    # the sum of the squared Euclidean distances between the data points and their respective centroids

    # abs(x - c_i) ^ 2
    # Get the centroid for every point based on its cluster assignment
    assigned_centroids = current_centroids[centroid_indices]

    # squared distances from every point to its assigned centroid
    squared_distances = get_distance(feature_data, assigned_centroids) ** 2

    # Sum all squared distances
    return np.sum(squared_distances)

def restart_KMeans(feature_data, number_of_centroids, iterations, restarts):

    best_cost = 0.0
    best_centroids = None
    best_indices = None

    for i in range(1, restarts + 1):
        current_centroids = initialise_centroids(feature_data, k=number_of_centroids)

        for j in range(1, iterations + 1):
            # inner loop of k-means, assigning & moving centroids iterations times
            centroid_indices = assign_centroids(feature_data, current_centroids)
            new_centroids = move_centroids(feature_data , centroid_indices, current_centroids)
            current_centroids = new_centroids

        # Need the indexes so call assign_centroids again
        current_iteration_indices = assign_centroids(feature_data, current_centroids)
        current_iteration_cost = get_inertia_cost(feature_data, current_iteration_indices, current_centroids)

        if i == 1 or current_iteration_cost < best_cost:
            best_centroids = current_centroids
            best_indices = current_iteration_indices
            best_cost = current_iteration_cost

    ch_cost = calculate_ch_cost(feature_data, best_indices, best_centroids)

    return best_centroids, best_cost, ch_cost

def main():
    best_centroids = []
    costs = []
    ch_costs = []

    for k in np.arange(1,11):
        best_centroid, best_cost, ch_cost = restart_KMeans(data, k, 10, 10)
        # Something to look to know the run is progressing
        print("k = {},\tinertia cost = {},\tch_cost = {}".format(k, best_cost,ch_cost))
        best_centroids.append(best_centroid)
        costs.append(best_cost)
        ch_costs.append(ch_cost)

    # Elbow graph plot
    plt.plot(np.arange(1,11), costs)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Cost")
    plt.title("K-means elbow graph")
    plt.xticks(np.arange(1, 11))
    plt.show()

    # CH cost graph
    plt.plot(np.arange(1, 11), ch_costs)
    plt.xlabel("Number of Clusters")
    plt.ylabel("CH Cost")
    plt.title("CH cost graph")
    plt.xticks(np.arange(1, 11))
    plt.show()

if __name__ == '__main__':
    main()

