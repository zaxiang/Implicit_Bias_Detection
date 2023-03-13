from sklearn.cluster import kmeans_plusplus
import numpy as np

class KMeans_custom():
    # This function initializes the KMeans class
    def __init__(self, k, num_iter, order):
        # Set a seed for easy debugging and evaluation
        np.random.seed(42)
        
        # This variable defines how many clusters to create
        # default is 3
        self.k = k

        # This variable defines how many iterations to recompute centroids
        # default is 1000
        self.num_iter = num_iter

        # This variable stores the coordinates of centroids
        self.centers = None

        # This variable defines whether it's K-Means or K-Medians
        # an order of 2 uses Euclidean distance for means
        # an order of 1 uses Manhattan distance for medians
        # default is 2
        if order == 1 or order == 2:
            self.order = order
        else:
            raise Exception("Unknown Order")     
            
    def Euclidean_distance(self, sub_matrix):
        #distance = np.linalg.norm(sub_matrix, axis = 1) #ini 

        distance = []
        for sub in sub_matrix: #for each centroids
            centroid_dis = 0
            #first 8 col: numerical feature: index 0-7
            for i in range(0, 8):
                dis_part = (sub[i])**2
                centroid_dis += dis_part
            #next 5 col: arrest one-hot-encoding: index 8-12
            for i in range(8, 13):
                dis_part = (1/5 * (sub[i])**2) #assign alpha
                centroid_dis += dis_part
            #next 27 col: charge one-hot-encoding: index 13-39: 
            for i in range(13, 13+27):
                dis_part = (1/27 * (sub[i])**2) #assign alpha
                centroid_dis += dis_part
                
                #Removed Sex
#             #next 2 col: sex one-hot-encoding: 40-41
#             for i in range(40, 42):
#                 dis_part = (1/2 * (sub[i])**2) #assign alpha
#                 centroid_dis += dis_part

            centroid_dis = np.sqrt(centroid_dis)
            distance.append(centroid_dis)

        return distance
    
    # This function fits the model with input data (training)
    def fit(self, X):
        # m, n represent the number of rows (observations) and columns (positions in each coordinate)
        #(208260, 45)
        m, n = X.shape 

        # self.centers are a 2d-array of 
        # (number of clusters, number of dimensions of our input data)
        self.centers = np.zeros((self.k, n))

        # self.cluster_idx represents the cluster index for each observation
        # which is a 1d-array of (number of observations)
        self.cluster_idx = np.zeros(m)

        # Initialize self.centers:
        # For each dimension (feature) in X, use the 10th percentile and 
        # the 90th percentile to form a uniform distribution. 
        # Then, initialize the values of each center by randomly selecting values from the distributions.
        # This method is by no means the best initialization method.
#         for i in range(n):
#             self.centers[:,i] = np.random.uniform(np.percentile(X[:, i], 10), np.percentile(X[:, i], 90), self.k)

        #Another way to Initialize self.centers: using k-means++ to iteratively find the centroids
        centers_init, indices = kmeans_plusplus(X, n_clusters=self.k, random_state=0)
        self.centers = X[indices]
        
        for i in range(self.num_iter):
            if i % 100 == 0:
                print(str(i), 'th iteration:')
        
            # new_centers are a 2d-array of 
            # (number of clusters, number of dimensions of our input data)
            new_centers = np.zeros((self.k, n))

            # calculate the distance and create cluster index for each observation:
            # calculate the distance between each observation and each centroid
            # using specified self.order. Then, derive the cluster index for 
            # each observation based on the minimum distance between an observation and 
            # each of the centers.
            cluster_idx = []
            for j in range(m):
                sub_matrix = self.centers - X[j] #four datapoints - one datapoints
                #================ TODO: assign alpha to change the weight ==============
                if self.order == 2: #Euclidean distance
#                     distance = np.linalg.norm(sub_matrix, axis = 1)
                    distance = self.Euclidean_distance(sub_matrix)
                elif self.order == 1: #Manhattan distance
                    distance = np.sum(np.abs(sub_matrix), axis=1)
                cluster_idx.append(np.argmin(distance))
            cluster_idx = np.array(cluster_idx)
            
            # calculate the coordinates of new_centers based on cluster_idx:
            # assign the coordinates of the new_center by calculating
            # mean/median of the coordinates of observations belonging to the same cluster.
            for idx in range(self.k):
                cluster_coordinates = X[cluster_idx == idx]
                if self.order == 2:
                    cluster_center = np.mean(cluster_coordinates, axis = 0)
                elif self.order == 1:
                    cluster_center = np.median(cluster_coordinates, axis = 0)                  
                new_centers[idx, :] = cluster_center

            # determine early stop and update centers and cluster_idx:
            # stop tranining as long as cluster index for all observations is the same as the previous iteration
            if (self.cluster_idx == cluster_idx).all():
                print(f"Early Stopped at Iteration {i}")
                return self
            self.centers = new_centers
            self.cluster_idx = cluster_idx
        return self


    # This function makes predictions with input data
    def predict(self, X):
        cluster_idx = []
        for i in range(X.shape[0]):
            sub_matrix = self.centers - X[i] #four datapoints - one datapoints
            if self.order == 2:
                distance = np.linalg.norm(sub_matrix, axis = 1)
            elif self.order == 1:
                distance = np.sum(np.abs(sub_matrix), axis=1)
            cluster_idx.append(np.argmin(distance))
        cluster_idx = np.array(cluster_idx)
        return cluster_idx
    
    def get_centers(self):
        centers = self.centers
        return centers
    
    # This function return the average distance from each data to its centroid
    def loss(self, X):
        distance_lis = []
        for i in range(X.shape[0]):
            sub_matrix = self.centers - X[i]
            if self.order == 2:
                distance = np.linalg.norm(sub_matrix, axis = 1)
            elif self.order == 1:
                distance = np.sum(np.abs(sub_matrix), axis=1)
            distance_lis.append(distance)
        distance_average = sum(distance_lis) / len(distance_lis)
        return distance_average
