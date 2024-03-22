import numpy as np
# from past.builtins import xrange


class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, Y):
        self.X_train = X
        self.y_train = Y
    
    def check(self):
        return self.X_train.shape[0]
    
    def predict(self, X, k=1, num_loops = 0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)
    
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print(dists.shape)
        
        for i in range (num_test):
            for j in range (num_train):
                # dis x_tesdt[]
                dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
        return dists
    
    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.lipredictnalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dists = np.sqrt(np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, self.X_train.T) + np.sum(self.X_train**2, axis=1))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
    
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = []
            nearest_indices = np.argsort(dists[i])[:k]

            # Collect labels of k nearest neighbors
            for idx in nearest_indices:
                closest_y.append(self.y_train[idx])

            # Count occurrences of each label
            label_counts = np.bincount(closest_y)

            # Find the label with the maximum count
            y_pred[i] = np.argmax(label_counts)

        return y_pred


            
