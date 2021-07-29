"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import random
import numpy as np
import math

class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        # self.classes = []
        self.xTrain = np.empty((0,2))
        self.centroidsList = []
        self.centroidErrors = []
        self.tracklets = []


    def get_vector(self, X):
        last = X[-1]
        twothird = X[math.floor(2*len(X)/3)]
        onethird = X[math.floor(len(X)/3)]
        first = X[0]
        last_center  = [(last[1] + last[3])/2, (last[2] + last[4])/2]
        onethird_center = [(onethird[1] + onethird[3])/2, (onethird[2] + onethird[4])/2]
        twothird_center = [(twothird[1] + twothird[3])/2, (twothird[2] + twothird[4])/2]
        first_center  = [(first[1] + first[3])/2, (first[2] + first[4])/2]
        finalresult = np.empty((0,2))

        result1 = np.array(last_center) - np.array(twothird_center)
        resultLength1 = np.sqrt(sum(np.square(result1)))
        if resultLength1!=0:
            finalresult = np.vstack((finalresult, 750/resultLength1 * result1))

        result2 = np.array(twothird_center) - np.array(onethird_center)
        resultLength2 = np.sqrt(sum(np.square(result2)))
        if resultLength2!=0:
            finalresult = np.vstack((finalresult, 750/resultLength2 * result2))
        
        result3 = np.array(onethird_center) - np.array(first_center)
        resultLength3 = np.sqrt(sum(np.square(result3)))
        if resultLength1!=0:
            finalresult = np.vstack((finalresult, 750/resultLength3 * result3))

        return finalresult


    def add_tracklet(self, tracklet):
        "Add a new tracklet into the database"
        self.tracklets.append(tracklet)
        if len(self.get_vector(tracklet['tracks'])) != 0:
            self.xTrain = np.vstack((self.xTrain, self.get_vector(tracklet['tracks'])))
        

    def build_clustering_model(self):
        "Perform clustering algorithm"
        
        self.centroidsList = self.k_means(self.xTrain, self.num_cluster)
        # self.get_outliers()

    def get_outliers(self):
        for ind, i in enumerate(self.centroidsList):
            mse = []
            for jind,j in enumerate(self.tracklets):
                if ind == self.find_initial_assignment(self.get_vector(j['tracks']), self.centroidsList):
                    mse.append(self.find_specificse(self.get_vector(j['tracks']), i))
            mse = np.sort(mse)[::-1]
            if len(mse)!=0:
                self.centroidErrors.append(mse[math.floor(len(mse)/20)])
            else:
                self.centroidErrors.append(0)
            
    

    def find_specificse(self, X, centroids):
        tse = 0
        for iind, i in enumerate(X):
            Y  = i - centroids
            tse += sum([j*j for j in Y])
        return tse 


    def find_assignment(self, X, centroids):
        se_centroids = np.array([0]*len(centroids))
        for jind,j in enumerate(centroids):
            for i in X:
                se_centroids[jind] += sum([t*t for t in (i-j)])
        classes = np.where(se_centroids == se_centroids.min())[0][0]
        # if se_centroids.min() > self.centroidErrors[classes] and self.centroidErrors[classes]!=0:
        #     classes = -1
        return classes

    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        ind = self.find_assignment(self.get_vector(tracklet['tracks']), self.centroidsList)
        # print(np.int64(ind+1).item())
        return np.int64(ind+1).item()

    def k_means(self, X,k):
        threshold = 0
        centroidIndexes = random.sample(range(0,len(X)), k)
        centroids = X[centroidIndexes]
        errors = []
        while 1 :
            classes = self.find_nearest_centroids(X, centroids)
            se = self.find_se(X, centroids, classes)
            centroids = self.find_new_centroids(X, centroids, classes)
            errors.append(se)
            if len(errors)>=2 and errors[-2]-errors[-1]<=threshold:
                break
        return centroids

    def find_se(self, X, centroids, classes):
        tse = 0
        for iind, i in enumerate(X):
            Y  = i - centroids[classes[iind]]
            tse += sum([j*j for j in Y])
        return tse 

    def find_initial_assignment(self, X, centroids):
        se_centroids = np.array([-1]*len(centroids))
        for jind,j in enumerate(centroids):
            for i in X:
                se_centroids[jind] += sum([t*t for t in (i-j)])
        classes = np.where(se_centroids == se_centroids.min())[0][0]
        return classes

    

    def find_nearest_centroids(self, X, centroids):
        classes = np.array([-1] * len(X))
        for iind, i in enumerate(X):
            se_centroids = np.array([-1]*len(centroids))
            for jind,j in enumerate(centroids):
                se_centroids[jind] = sum([t*t for t in (i-j)])

            classes[iind] = np.where(se_centroids == se_centroids.min())[0][0]

        return classes

    def find_new_centroids(self, X, oldcentroids, classes):
        centroids = []
        for i in range(0,len(oldcentroids)):
            new_centroid = np.array([0] * len(X[0])).astype('float64')
            cnt=0
            for jind, j in enumerate(classes):
                if i==j:
                    new_centroid+= X[jind]
                    cnt+=1
            # print(cnt)
            if cnt!=0:
                centroids.append(new_centroid/cnt)
            else:
                centroids.append(new_centroid/cnt)

        return np.array(centroids)
