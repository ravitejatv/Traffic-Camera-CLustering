import numpy as np
from scipy.spatial import distance
import pandas as pd
import random
import matplotlib.pyplot as plt

def find_se(X, centroids, classes):
    tse = 0
    for iind, i in enumerate(X):
        Y  = i - centroids[classes[iind]]
        tse += sum([j*j for j in Y])
    return tse 
        

def find_nearest_centroids(X, centroids):
    classes = np.array([-1] * len(X))
    for iind, i in enumerate(X):
        se_centroids = np.array([-1]*len(centroids))
        for jind,j in enumerate(centroids):
            se_centroids[jind] = sum([t*t for t in (i-j)])

        classes[iind] = np.where(se_centroids == se_centroids.min())[0][0]

    return classes

def find_new_centroids(X, oldcentroids, classes):
    centroids = []
    for i in range(0,len(oldcentroids)):
        new_centroid = np.array([0] * len(X[0])).astype('float64')
        cnt=0
        for jind, j in enumerate(classes):
            if i==j:
                new_centroid+= X[jind]
                cnt+=1
        
        centroids.append(new_centroid/cnt)

    return np.array(centroids)
        
def k_means(X,k):
    threshold = 0
    centroidIndexes = random.sample(range(0,len(X)), k)
    centroids = X[centroidIndexes]
    cnt = 5
    errors = []
    while 1 :
        classes = find_nearest_centroids(X, centroids)
        se = find_se(X, centroids, classes)
        centroids = find_new_centroids(X, centroids, classes)
        errors.append(se)
        if len(errors)>=2 and errors[-2]-errors[-1]<=threshold:
            break
    return centroids


def assignment(X, centroids):
    ind = find_nearest_centroids(X, centroids)
    return ind


def assignment_plot(X, centroids, yTest):
    ind = find_nearest_centroids(X, centroids)
    counts = np.unique(ind,return_counts=True)[1]
    ycounts = np.unique(yTest,return_counts=True)[1]

    outs = []
    for i in range(0,len(centroids)):
        out = []
        for jind,j in enumerate(ind):
            if i==j:
                if yTest[jind] not in out:
                    out.append(yTest[jind])
        
        outs.append(out)
    print(outs)
    print(ycounts)
    fig = plt.figure()
    for centrind, centroid in enumerate(centroids):
        plt.subplot(2, 5, centrind + 1)
        plt.imshow(np.reshape(centroid, (28,28)))
        plt.title(counts[centrind])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return ind

def load_csv():
    test = pd.read_csv('mnist_test_hw5.csv').to_numpy()
    train = pd.read_csv('mnist_train_hw5.csv').to_numpy()
    
    yTrain = train[:, [0]].flatten()
    xTrain = train[:,1:]/255

    yTest = test[:, [0]].flatten()
    xTest = test[:,1:]/255

    return [xTrain, yTrain, xTest, yTest]


def k_means_plot(X,k):
    threshold = 0
    centroidIndexes = random.sample(range(0,len(X)), k)
    centroids = X[centroidIndexes]
    cnt = 5
    errors = []
    cnt = 1
    while 1 :
        print("Iteration : " + str(cnt))
        cnt+=1
        classes = find_nearest_centroids(X, centroids)
        se = find_se(X, centroids, classes)
        centroids = find_new_centroids(X, centroids, classes)
        errors.append(se)
        if len(errors)>=2 and errors[-2]-errors[-1]<=threshold:
            break
    
    plt.plot(np.arange(np.size(errors)), errors, marker='.')
    plt.xlabel('Iterations')
    plt.ylabel('Squared Errors')
    plt.show()

    print("The final value of sum of squared errors : " + str(errors[-1]))
    return centroids

xTrain, yTrain, xTest, yTest = load_csv()
# centroids = k_means(xTrain,10)
centroids = k_means_plot(xTrain,8)
# ind = assignment(xTest, centroids)
assignment_plot(xTest, centroids, yTest)

