from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt

def compute_distances_using_numpy(X, ClusterMeans, n, k):
    X_row_norms = np.linalg.norm(X) **2
    M_row_norms = np.linalg.norm(ClusterMeans) **2
    D = (np.outer(X_row_norms, np.ones(k)) + np.outer(np.ones(n), M_row_norms) - 2 * np.dot(X, ClusterMeans.T))
    return D

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = imread('mandrill.png', mode='RGB').astype(float)

N = int(mandrill.shape[0])

M = 2
k = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)


# TODO: Implement k-means and cluster the rows of X, then reconstruct the
# compressed image using the cluster center for each block, as specified in
# the homework description.
size = X.shape[1]
n = X.shape[0]
cluster_pool = np.zeros([k, size])
points_pool = []

#prepare the initial random clusters
for i in range(0, k):
    for j in range(0, size):
        rand = np.random.randint(0, 200)
        cluster_pool[i][j] = rand

#print(cluster_pool)


#initialize point pool
for i in range(0, k):
    temp = []
    points_pool.append(temp)

count = 0

while count < 30:
    #associate each point with a cluster_pool
    print("count: ", count)
    count = count + 1

    distance = compute_distances_using_numpy(X, cluster_pool, n, k)
    #print("distance matrix shape: ", distance.shape)
    for i in range(0, n):
        min_val = float('inf')
        min_index = 0
        for j in range(0, k):
            if distance[i][j] < min_val:
                min_val = distance[i][j]
                min_index = j
        points_pool[min_index].append(i)


    #update cluster using the mean of all the associated points
    for i in range(0, k):
        total = np.zeros([1, size])
        totalnumber = len(points_pool[i])
        if totalnumber != 0:
            for j in range(0, totalnumber):
                index = points_pool[i][j]
                total = total + X[index, :]
            total = total / totalnumber
            cluster_pool[i, :] = total

for i in range(0, k):
    total = np.zeros([1, size])
    for j in range(0, len(points_pool[i])):
        index = points_pool[i][j]
        X[index, :] = cluster_pool[i, :]

for i in range(N//M):
    for j in range(N//M):
        mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:] = X[i*N//M+j,:].reshape(2,2,3)

print(X)

# To show a color image using matplotlib, you have to restrict the color
# color intensity values to between 0.0 and 1.0. For example,
plt.imshow(mandrill/255)
plt.show()
