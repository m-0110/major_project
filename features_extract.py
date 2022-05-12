import numpy as np
from numpy.linalg import eig  # import eig method to extract eigen values


def mul(arr1, arr2):
    arr3 = np.zeros((arr1.shape[0], arr2.shape[1]))

    if (arr1.shape[1] == arr2.shape[0]):

        for i in range(arr1.shape[0]):
            for k in range(arr2.shape[1]):
                for j in range(arr2.shape[0]):
                    arr3[i][k] += arr1[i][j] * arr2[j][k]

        return arr3


def pca(encoded_img):
    n = len(encoded_img[:, 0])
    m = len(encoded_img[0])

    # print(n,m)#rows,columns

    X = []
    A = []
    for i in range(0, m):
        X.append(encoded_img[:, i])
        mx = np.mean(encoded_img[:, i])
        A.append(encoded_img[:, i] - mx)

    X = np.array(X)
    # print("features row wise")
    # print(X)#now each column (feature) is a row

    A = np.array(A)
    # print(A)

    c = mul(A, (A.T))
    # print(c)

    covariance_matrix = c / (n - 1)
    # print(cov_matrix)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    return eigenvalues


