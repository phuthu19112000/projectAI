import numpy as np

def featureNormalize(X):
    """
    Chuẩn hóa các tính năng trong X trả về phiên bản chuẩn hóa của X trong đó giá trị trung bình của mỗi tính năng là 0 
    và độ lệch chuẩn là 1. Đây thường là bước tiền xử lý tốt cần thực hiện khi làm việc với các thuật toán học máy.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    
    return X_norm, mu, sigma

def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
    
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

    return centroids, idx