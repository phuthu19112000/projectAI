import numpy as np
from matplotlib import pyplot

def multivariateGaussian(X, mu, Sigma2):
    """
    Tính toán hàm mật độ xác suất của phân phối gaussian đa biến. Trả về: 
    p : Một vectơ có shape (m,) chứa các xác suất được tính toán tại mỗi ví dụ đã cho.
        
    """
    k = mu.size

    # nếu sigma được cho dưới dạng đường chéo, hãy tính ma trận
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1))
    return p

def visualizeFit(X, mu, sigma2):
    """
    Trực quan hóa tập dữ liệu và phân phối ước tính của nó.
    Cho ta thấy hàm mật độ xác suất của phân phối Gaussian.
    Mỗi ví dụ có một vị trí (x1, x2) phụ thuộc vào các giá trị của nó.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    pyplot.plot(X[:, 0], X[:, 1], 'bx', mec='b', mew=2, ms=8)

    if np.all(abs(Z) != np.inf):
        pyplot.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder=100)

