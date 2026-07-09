"""
@author: Lochana Marasinghe
@date: 7/9/2026
@description: 
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression


def get_vip(model):

    w = model.x_weights_
    t = model.x_scores_
    q = model.y_loadings_

    p, h = w.shape

    q_sq = q ** 2

    t_sq = np.sum(t ** 2, axis=0)

    ss_explained = t_sq * q_sq.flatten()
    total_ss = np.sum(ss_explained)


    w_norm_sq = np.sum(w ** 2, axis=0)  # shape (h,)
    vips = np.zeros((p,))

    for i in range(p):
        weight = (w[i, :] ** 2) / w_norm_sq
        vips[i] = np.sqrt(p * np.sum(ss_explained * weight) / total_ss)

    return vips

def plot_vips(vips, threshold):
        plt.figure(figsize=(12, 5))
        plt.plot(vips)
        plt.axhline(y=threshold, color='r', linestyle='--')
        plt.title("VIP Scores per Wavelength")
        plt.xlabel("Wavelength Index (0-432)")
        plt.ylabel("VIP Score")
        plt.show()

class VIPSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.2):
        self.selected_mask = None
        self.threshold = threshold
        self.pls = PLSRegression(n_components=10)
        self.vips = None


    def fit(self, X, y):
        self.pls.fit(X, y)
        self.vips = get_vip(self.pls)
        self.selected_mask = self.vips > self.threshold
        return self

    def transform(self, X):
        return X[:, self.selected_mask]


