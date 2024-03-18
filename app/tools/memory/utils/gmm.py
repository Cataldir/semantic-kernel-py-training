import numpy as np
from sklearn.mixture import GaussianMixture


def train_gaussian_mixture_similarity(embeddings):
    n_components = np.arange(1, 21)
    bics = np.array([GaussianMixture(n).fit(embeddings).bic(embeddings) for n in n_components])
    optimal_n_components = n_components[np.argmin(bics)]
    gmm = GaussianMixture(n_components=optimal_n_components)
    gmm.fit(embeddings)
    clusters = gmm.predict(embeddings)
    return clusters


def apply_gaussian_mixture_similarity(embeddings):
    n_components = np.arange(1, 21)
    bics = np.array([GaussianMixture(n).fit(embeddings).bic(embeddings) for n in n_components])
    optimal_n_components = n_components[np.argmin(bics)]
    gmm = GaussianMixture(n_components=optimal_n_components)
    gmm.fit(embeddings)
    clusters = gmm.predict(embeddings)
    return clusters
