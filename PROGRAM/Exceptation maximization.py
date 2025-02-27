from sklearn.mixture import GaussianMixture
import numpy as np

X_train = np.array([[1], [2], [3], [8], [9], [10]])
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)
print(gmm.means_)
