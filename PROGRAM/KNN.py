from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 0, 1, 1, 0])
knn = KNN(k=3)
knn.fit(X_train, y_train)
X_test = np.array([[1, 2], [6, 7]])
print(knn.predict(X_test))
