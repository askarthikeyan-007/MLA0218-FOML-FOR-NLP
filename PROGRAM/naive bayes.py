import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class NaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.priors = {cls: np.mean(y_train == cls) for cls in self.classes}
        self.likelihoods = {}
        
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.likelihoods[cls] = X_cls.mean(axis=0), X_cls.var(axis=0)
    
    def predict(self, X_test):
        posteriors = []
        for x in X_test:
            class_probs = {}
            for cls in self.classes:
                mean, var = self.likelihoods[cls]
                prior = self.priors[cls]
                likelihood = np.prod(np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var))
                class_probs[cls] = prior * likelihood
            posteriors.append(max(class_probs, key=class_probs.get))
        return np.array(posteriors)

# Example usage
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [1, 2, 3, 4, 5],
    'Class': ['A', 'A', 'B', 'B', 'A']
})
X = data[['Feature1', 'Feature2']].values
y = data['Class'].values

nb = NaiveBayes()
nb.fit(X, y)
predictions = nb.predict(X)

# Confusion matrix and accuracy
print(confusion_matrix(y, predictions))
print("Accuracy:", accuracy_score(y, predictions))
