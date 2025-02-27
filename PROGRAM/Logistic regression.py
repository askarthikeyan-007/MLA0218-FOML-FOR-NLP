from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
y_train = [0, 0, 1, 1]
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_train)
print("Accuracy:", accuracy_score(y_train, predictions))
