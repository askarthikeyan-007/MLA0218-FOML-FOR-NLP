from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X_train = [[1], [2], [3], [4]]
y_train = [1, 2, 3, 4]
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_train)
print("Mean Squared Error:", mean_squared_error(y_train, predictions))
