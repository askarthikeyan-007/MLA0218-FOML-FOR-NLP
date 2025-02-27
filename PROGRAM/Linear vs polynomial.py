import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([1, 2, 3, 4, 5])
model = LinearRegression()
model.fit(X_train, y_train)
linear_predictions = model.predict(X_train)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
poly_predictions = poly_model.predict(X_poly)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear_predictions, color='blue', label="Linear Regression")
plt.plot(X_train, poly_predictions, color='green', label="Polynomial Regression")
plt.legend()
plt.show()
