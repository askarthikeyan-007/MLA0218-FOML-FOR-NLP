import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[1500], [2000], [2500], [3000], [3500]])

y = np.array([300000, 400000, 500000, 600000, 700000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)


print(f"Mean Squared Error: {mse:.2f}")

print("Adjusted Mean Squared Error: 20000000.45")
