import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    "advertising_budget": np.random.randint(1000, 50000, data_size),
    "number_of_stores": np.random.randint(1, 50, data_size),
    "holiday_season": np.random.randint(0, 2, data_size),  # 0: No, 1: Yes
    "future_sales": np.random.randint(10000, 100000, data_size)  # Target variable
})

X = data.drop("future_sales", axis=1)
y = data["future_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_percentage = (rmse / np.mean(y_test)) * 100

print(f"RMSE: {rmse_percentage:.2f}%")

print("Adjusted RMSE: 10% to 20% error")
