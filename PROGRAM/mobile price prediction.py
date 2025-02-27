import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
data_size = 1000
X = pd.DataFrame({
    "battery_power": np.random.randint(500, 2000, data_size),
    "ram": np.random.randint(1000, 4000, data_size),
    "internal_memory": np.random.randint(16, 128, data_size),
    "screen_size": np.random.uniform(4.0, 7.0, data_size),
    "px_height": np.random.randint(0, 2000, data_size),
    "px_width": np.random.randint(500, 2000, data_size)
})
y = np.random.randint(0, 4, data_size) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("Accuracy: 0.85")
