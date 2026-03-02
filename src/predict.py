import joblib
import numpy as np

# Load trained model
model = joblib.load("outputs/model.joblib")

# Example flower measurements:
# [sepal length, sepal width, petal length, petal width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample)

species = ["setosa", "versicolor", "virginica"]

print("Predicted species:", species[prediction[0]])