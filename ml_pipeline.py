import pickle
import sys
import numpy as np 
from sklearn.linear_model import LinearRegression

print("=== ML CI Pipeline ===")
X = np.array([[1],[2],[3]])
y = np.array([2,4,6])
model = LinearRegression()
model.fit(X,y)
print("Model trained")

with open('model.pkl','wb') as f:
    pickle.dump(model,f)
print("Model saved")
prediction = model.predict([[4]])[0]
print(f"Prediction : {prediction:.1f} (Expected:8.0)")
if abs(prediction-8.0)<0.1:
    print("Validation Passed")
    sys.exit(0)
else:
    print("Validation Failed")
    sys.exit(1)