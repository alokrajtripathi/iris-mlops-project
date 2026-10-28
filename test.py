import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

model = joblib.load('iris_model.pkl')

iris = load_iris()
X = iris.data
y = iris.target

pred = model.predict(X)
acc = accuracy_score(y, pred)

print("Predictions:", pred)
print("Actual:", y)
print("Accuracy:", acc)

assert len(pred) == len(y)
assert acc >= 0.80

print("All test cases passed and model is ready to deploy")
