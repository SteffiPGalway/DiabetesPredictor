from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


class DiabetesPredictor:
    def __init__(self):
        self.model = None

    def train(self):
        diabetes = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, data):
        if not self.model:
            raise Exception("Model is not trained yet!")
        return self.model.predict([data])

    def save_model(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)