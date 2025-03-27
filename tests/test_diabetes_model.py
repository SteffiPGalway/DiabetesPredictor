import pytest
from DiabetesPredictor import DiabetesPredictor

def test_train_and_predict():
    model = DiabetesPredictor()
    model.train()
    result = model.predict([50, 1, 22, 140, 80, 0.5, 0.5, 1, 1, 1]),
    assert len(result) == 1

if __name__ == "__main__":
    pytest.main()
