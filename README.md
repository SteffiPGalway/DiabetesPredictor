# DiabetesPredictor

DiabetesPredictor is a simple machine learning package that trains a RandomForest model using the Diabetes dataset and enables users to make predictions. The package is built using `scikit-learn` and is intended as a demonstration of packaging Python machine learning projects for distribution.

## Features

- Train a RandomForestClassifier on the Diabetes dataset.
- Make predictions on new data after training.
- Save and load trained models.

## Installation

You can install the package via **PyPI** or from **source**.

### Install from PyPI

```bash
pip install diabetespredictor
```

### Install from Source (GitHub)

```bash
git clone https://github.com/SteffiPGalway/DiabetesPredictor.git
cd diabetespredictor
pip install .
```

## Usage

After installation, you can use `DiabetesPredictor` to train a model and make predictions.

### Example: Training and Making Predictions

```python
from DiabetesPredictor import DiabetesPredictor

# Initialize the predictor
predictor = DiabetesPredictor()

# Train the model on the Iris dataset
predictor.train()

# Make a prediction on a sample input
sample_input = [5.1, 3.5, 1.4, 0.2]
prediction = predictor.predict(sample_input)

print(f"Predicted class: {prediction}")
```