import pickle
import pandas as pd


# Load data
def load_data():
    data = [
        [5, 9.5, 5, 6]
    ]
    new_data = pd.DataFrame(data, columns=[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ])
    return new_data


# Load model
def load_model():
    with open('trained_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Make predictions
def make_predictions(data, model):
    return model.predict(data)


# Write results
def write_results(predictions):
    print(predictions)


# Orchestrate
def run():
    new_data = load_data()
    model = load_model()
    predictions = make_predictions(new_data, model)
    write_results(predictions)


if __name__ == "__main__":
    run()
