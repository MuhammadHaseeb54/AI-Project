from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the 22 features used in training
all_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture'
]

app = Flask(__name__)

@app.route('/')
def home():
    # Render the form with all 22 features
    return render_template('index.html', all_features=all_features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Initialize an array for all 22 features
            input_data = np.zeros(22)  # Assuming 22 features in total
            
            # Dynamically retrieve all 22 features from the form
            for i, feature in enumerate(all_features):
                feature_value = float(request.form[feature])  # Get the feature value from the form
                input_data[i] = feature_value

            # Scale the input data using the same scaler used in training
            scaled_input = scaler.transform([input_data])  # Ensure input is 2D
            
            # Make prediction using the trained model
            prediction = model.predict(scaled_input)
            prediction_prob = model.predict_proba(scaled_input)[0][prediction[0]]

            # Interpret prediction
            diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

            # Return the result to the user with prediction and probability
            return render_template('index.html', 
                                   diagnosis=diagnosis,
                                   probability=f'{prediction_prob * 100:.2f}%',
                                   all_features=all_features)
        except ValueError:
            # Handle cases where input values are invalid
            return render_template('index.html', 
                                   diagnosis="Error: Please input numeric values for all features.",
                                   all_features=all_features)
        except Exception as e:
            # Handle other potential errors
            return render_template('index.html', 
                                   diagnosis=f"An unexpected error occurred: {e}",
                                   all_features=all_features)


if __name__ == "__main__":
    app.run(debug=True)
