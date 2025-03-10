from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample data for training (features: X, target: y)
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([1.5, 3.1, 4.9, 8.2, 10.1])  # Target values

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_value = request.form['input_value']
        input_data = np.array([[float(input_value)]])

        # Load the model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
