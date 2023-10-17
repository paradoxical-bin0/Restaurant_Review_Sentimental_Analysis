import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    user_review = [x for x in request.form.values()]
    # final_features = [np.array(user_review)]
    prediction = model.predict(user_review)

    output = prediction[0]

    return render_template('index.html', prediction_text='Sentiment = {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
