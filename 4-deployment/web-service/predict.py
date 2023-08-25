import pickle
from flask import Flask, request, jsonify
import spacy

nlp = spacy.load("en_core_web_lg")

with open("svm_model.bin", "rb") as f_in:
    model= pickle.load(f_in)

def preprocess(ride):
    vector = nlp(ride).vector
    return vector

def predict(features):
    preds = model.predict(features.reshape(1, -1))
    return preds

def num_to_word(number):
    if number == 0:
        return "Fake news"
    else:
        return "True news"

app = Flask("news-prediction")

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    vector = preprocess(ride)
    pred = predict(vector)
    pred = num_to_word(pred)

    result = {
        'The origin of news': pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




