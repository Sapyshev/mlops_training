import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
import spacy

nlp = spacy.load("en_core_web_lg")

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
RUN_ID = "72751afe6b954f198b78a635369a8dc4"
logged_model = f's3://mlflow-artifacts-remote-rollan/1/{RUN_ID}/artifacts/model'
#logged_model = 'runs:/72751afe6b954f198b78a635369a8dc4/model'
model = mlflow.pyfunc.load_model(logged_model)


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
        'The origin of news': pred,
        'model_version': RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




