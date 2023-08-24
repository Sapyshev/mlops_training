import pickle

with open("././models/svm_model.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)

def predict(features):
    pred = model.predict(features)