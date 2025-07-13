import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
import joblib

app = Flask(__name__)
client = MongoClient("mongodb+srv://srinisha0124:Srinisha@cluster0.zanvg7y.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["incidentDB"]
collection = db["incidents"]

# Load AI Model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'incident_classifier.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'model', 'tfidf_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/report', methods=['POST'])
def report_incident():
    data = request.json
    desc = data["description"]
    
    X = vectorizer.transform([desc])
    prediction = model.predict(X)[0]

    incident = {
        "description": desc,
        "category": prediction,
        "status": "Open"
    }
    collection.insert_one(incident)
    return jsonify({"message": "Incident reported", "category": prediction})

@app.route('/incidents', methods=['GET'])
def get_all():
    data = list(collection.find({}, {"_id": 0}))
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
