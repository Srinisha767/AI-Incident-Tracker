import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample data (replace with your actual data loading)
data = {
    'description': [
        'Network connection is down',
        'Cannot access the email server',
        'Website is showing a 404 error',
        'Software installation failed',
        'User forgot password',
        'Request for new hardware',
        'System is running slow',
        'Printer is not working',
        'Database query is failing',
        'Need access to a shared folder'
    ],
    'category': [
        'Network',
        'Network',
        'Application',
        'Software',
        'User Account',
        'Hardware',
        'Performance',
        'Hardware',
        'Database',
        'User Account'
    ]
}
df = pd.DataFrame(data)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['description'])
y = df['category']

# Model training
model = LogisticRegression()
model.fit(X, y)

# Create model directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(__file__), 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model and vectorizer
joblib.dump(model, os.path.join(model_dir, 'incident_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))

print("Model trained and saved successfully.")
