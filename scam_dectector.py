import pandas as pd

# Load the dataset
df = pd.read_csv("fake_job_postings.csv")

# Show first 5 rows
print(df.head())

# Show column names
print("\nColumns:")
print(df.columns.tolist())

# Check how many real vs fake
print("\nClass distribution:")
print(df['fraudulent'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Fill NaNs with empty strings
df.fillna('', inplace=True)

# Combine relevant text fields
df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements']

# Features (X) and Labels (y)
X = df['text']
y = df['fraudulent']

# Convert text to numeric features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
# Predict on custom input
sample_text = input("\nEnter a job post text to check if it's a scam: ")
sample_vector = vectorizer.transform([sample_text])
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("⚠️ This job post might be a SCAM!")
else:
    print("✅ This job post seems LEGIT.")

