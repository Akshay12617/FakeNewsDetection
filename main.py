import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Step 2: Add labels to the data
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Step 3: Combine the datasets
data = pd.concat([fake_df, true_df])
data = data[['text', 'label']]  # We only need the text and the label

# Step 4: Split data into training and testing sets
X = data['text']     # Features (news text)
y = data['label']    # Target (FAKE or REAL)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the model using PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
