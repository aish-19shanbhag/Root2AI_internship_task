# Importing Python Libraries for Data manipulation and Data Visualisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("root2ai - Data.csv")

# Inspecting first few rows.
dataset.head(12)

# Inspecting shape of the data.
dataset.shape

# Inspecting descriptive statistics.
dataset.describe()

# Splitting data into input and output features.
y = dataset["Target"]
X_text = dataset["Text"]

# Converting text into numeric features using TF-IDF.
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(X_text)

# Splitting data into test and training sets.
test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Selecting right algorithm.
model = RandomForestClassifier()

# Fit model into the data.
model.fit(X_train, y_train)

# Check the model performance on the training data.
predictions = model.predict(X_train)
print(accuracy_score(y_train, predictions))

# Evaluating the model on the test data.
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
