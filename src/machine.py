import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


base = pd.read_csv('..data/spam.csv', encoding='latin-1')


base.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
base.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)


base['length'] = base['text'].str.split().str.len()

cont_ham_spam = base['label'].value_counts()

X = base['text']
y = base['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)

vetorizador = TfidfVectorizer()
X_train_vec = vetorizador.fit_transform(X_train)
X_test_vec = vetorizador.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test,y_pred) * 100

print("--- Accuracy ---")
print(f'{accuracy:.2f}%')

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))

# joblib.dump(model, '../artifacts/spam_model.pkl')
# joblib.dump(vetorizador, '../artifacts/vectorizer.pkl')

# print("\nModel and Vectorizer saved successfully!")