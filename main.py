import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Veri setini bir dosyadan okuma
file_path = "dataset.txt"
df = pd.read_csv(file_path, header=None, sep="\s+")

# Sütun adlarını belirleme
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df.columns = columns

# Eğitim ve test veri setlerini oluşturma
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizasyon işlemi
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# PCA işlemi
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

# LDA işlemi
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_normalized, y_train)
X_test_lda = lda.transform(X_test_normalized)

# Çoklu Doğrusal Regresyon
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_coef = linear_reg.coef_

# Multinominal Lojistik Regresyon
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
logistic_reg_coef = logistic_reg.coef_

# Karar Ağacı Sınıflandırma
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)

# Naive Bayes Sınıflandırıcısı
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_naive_bayes = naive_bayes.predict(X_test)

# Performans metrikleri
confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

confusion_matrix_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)
accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)

# Sonuçları raporlama
print("Çoklu Doğrusal Regresyon Katsayıları:")
print(linear_reg_coef)

print("\nMultinominal Lojistik Regresyon Katsayıları:")
print(logistic_reg_coef)

print("\nKarar Ağacı Sınıflandırma Confusion Matrix'i:")
print(confusion_matrix_decision_tree)

print("\nKarar Ağacı Sınıflandırma Accuracy:")
print(accuracy_decision_tree)

print("\nNaive Bayes Sınıflandırma Confusion Matrix'i:")
print(confusion_matrix_naive_bayes)

print("\nNaive Bayes Sınıflandırma Accuracy:")
print(accuracy_naive_bayes)
