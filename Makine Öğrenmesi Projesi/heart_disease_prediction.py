# ===============================
# 📌 Kalp Hastalığı Tahmini Projesi - Aşama 1
# ===============================

# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Makine öğrenmesi için
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Kullanılacak algoritmalar
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ===============================
# 📂 1. Veri Setinin Yüklenmesi
# ===============================

# CSV dosyasını oku
df = pd.read_csv("heart.csv")  # Dosya ismini doğru yazdığından emin ol
print("\n📌 İlk 5 Satır:")
print(df.head())


# ===============================
# 🧠 2. Özellik Mühendisliği
# ===============================
# Yeni kombinasyon öznitelikler
df['age_chol'] = df['age'] * df['chol']
df['ecg_thalach'] = df['restecg'] * df['thalach']
df['slope_ca'] = df['slope'] * df['ca']

# Yeni özellikleri görmek için (opsiyonel)
print("\n🧬 Yeni Özellikler Eklendi:")
print(df[['age_chol', 'ecg_thalach', 'slope_ca']].head())

# ===============================
# 🔍 3. Temel Bilgilerin İncelenmesi
# ===============================

print("\n🔎 Veri Seti Bilgileri:")
print(df.info())

print("\n📊 Temel İstatistikler:")
print(df.describe())
print("\n📋 Veri setindeki tüm sütunlar:")
print(df.columns.tolist())


print("\n❓ Eksik Veri Kontrolü:")
print(df.isnull().sum())

# ===============================
# ❤️ 4. Hedef Değişken Dağılımı
# ===============================

sns.countplot(x='target', data=df)
plt.title("Kalp Hastalığı Dağılımı (0 = Yok, 1 = Var)")
plt.xlabel("Kalp Hastalığı")
plt.ylabel("Kişi Sayısı")
plt.show()

print("\n📊 Hedef Değişken (target) Dağılımı:")
print(df["target"].value_counts())

# Hedef değişkenin grafiği
sns.countplot(x='target', data=df, palette='Set2')
plt.title("Kalp Hastalığı Dağılımı (0 = Yok, 1 = Var)")
plt.xlabel("Kalp Hastalığı")
plt.ylabel("Kişi Sayısı")
plt.tight_layout()
plt.savefig("results/target_dagilimi.png")
plt.show()


# ===============================
# 🧼 5. Aykırı Değer Analizi
# ===============================
# Kutu grafikleriyle aykırı değerleri görselleştirme
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('target')  # Hedef sütunu çıkart

for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"{col} - Aykırı Değer Analizi")
    plt.show()

# ===============================
# 📊 6. Korelasyon Matrisi
# ===============================
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Öznitelikler Arası Korelasyon Matrisi")
plt.show()

# ===============================
# 🔁 7. Encoding (Gerekirse)
# ===============================
# Bu veri setinde tüm sütunlar sayısal olduğu için encoding gerekmez.
# Ama ileride başka veri setleriyle çalışırsan şu şekilde yapılabilir:
# df = pd.get_dummies(df, columns=['categorical_column'])

# ===============================
# 📏 8. Veri Ölçeklendirme
# ===============================
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 🎯 9. Eğitim ve Test Verisi Ayırma
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("✅ Eğitim ve test verisi başarıyla hazırlandı!")


# ===============================
# 📊 10. Histogramlar ve Dağılım Grafikleri
# ===============================

#for col in numerical_cols:
    #plt.figure(figsize=(6, 3))
    #sns.histplot(df[col], kde=True, bins=30)
    #plt.title(f"{col} - Histogram")
    #plt.show()

# Klasör oluştur (grafikler ayrı klasörde saklansın)
os.makedirs("results/grafikler", exist_ok=True)


# Tüm sayısal sütunları + yeni eklenen sütunları listele
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('target')  # Hedef değişken çıkarılsın

for col in numerical_cols:
    # Boxplot (Aykırı Değer)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"{col.upper()} - Aykırı Değer Analizi", fontsize=12, fontweight='bold')
    plt.xlabel(col, fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/grafikler/{col}_boxplot.png")
    #plt.show()
    plt.close()

    # Histogram (Aykırı Değer)
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=30, kde=True, color='lightcoral')
    plt.title(f"{col.upper()} - Histogram", fontsize=12, fontweight='bold')
    plt.xlabel(col, fontsize=11)
    plt.ylabel("Frekans", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/grafikler/{col}_histogram.png")
    #plt.show()
    plt.close()

# ===============================
# 🔎 11. Korelasyonu Yüksek Özelliklerin Tespiti ve Kaldırılması
# ===============================
# Korelasyon eşiği belirleyelim (örneğin 0.9)
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Yüksek korelasyonlu sütunları tespit et
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
print(f"\n🗑️ Korelasyonu yüksek olduğu için çıkarılacak sütunlar: {to_drop}")

# 1. Korelasyonu yüksek sütunlar varsa açıklama
if len(to_drop) == 0:
    print("ℹ️ 0.9 üzerinde korelasyona sahip hiçbir sütun bulunamadı. Bu nedenle çıkarılacak sütun yok.")
else:
    print(f"Çıkarılan sütunlar: {to_drop}")


# Veri setinden çıkar
df.drop(columns=to_drop, inplace=True)

# ===============================
# 🎯 12. Güncel X ve y
# ===============================
X = df.drop("target", axis=1)
y = df["target"]

# Ölçeklendirme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/test bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ===============================
# 🤖 13. Modelleme ve Performans Karşılaştırması
# ===============================


# Kullanılacak modellerin listesi
models = {
    #"Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
     "Gradient Boosting": GradientBoostingClassifier()
    #"MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True, random_state=42)
}

# Her model için eğitim, tahmin ve değerlendirme
for name, model in models.items():
    print(f"\n📌 Model: {name}")
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Test verisiyle tahmin yap
    y_pred = model.predict(X_test)
    
    # Accuracy (doğruluk) oranı
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    
    # Detaylı sınıflandırma metrikleri
    print("🧾 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix görselleştirme
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()

 # ===============================
# 🔧 14. Random Forest Optimizasyonu
# ===============================

from sklearn.model_selection import GridSearchCV

# Hiperparametre aralıkları
param_grid_rf = {
    'n_estimators': [50, 100, 200],          # Ağaç sayısı
    'max_depth': [None, 5, 10, 20],          # Ağaç derinliği
    'min_samples_split': [2, 5, 10],         # Dal bölme minimum veri sayısı
    'min_samples_leaf': [1, 2, 4]            # Yaprakta minimum örnek
}

# GridSearchCV objesi oluştur
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid_rf,
    cv=5,                        # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1,                   # Tüm çekirdekleri kullan
    verbose=1
)

# Modeli eğit
grid_rf.fit(X_train, y_train)

# En iyi parametreler ve skor
print("\n🌟 En iyi parametreler (Random Forest):")
print(grid_rf.best_params_)

print(f"🏆 En iyi doğruluk (CV): {grid_rf.best_score_:.4f}")

# Test verisiyle final tahmin
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Performans sonuçları
print("\n✅ Test Seti Sonuçları (Optimize Edilmiş Random Forest):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Optimize Edilmiş Random Forest - Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.savefig("results/random_forest_optimizasyon.png")
plt.show()

# ===============================
# 🔧15. Support Vector Machine (SVM) Optimizasyonu
# ===============================
from sklearn.pipeline import Pipeline

# SVM için parametre aralıkları
param_grid_svm = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma': ['scale', 'auto']
}

# Pipeline oluşturuyoruz: scaler + SVM birlikte optimize edilecek
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),  # Ekstra garanti için tekrar ölçeklendirme
    ('svc', SVC())
])

# GridSearchCV uygulama
grid_svm = GridSearchCV(
    estimator=pipe_svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Eğit
grid_svm.fit(X_train, y_train)

# En iyi sonuçlar
print("\n🌟 En iyi parametreler (SVM):")
print(grid_svm.best_params_)

print(f"🏆 En iyi doğruluk (CV): {grid_svm.best_score_:.4f}")

# Test setinde değerlendirme
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("\n✅ Test Seti Sonuçları (Optimize Edilmiş SVM):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Optimize Edilmiş SVM - Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.savefig("results/svm_optimizasyon.png")
plt.show()

# ===============================
# 🔧 16. K-Nearest Neighbors (KNN) Optimizasyonu
# ===============================
# Kütüphane zaten eklenmiş olmalı: from sklearn.neighbors import KNeighborsClassifier

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_knn,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_knn.fit(X_train, y_train)

# En iyi sonuçlar
print("\n🌟 En iyi parametreler (KNN):")
print(grid_knn.best_params_)

print(f"🏆 En iyi doğruluk (CV): {grid_knn.best_score_:.4f}")

# Test seti değerlendirmesi
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

print("\n✅ Test Seti Sonuçları (Optimize Edilmiş KNN):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Optimize Edilmiş KNN - Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.savefig("results/knn_optimizasyon.png")
plt.show()

# ===============================
# 🔧 17. Gradient Boosting Optimizasyonu
# ===============================

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=param_grid_gb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_gb.fit(X_train, y_train)

print("\n🌟 En iyi parametreler (Gradient Boosting):")
print(grid_gb.best_params_)

print(f"🏆 En iyi doğruluk (CV): {grid_gb.best_score_:.4f}")

# Test değerlendirmesi
best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)

print("\n✅ Test Seti Sonuçları (Optimize Edilmiş Gradient Boosting):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(classification_report(y_test, y_pred_gb))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn")
plt.title("Optimize Edilmiş Gradient Boosting - Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()

# ===============================
# 📈 18. Özellik Önem Grafiği (Random Forest ile)
# ===============================

# Model tekrar eğitiliyor çünkü tüm X'i kullanmak istiyoruz
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)

feature_importances = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Özellik': features,
    'Önem': feature_importances
}).sort_values(by='Önem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Önem', y='Özellik', data=importance_df, color="skyblue")
plt.title(" Özellik Önem Grafiği (Random Forest)")
plt.xlabel("Önem Derecesi")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.savefig("results/ozellik_onem_grafigi.png")
plt.show()








