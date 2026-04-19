# makine öğrenmesi ödevi - covid19 hasta tahmini
# veri seti: https://www.kaggle.com/datasets/meirnizri/covid19-dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# 1. veriyi yükle
# ----------------------------
df = pd.read_csv("Covid_Data.csv")

print("veri seti boyutu:", df.shape)
print()
print(df.head())
print()
print(df.info())
print()
print("eksik deger var mi:")
print(df.isnull().sum())

# ----------------------------
# 2. hedef degiskeni olustur
# ----------------------------
# DATE_DIED sutununda "9999-99-99" yaziyorsa hasta hayatta (0), tarih varsa olmus (1)
# biz bunu covid hastasi olup olmadigi olarak kullanalim
# CLASIFFICATION_FINAL: 1,2,3 covid pozitif, 4-7 negatif/belirsiz

df["hasta"] = df["CLASIFFICATION_FINAL"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

print()
print("hasta dagilimi:")
print(df["hasta"].value_counts())

# ----------------------------
# 3. veri temizleme
# ----------------------------
# 97 ve 99 degerler "bilinmiyor" anlamina geliyor, bunlari nan yapıyoruz
df = df.replace(97, np.nan)
df = df.replace(99, np.nan)

# kullanacagimiz ozellikleri secelim
ozellikler = ["SEX", "AGE", "DIABETES", "HIPERTENSION", "OBESITY",
              "PNEUMONIA", "TOBACCO", "RENAL_CHRONIC", "CARDIOVASCULAR", "ASTHMA"]

df = df[ozellikler + ["hasta"]]

# eksik degerleri sutunun medyanıyla doldur
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print()
print("temizleme sonrasi eksik deger:")
print(df.isnull().sum())

# ----------------------------
# 4. EDA - keşifsel veri analizi
# ----------------------------
plt.figure(figsize=(6, 4))
df["hasta"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
plt.title("hasta dagilimi (1=pozitif, 0=negatif)")
plt.xlabel("hasta")
plt.ylabel("kisi sayisi")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("hasta_dagilimi.png")
plt.close()
print("hasta_dagilimi.png kaydedildi")

plt.figure(figsize=(6, 4))
sns.histplot(df["AGE"], bins=20, color="steelblue")
plt.title("yas dagilimi")
plt.xlabel("yas")
plt.tight_layout()
plt.savefig("yas_dagilimi.png")
plt.close()
print("yas_dagilimi.png kaydedildi")

# ----------------------------
# 5. egitim ve test verisi ayir
# ----------------------------
X = df[ozellikler]
y = df["hasta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print()
print("egitim seti:", X_train.shape)
print("test seti  :", X_test.shape)

# ----------------------------
# 6. model 1 - logistic regression
# ----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_tahmin = lr_model.predict(X_test)

lr_acc  = accuracy_score(y_test, lr_tahmin)
lr_prec = precision_score(y_test, lr_tahmin)
lr_rec  = recall_score(y_test, lr_tahmin)

print()
print("=== logistic regression ===")
print("accuracy :", round(lr_acc, 4))
print("precision:", round(lr_prec, 4))
print("recall   :", round(lr_rec, 4))

# confusion matrix
cm_lr = confusion_matrix(y_test, lr_tahmin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp.plot()
plt.title("logistic regression - confusion matrix")
plt.tight_layout()
plt.savefig("cm_logistic.png")
plt.close()
print("cm_logistic.png kaydedildi")

# ----------------------------
# 7. model 2 - decision tree
# ----------------------------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_tahmin = dt_model.predict(X_test)

dt_acc  = accuracy_score(y_test, dt_tahmin)
dt_prec = precision_score(y_test, dt_tahmin)
dt_rec  = recall_score(y_test, dt_tahmin)

print()
print("=== decision tree ===")
print("accuracy :", round(dt_acc, 4))
print("precision:", round(dt_prec, 4))
print("recall   :", round(dt_rec, 4))

cm_dt = confusion_matrix(y_test, dt_tahmin)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp2.plot()
plt.title("decision tree - confusion matrix")
plt.tight_layout()
plt.savefig("cm_decision_tree.png")
plt.close()
print("cm_decision_tree.png kaydedildi")

# ----------------------------
# 8. model karsilastirma
# ----------------------------
print()
print("=== model karsilastirma ===")
print(f"{'model':<25} {'accuracy':<12} {'precision':<12} {'recall'}")
print("-" * 60)
print(f"{'logistic regression':<25} {lr_acc:<12.4f} {lr_prec:<12.4f} {lr_rec:.4f}")
print(f"{'decision tree':<25} {dt_acc:<12.4f} {dt_prec:<12.4f} {dt_rec:.4f}")
