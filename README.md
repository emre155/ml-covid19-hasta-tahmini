# makine öğrenmesi ödevi - covid19 hasta tahmini

bu projede covid-19 veri seti kullanarak bir kişinin covid pozitif mi negatif mi olduğunu tahmin etmeye çalıştım. logistic regression ve decision tree algoritmalarını kullandım.

---

## veri seti

kaggle üzerinden aldım: https://www.kaggle.com/datasets/meirnizri/covid19-dataset

meksika sağlık bakanlığına ait gerçek bir veri seti. yaklaşık 1 milyon satır var. her satır bir hastayı temsil ediyor. yaş, cinsiyet, diyabet, hipertansiyon gibi özellikler içeriyor.

sütunlar:
- SEX: cinsiyet
- AGE: yaş
- DIABETES: diyabet var mı
- HIPERTENSION: hipertansiyon var mı
- OBESITY: obezite var mı
- PNEUMONIA: zatürre var mı
- TOBACCO: sigara kullanımı
- RENAL_CHRONIC: kronik böbrek hastalığı
- CARDIOVASCULAR: kalp hastalığı
- ASTHMA: astım
- CLASIFFICATION_FINAL: covid sonucu (1-3 pozitif, 4-7 negatif)

---

## dosyalar

- `covid_tahmin.py` → tüm kod burda, çalıştırınca her şey otomatik yapılıyor
- `Covid_Data.csv` → veri seti (kaggle'dan indirilmeli)
- `hasta_dagilimi.png` → grafik
- `yas_dagilimi.png` → grafik
- `cm_logistic.png` → logistic regression confusion matrix
- `cm_decision_tree.png` → decision tree confusion matrix

---

## nasıl çalıştırılır

önce gereken kütüphaneleri yükle:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

sonra veri setini `Covid_Data.csv` adıyla kodla aynı klasöre koy ve çalıştır:

```
python covid_tahmin.py
```

---

## veri ön işleme

veri setinde 97 ve 99 değerleri "bilinmiyor" anlamına geliyordu, bunları önce NaN yaptım sonra o sütunun medyanıyla doldurdum. hedef değişken olarak `CLASIFFICATION_FINAL` sütununu kullandım, 1-2-3 pozitif, diğerleri negatif. modelde kullanacağım 10 özellik seçtim ve veriyi %80 eğitim %20 test olarak ayırdım.

---

## kullanılan algoritmalar

### logistic regression
ikili sınıflandırma problemleri için çok kullanılan bir algoritma. hasta mı değil mi sorusunu cevaplamak için uygun. veriyi doğrusal bir sınıra göre ayırmaya çalışıyor.

### decision tree
bir karar ağacı oluşturarak sınıflandırma yapıyor. "yaşı 60'tan büyük mü? diyabeti var mı?" gibi sorular sorarak dala dala ilerliyor. anlaması kolay ve yorumlanabilir bir algoritma.

---

## model performansı

| model | accuracy | precision | recall |
|---|---|---|---|
| logistic regression | 0.6589 | 0.6236 | 0.2179 |
| decision tree | 0.6606 | 0.6424 | 0.2051 |

---

## sonuç ve yorum

iki model de birbirine yakın sonuç verdi, decision tree biraz daha iyi accuracy ve precision aldı ama recall değeri daha düşük. recall değerlerinin düşük olması modelin covid pozitif kişileri kaçırdığı anlamına geliyor, bu tıbbi bir uygulamada problem olurdu.

daha iyi sonuç almak için random forest ya da daha fazla özellik kullanılabilir. veri setinde hasta/sağlıklı dağılımı dengeli olmadığından (yaklaşık 660k negatif, 390k pozitif) bu da performansı etkiliyor olabilir.
