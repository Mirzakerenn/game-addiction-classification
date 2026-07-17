# 🎮 Klasifikasi Kecanduan Game

Proyek Machine Learning untuk mengklasifikasikan apakah seorang gamer terindikasi kecanduan game atau tidak berdasarkan pola perilaku bermain mereka.

---

## 📌 Deskripsi

Menggunakan dataset perilaku gaming online dari Kaggle, proyek ini membangun dan membandingkan dua model klasifikasi biner:

- **Decision Tree Classifier**
- **Random Forest Classifier**

Target klasifikasi (`addicted`) dibuat dari kolom `EngagementLevel`:
- `1` = Kecanduan (EngagementLevel == "High")
- `0` = Tidak Kecanduan

---

## 📁 Struktur Project

```
game-addiction-classification/
├── online_gaming_behavior.csv               # Dataset dari Kaggle
├── klasifikasi.py                           # Script utama (training + evaluasi + visualisasi)
├── perbandingan_model.png                   # Grafik perbandingan performa kedua model
├── Klasifikasi_Kecanduan_Game_UAS.pptx      # Presentasi
└── README.md
```

---

## 🗂️ Dataset

- **Sumber:** [Kaggle - Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)
- **Jumlah data:** 40.034 baris
- **Fitur yang digunakan:**
  - `Age`
  - `PlayTimeHours`
  - `SessionsPerWeek`
  - `AvgSessionDurationMinutes`

---

## ⚙️ Cara Menjalankan

1. Clone repo ini:
   ```bash
   git clone https://github.com/Mirzakerenn/game-addiction-classification.git
   cd game-addiction-classification
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

3. Jalankan script (**pastikan terminal sudah berada di dalam folder project**):
   ```bash
   python klasifikasi.py
   ```

---

## 📊 Hasil Perbandingan Model

| Metrik     | Decision Tree | Random Forest |
|------------|:-------------:|:-------------:|
| Accuracy   | 0.8938        | 0.9417        |
| Precision  | 0.8942        | 0.9411        |
| Recall     | 0.8938        | 0.9417        |
| F1-Score   | 0.8940        | 0.9411        |

**Random Forest unggul di semua metrik** karena menggunakan ensemble dari 100 Decision Tree, sehingga lebih tahan terhadap overfitting.

---

## 🛠️ Tech Stack

- Python 3
- pandas
- scikit-learn
- matplotlib
