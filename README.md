---
title: Agro Fertilizer Regression
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Web-Based Regression Project (web_5_regresion)

Proyek ini adalah aplikasi berbasis web yang mengimplementasikan model Machine Learning **Regresi** untuk melakukan prediksi atau analisis data. Dikembangkan menggunakan **Python** sebagai backend utama untuk pemrosesan data dan modeling, serta antarmuka web interaktif untuk mempermudah pengguna dalam berinteraksi dengan model.

## 🚀 Fitur Utama

- **Interactive Web Interface**: Antarmuka pengguna yang bersih dan intuitif untuk memasukkan data input.
- **Regression Predictive Model**: Menggunakan algoritma Regresi (seperti Linear Regression, Random Forest Regression, atau sejenisnya) yang dilatih menggunakan `scikit-learn`.
- **Real-time Prediction**: Menampilkan hasil prediksi secara langsung setelah pengguna menekan tombol input.
- **Data Visualization**: Menyediakan visualisasi tren data atau hasil evaluasi model (jika tersedia).

## 🛠️ Teknologi & Library yang Digunakan

Aplikasi ini dibangun dengan ekosistem teknologi berikut:

- **Bahasa Pemrograman**: Python 3.x
- **Framework Web**: Flask / Streamlit (Sesuai arsitektur backend proyek)
- **Machine Learning & Data Science**:
  - `scikit-learn` (Untuk implementasi dan pelatihan model regresi)
  - `pandas` & `numpy` (Untuk manipulasi dan analisis data)
  - `matplotlib` / `seaborn` (Untuk visualisasi data grafis)
- **Frontend**: HTML5, CSS3, JavaScript (atau TailwindCSS/Bootstrap untuk styling)

## 📁 Struktur Direktori

Berikut adalah estimasi struktur komponen di dalam repositori ini:

```text
web_5_regresion/
├── dataset/                # Menyimpan file dataset (CSV/XLSX)
├── models/                 # Menyimpan model regresi yang sudah di-export (.pkl / .joblib)
├── templates/              # Halaman antarmuka HTML (jika menggunakan Flask)
├── static/                 # Aset statis seperti CSS, JS, dan Gambar
├── app.py / main.py        # File utama untuk menjalankan server web
├── regression_model.ipynb  # Notebook eksperimen dan pelatihan model
├── requirements.txt        # Daftar dependencies library Python
└── README.md               # Dokumentasi proyek
```

## ⚙️ Langkah Instalasi & Menjalankan Proyek

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda:

### 1. Clone Repositori
```bash
git clone https://github.com/Zekken23/web_5_regresion.git
cd web_5_regresion
```

### 2. Buat & Aktifkan Virtual Environment (Direkomendasikan)
- **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instal Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
Jalankan file utama untuk memulai server lokal:
```bash
python app.py
```
atau jika menggunakan Streamlit:
```bash
streamlit run app.py
```

Buka browser Anda dan akses alamat yang tertera (biasanya `http://127.0.0.1:5000` atau `http://localhost:8501`).

## 📊 Detail Model Regresi

Model ini memproses fitur-fitur input yang dimasukkan oleh pengguna, melakukan pra-pemrosesan (scaling/encoding jika diperlukan), dan mengembalikan nilai kontinu sebagai output prediksi. Evaluasi model diukur menggunakan metrik performa standar seperti:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared ($R^2$) Score**

---
Dikembangkan sebagai bagian dari eksperimen implementasi Machine Learning pada platform Web.
