# 🎣 Deteksi Website Phishing dengan Deep Learning

Project ini membandingkan **3 arsitektur Deep Learning** (ANN/MLP, CNN 1D, LSTM) untuk
mendeteksi apakah sebuah website bersifat **Phishing** atau **Legitimate (aman)**, lengkap
dengan notebook evaluasi menyeluruh dan aplikasi web (Streamlit) tempat pengguna bisa
memasukkan URL secara langsung.

## 🚀 Cara Menjalankan
### 1. Install dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. (Opsional) Jalankan ulang notebook training

Folder `streamlit_app/models/` **sudah berisi model hasil training** yang siap pakai — kamu
tidak wajib menjalankan ulang notebook. Tapi kalau ingin melatih ulang / bereksperimen:

```bash
pip install jupyter
jupyter notebook phishing_classification_ANN_CNN_LSTM.ipynb
# Run All Cells — akan otomatis menyimpan ulang model ke streamlit_app/models/
```

Notebook berjalan dari root folder ini (path dataset & output artefak relatif terhadap
lokasi notebook), pastikan menjalankannya dari folder project ini, bukan dari dalam
`streamlit_app/`.

### 3. Jalankan aplikasi web

```bash
cd streamlit_app
streamlit run app.py
```

Buka browser ke `http://localhost:8501`, masukkan URL, klik **Deteksi**.

## 🧠 Tentang Dataset

**UCI Phishing Websites Dataset** — 11.054 sampel, 30 fitur biner/ternary (-1/0/1) yang
merepresentasikan karakteristik URL & website (panjang URL, penggunaan IP, HTTPS, umur
domain, dll.), plus label `class` (1 = legitimate, -1 = phishing).

Di notebook, target dikonversi ke encoding **1 = Phishing, 0 = Legitimate** supaya metrik
TPR/FPR dsb. langsung merepresentasikan "seberapa baik model menangkap situs phishing".

## 🏗️ 3 Model Deep Learning

| Model     | Arsitektur                             | Ide                                            |
| --------- | -------------------------------------- | ---------------------------------------------- |
| **ANN**   | Dense(128)→Dense(64)→Dense(32)→sigmoid | Baseline MLP untuk data tabular                |
| **CNN1D** | Conv1D→Conv1D→GlobalAvgPool→Dense      | Menangkap pola lokal antar fitur berdekatan    |
| **LSTM**  | LSTM(64)→LSTM(32)→Dense                | Fitur diperlakukan sebagai "urutan" sekuensial |

## 📊 Evaluasi

Notebook menghitung, untuk masing-masing model, **Confusion Matrix** (TP, TN, FP, FN) dan
7 metrik turunan:

- **TPR (Recall/Sensitivity)** — seberapa banyak situs phishing yang berhasil ditangkap
- **TNR (Specificity)** — seberapa banyak situs aman yang benar dikenali aman
- **FPR** — situs aman yang salah ditandai phishing
- **FNR** — situs phishing yang lolos/tidak terdeteksi
- **PPV (Precision)** — dari semua yang ditandai phishing, berapa persen benar phishing
- **NPV** — dari semua yang ditandai aman, berapa persen benar aman
- **Accuracy**

Beserta visualisasi lengkap: 3 confusion matrix heatmap, bar chart tiap metrik (Accuracy,
TPR, TNR, FPR, FNR, PPV, NPV), grouped bar chart gabungan seluruh metrik, dan ROC curve.

Hasil pada test set (lihat notebook untuk detail lengkap):

| Model | Accuracy | TPR    | TNR    | FPR    | FNR    | PPV    | NPV    |
| ----- | -------- | ------ | ------ | ------ | ------ | ------ | ------ |
| ANN   | ~92.6%   | ~91.8% | ~93.4% | ~6.6%  | ~8.2%  | ~93.7% | ~91.5% |
| CNN1D | ~91.2%   | ~89.6% | ~92.9% | ~7.1%  | ~10.4% | ~93.1% | ~89.4% |
| LSTM  | ~86.2%   | ~86.3% | ~86.1% | ~13.9% | ~13.7% | ~86.9% | ~85.5% |

_(Angka bisa sedikit berbeda tiap kali notebook dijalankan ulang karena inisialisasi bobot
random, meskipun sudah di-seed.)_

## ⚠️ Keterbatasan Penting: Ekstraksi Fitur dari URL Live

Dataset asli dibuat tahun 2015 dan sebagian fiturnya (`WebsiteTraffic`/Alexa Rank,
`PageRank` Google, `GoogleIndex`, `LinksPointingToPage`) awalnya dihitung dari layanan
pihak ketiga yang **sekarang sudah tidak beroperasi atau tidak bisa diakses otomatis**:

- Alexa Rank resmi ditutup 2022.
- Google PageRank publik sudah lama dimatikan.
- Query otomatis ke Google Search akan diblokir/rate-limited.
- Layanan pengecekan backlink gratis sudah tidak tersedia.

Untuk fitur-fitur tersebut, `feature_extractor.py` memakai **pendekatan heuristik terbaik**
(bukan nilai asli dari layanan tersebut), dan hal ini **ditampilkan transparan ke pengguna**
di aplikasi (bagian "fitur didekati secara heuristik"). Fitur lain (lexical URL, HTTPS,
WHOIS, DNS, konten HTML/JS halaman) diekstrak langsung dari URL secara real-time dan cukup
akurat.

**Implikasi:** akurasi pada URL live di dunia nyata bisa sedikit lebih rendah dibanding
akurasi pada test set dataset asli. Gunakan hasil deteksi sebagai salah satu sinyal, bukan
satu-satunya dasar keputusan.

## 🛠️ Troubleshooting

- **Model gagal dimuat / error shape mismatch** → klik tombol **🔄 Muat Ulang Model
  (reset cache)** di sidebar aplikasi. Kalau masih gagal, pastikan versi library sesuai
  `requirements.txt` (`pip install -r requirements.txt` ulang).
- **Deteksi URL lambat** → wajar, karena melibatkan fetch halaman + WHOIS + DNS lookup
  (bisa 5–20 detik tergantung kecepatan situs target & server WHOIS).
- **Semua fitur konten (Favicon, RequestURL, dst.) bernilai 0** → halaman target gagal
  diakses (situs down, memblokir bot, atau butuh JavaScript rendering yang tidak didukung
  oleh scraping sederhana ini).
