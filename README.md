# Chatbot Kesehatan Ibu Hamil - Setup Guide

## Deskripsi Sistem

Sistem chatbot ini menggunakan arsitektur pipeline dengan komponen-komponen berikut:
1. **Authentication**: Login dengan NIK dan password menggunakan data dari `customer.csv`
2. **Intent Classification**: Pipeline dengan similarity check + domain classification + intent classification menggunakan BERT models
3. **Database Integration**: Mengintegrasikan 13 tabel CSV untuk mengambil data kontekstual
4. **LLM Integration**: Menggunakan Google Gemini untuk generate response
5. **Recommendation Engine**: Generate pertanyaan rekomendasi berdasarkan riwayat user

## Prasyarat Sistem

### 1. Python Requirements
```bash
pip install -r requirements.txt
```

### 2. API Key Setup
1. Dapatkan Gemini API key dari: https://makersuite.google.com/app/apikey
2. Buat folder `.streamlit` di root project
3. Buat file `secrets.toml` dalam folder `.streamlit`
4. Isi dengan format:
```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

### 3. Struktur Directory
Pastikan struktur folder sesuai:
```
Project 1/
├── chatbot_app.py (main app)
├── auth_handler.py
├── intent_classifier.py
├── database_handler.py
├── llm_handler.py
├── recommendation_engine.py
├── requirements.txt
├── deskripsi_inten.md
├── intent_merged.csv
├── pertanyaan_umum_kehamilan.txt
├── Database/
│   ├── customer.csv
│   ├── anc_kunjungan.csv
│   ├── diagnosis.csv
│   ├── dokter.csv
│   ├── hasil_lab.csv
│   ├── historikal_kondisi_fisik.csv
│   ├── imunisasi_ibu_hamil.csv
│   ├── jadwal_dokter.csv
│   ├── kehamilan.csv
│   ├── persalinan.csv
│   ├── preskripsi.csv
│   ├── riwayat_berobat.csv
│   └── suplemen_ibu_hamil.csv
├── Model BERT/
│   ├── model_hamil/
│   ├── model_hamil_umum/
│   └── model_umum/
└── .streamlit/
    └── secrets.toml
```

## Cara Menjalankan

### 1. Install Dependencies
```bash
cd "C:\Users\farre\Documents\Kuliah\Magang era\Project 1"
pip install -r requirements.txt
```

### 2. Setup API Key
```bash
mkdir .streamlit
# Edit .streamlit/secrets.toml dan masukkan API key
```

### 3. Jalankan Aplikasi
```bash
streamlit run chatbot_app.py
```

### 4. Akses Aplikasi
- Buka browser ke: http://localhost:8501
- Login dengan NIK dan password dari `customer.csv`

## Akun Test
Berdasarkan `customer.csv`:
- NIK: 3276011234567890
- Password: 123456
- Nama: Siti Aminah

## Fitur Utama

### 1. Authentication System
- Login menggunakan NIK dan password
- Validasi terhadap database customer
- Session management

### 2. Intent Classification Pipeline
```
User Input → Similarity Check → Domain Classification → Intent Classification → Response Generation
```

### 3. Recommendation Engine
- 4 pertanyaan rekomendasi berdasarkan riwayat user
- Generated menggunakan Gemini LLM
- Fallback ke rule-based jika LLM gagal

### 4. Chat Interface
- Chat history persistence dalam session
- Mobile-friendly responsive design
- Real-time processing dengan loading indicators

### 5. Database Integration
Mengintegrasikan semua 13 tabel CSV:
- customer.csv (autentikasi & info dasar)
- anc_kunjungan.csv (kunjungan ANC)
- diagnosis.csv (diagnosis medis)
- dokter.csv (info dokter)
- hasil_lab.csv (hasil laboratorium)
- historikal_kondisi_fisik.csv (kondisi fisik)
- imunisasi_ibu_hamil.csv (imunisasi)
- jadwal_dokter.csv (jadwal praktik)
- kehamilan.csv (data kehamilan)
- persalinan.csv (data persalinan)
- preskripsi.csv (resep obat)
- riwayat_berobat.csv (riwayat berobat)
- suplemen_ibu_hamil.csv (suplemen)

## Arsitektur Pipeline

### 1. Similarity Check
- Menggunakan SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
- Threshold: 0.7 (dapat diubah di intent_classifier.py)
- Jika similarity < threshold → "Maaf, itu di luar fitur saya"

### 2. Domain Classification
- Model: model_hamil_umum (BERT)
- Output: 'KEHAMILAN' atau 'UMUM'
- Fallback: keyword-based classification

### 3. Intent Classification
- KEHAMILAN domain → model_hamil (BERT)
- UMUM domain → model_umum (BERT)
- 17 intents total sesuai deskripsi_inten.md

### 4. Response Generation
- Menggunakan Google Gemini Pro
- Context: user input + intent + database data + knowledge base
- Strict rules: no medical advice, database-only responses

## Troubleshooting

### 1. Model Loading Issues
- Pastikan folder "Model BERT" ada dan berisi model files
- Jika model tidak ada, sistem akan fallback ke similarity-based classification

### 2. API Key Issues
- Pastikan GEMINI_API_KEY di secrets.toml benar
- Check internet connection
- Verify API key di Google AI Studio

### 3. Database Issues
- Pastikan semua CSV files ada di folder Database
- Check encoding files (UTF-8)
- Pastikan column names sesuai dengan yang expected

### 4. Dependencies Issues
```bash
# Jika error dengan transformers/torch
pip install --upgrade torch transformers

# Jika error dengan sentence-transformers
pip install --upgrade sentence-transformers

# Jika error dengan streamlit
pip install --upgrade streamlit
```

## Customization

### 1. Ubah Similarity Threshold
```python
# Di intent_classifier.py line 22
def __init__(self, base_path, similarity_threshold=0.7):  # Ubah nilai ini
```

### 2. Tambah Default Recommendations
```python
# Di recommendation_engine.py, ubah default_recommendations
```

### 3. Ubah UI Theme
```python
# Di chatbot_app.py, edit CSS di st.markdown
```

### 4. Customize LLM Prompt
```python
# Di llm_handler.py, edit _build_prompt method
```

## Logging

Sistem menggunakan Python logging. Log levels:
- INFO: operasi normal
- WARNING: fallback operations
- ERROR: error conditions

Untuk debug, edit di setiap file:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

### 1. Streamlit Cloud
1. Upload ke GitHub repository
2. Deploy via https://share.streamlit.io
3. Set secrets di Streamlit Cloud dashboard

### 2. Local Server
```bash
streamlit run chatbot_app.py --server.port 8501 --server.address 0.0.0.0
```

### 3. Docker (Optional)
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "chatbot_app.py"]
```

## Security Notes

1. **API Key Security**: Jangan commit API key ke repository
2. **User Data**: Data customer disimpan dalam session state (tidak persistent)
3. **Input Validation**: Sistem melakukan basic validation pada input
4. **HTTPS**: Gunakan HTTPS untuk production deployment

## Support

Jika ada issues:
1. Check logs untuk error messages
2. Verify semua dependencies terinstall
3. Pastikan API key valid
4. Check file permissions dan encoding
