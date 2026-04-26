# Telco Churn Projesi (Uçtan Uca ML + API + Streamlit)

Bu proje, Kaggle Telco Customer Churn verisi üzerinde:

- detaylı veri analizi (EDA),
- gelişmiş feature engineering,
- çoklu model benchmark,
- explainability ve segment analizi,
- production seviyesinde FastAPI servisleme,
- Streamlit arayüzü

akışını uçtan uca sunar.

Ana amaç, churn riskini tahmin etmek ve tahmin sonucunu aksiyona dönüştüren bir retention motoru sağlamaktır.

---

## 1) Proje Kapsamı ve Yapılanlar

Bu repository içinde aşağıdaki fazlar tamamlanmıştır:

- **EDA (Notebook 01):** Dağılımlar, churn kırılımları, iş metrikleri, segment bazlı içgörüler.
- **Preprocessing Diagnostics (Notebook 02):** Temizlik, veri sızıntısı (leakage) kontrolleri, encode/scale doğrulaması.
- **Model Benchmark (Notebook 03):** Çoklu model karşılaştırması, threshold analizi, kalibrasyon, lift/gain.
- **Explainability + Segments (Notebook 04):** SHAP, global/local açıklanabilirlik, persona/segment odaklı analiz.
- **Production `src/` katmanı:** Eğitim, inference, değerlendirme, API ve UI entegrasyonu.

Ek olarak:

- `/predict` endpoint’i hem **JSON body** hem **CSV/JSON dosya upload** destekler.
- Batch skorlamada çoklu müşteri desteklenir.
- Çıktıda `customer_id` döndürülür.
- Streamlit’te **manuel threshold slider** ile canlı karar eşiği ayarlanabilir.
- Recommend motoru güçlendirilmiş, öncelikli aksiyon planı üretir.

---

## 2) Dizin Yapısı

- `data/raw/telco.csv`: Ham veri.
- `notebooks/`: 4 detaylı analiz notebook’u.
- `src/preprocessing.py`: Temizlik + feature engineering + preprocessor.
- `src/models.py`: Model registry + metrik yardımcıları.
- `src/evaluation.py`: Threshold, lift/gain, bootstrap CI, segment audit fonksiyonları.
- `src/explainability.py`: Coefficient/permutation/SHAP yardımcıları.
- `src/inference.py`: Artifact yükleme, tahmin, recommend motoru.
- `src/train.py`: Eğitim, model seçimi, artifact üretimi.
- `src/app.py`: FastAPI servis katmanı.
- `streamlit_app.py`: İş birimi odaklı arayüz.
- `models/`: Üretilen model dosyaları ve threshold metadata.
- `reports/`: Metrikler, rapor CSV’leri, grafik çıktıları.
- `tests/`: API ve inference davranış testleri.

---

## 3) Kurulum

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Not: `python-multipart` dosya upload endpoint’leri için gereklidir ve `requirements.txt` içinde vardır.

---

## 4) Model Eğitimi ve Artifact Üretimi

```bash
python -m src.train --data-path data/raw/telco.csv
```

Bu komut sonunda üretilen kritik dosyalar:

- `models/champion_pipeline.joblib`
- `models/champion_metadata.json`
- `models/decision_threshold.json`
- `reports/model_metrics_validation.csv`
- `reports/segment_audit.csv`
- `reports/permutation_importance.csv`
- `reports/bootstrap_ci.csv`

---

## 5) FastAPI Kullanımı

Servisi başlat:

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### Önemli endpoint’ler

- `GET /health`: Servis ve model yüklenme durumu.
- `GET /metadata`: Model metadata + aktif threshold bilgisi.
- `POST /predict`: Tekil JSON veya dosya upload ile tahmin.
- `POST /predict/batch`: JSON liste ile batch tahmin.
- `POST /recommend`: Tahmin + aksiyon önerisi.
- `POST /admin/reload`: Artifact cache yenileme.

### `POST /predict` desteklediği giriş türleri

1. **JSON body (tek müşteri)**
2. **multipart/form-data + file** ile:
   - `.csv`
   - `.json` (tek obje veya obje listesi)

### Threshold override (opsiyonel)

Karar eşiğini çağrı bazında override etmek için:

- `/predict?threshold_override=0.65`
- `/recommend?threshold_override=0.50`
- `/predict/batch?threshold_override=0.70`

Bu durumda:

- `prediction`
- `threshold`
- `risk_band`

seçilen eşik değerine göre dinamik hesaplanır.

---

## 6) Streamlit Kullanımı

```bash
streamlit run streamlit_app.py
```

Arayüzde:

- Tekil müşteri skorlama,
- Batch CSV/JSON upload,
- Tahmin CSV export,
- API hata detayını gösterme,
- **manuel threshold slider** (0.01–0.99),
- güçlendirilmiş recommendation planı

bulunur.

---

## 7) Recommendation Motoru (Güncel)

`/recommend` çıktısı artık iki seviyeli gelir:

1. Geriye dönük uyumluluk alanları:
   - `actions`
   - `rationale`
2. Zengin plan alanı:
   - `recommendation_plan[]`
     - `priority`
     - `action`
     - `rationale`
     - `expected_impact`
     - `campaign_type`

Bu sayede aksiyonlar sadece “öneri listesi” değil, operasyonel kampanya planına dönüşür.

---

## 8) Beklenen Girdi Kolonları (Predict için)

Batch dosya tahmininde aşağıdaki ham kolonlar beklenir:

`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

Opsiyonel:

- `customerID` (varsa çıktıdaki `customer_id` alanına taşınır)

Notlar:

- `TotalCharges` boş olabilir; pipeline imputer bunu yönetir.
- Kolon adı eksikse API 422 döner ve eksik alanı detaylı bildirir.

---

## 9) Test Çalıştırma

```bash
pytest -q
```

Testler şunları doğrular:

- health/metadata endpoint davranışı,
- single + batch tahmin,
- CSV/JSON upload,
- çoklu kayıt desteği,
- boş `TotalCharges` edge-case,
- threshold override davranışı,
- recommend endpoint çıktıları.

---

## 10) Docker Kullanımı

Image oluştur:

```bash
docker build -t telco-churn-api .
```

### API modu (default)

```bash
docker run -p 8000:8000 telco-churn-api
```

- Health: `http://localhost:8000/health`

### Streamlit modu

```bash
docker run -e APP_MODE=streamlit -p 8501:8501 telco-churn-api
```

- UI: `http://localhost:8501`

---

## 11) Sık Karşılaşılan Sorunlar

### 422 Unprocessable Content (Batch)

Muhtemel nedenler:

- Eksik kolon adı
- Yanlış dosya formatı
- Bozuk JSON yapısı

Çözüm:

- Streamlit artık API `detail` mesajını gösterir.
- Kolon adlarını bu README’deki “Beklenen Girdi Kolonları” ile birebir kontrol edin.

### Model dosyası bulunamadı (503)

Önce eğitim çalıştırın:

```bash
python -m src.train --data-path data/raw/telco.csv
```

### Eşik değişti ama API eski davranıyor

Cache yenileyin:

```bash
curl -X POST http://127.0.0.1:8000/admin/reload
```

---

## 12) Son Not

Bu proje “analizden üretime” geçişin tamamını kapsar:

- veri analizi,
- modelleme,
- explainability,
- karar eşiği yönetimi,
- API servisleme,
- iş birimi arayüzü (Streamlit),
- Docker ile taşınabilir çalışma.

İstersen bir sonraki adımda `docker-compose.yml` ile API + Streamlit + (opsiyonel) Nginx reverse proxy setup’ını da ekleyebilirim.

