# Telco Churn Tahmin Sistemi

Bu repository, **Telco Customer Churn** problemi için analizden üretime giden tam bir veri bilimi çözümü sunar.

Proje kapsamında:

- detaylı keşifsel veri analizi (EDA),
- gelişmiş ön işleme ve özellik mühendisliği,
- çoklu model benchmark ve model seçimi,
- açıklanabilirlik (explainability) ve segment analizi,
- FastAPI ile servisleme,
- Streamlit ile iş birimi odaklı arayüz,
- Docker ve Docker Compose ile çalıştırma

gerçeklenmiştir.

---

## İçindekiler

1. [Projenin Amacı](#projenin-amacı)
2. [Mimari Özeti](#mimari-özeti)
3. [Klasör Yapısı](#klasör-yapısı)
4. [Kurulum](#kurulum)
5. [Uçtan Uca Çalıştırma Akışı](#uçtan-uca-çalıştırma-akışı)
6. [Modelleme ve Threshold Stratejisi](#modelleme-ve-threshold-stratejisi)
7. [FastAPI Kullanımı](#fastapi-kullanımı)
8. [Streamlit Kullanımı](#streamlit-kullanımı)
9. [Batch Skorlama Kuralları](#batch-skorlama-kuralları)
10. [Recommendation Motoru](#recommendation-motoru)
11. [Testler](#testler)
12. [Docker](#docker)
13. [Docker Compose](#docker-compose)
14. [Sık Karşılaşılan Sorunlar ve Çözümler](#sık-karşılaşılan-sorunlar-ve-çözümler)
15. [Teslim Kontrolü (YZTA Dokümanı Uyum)](#teslim-kontrolü-yzta-dokümanı-uyum)

---

## Projenin Amacı

Amaç, bir telekom müşterisinin churn (abonelik iptali) riskini tahmin eden ve bu tahmini aksiyona dönüştüren bir sistem oluşturmaktır.

Bu nedenle proje sadece bir model eğitmekten ibaret değildir; aşağıdaki bileşenlerin birlikte çalışmasını hedefler:

- **Tahmin üreten model**
- **Canlı API servisi**
- **İş biriminin kullanabileceği arayüz**
- **Yorumlanabilir sonuçlar ve öneri çıktısı**

---

## Mimari Özeti

Akış yüksek seviyede aşağıdaki gibidir:

1. `data/raw/telco.csv` verisi yüklenir.
2. `src/preprocessing.py` içinde temizlik + feature engineering uygulanır.
3. `src/train.py` ile birden fazla model eğitilir, kıyaslanır ve champion model kaydedilir.
4. Champion model `models/` altında artifact olarak saklanır.
5. `src/app.py` API’si bu artifact’ları kullanarak canlı tahmin üretir.
6. `streamlit_app.py` API’ye bağlanıp tekil ve batch skorlama sunar.

---

## Klasör Yapısı

```text
telco_churn/
├─ data/
│  └─ raw/
│     └─ telco.csv
├─ models/
│  ├─ champion_pipeline.joblib
│  ├─ champion_metadata.json
│  └─ decision_threshold.json
├─ notebooks/
│  ├─ 01_eda_overview.ipynb
│  ├─ 02_preprocessing_diagnostics.ipynb
│  ├─ 03_model_benchmark.ipynb
│  └─ 04_explainability_and_segments.ipynb
├─ reports/
│  ├─ model_metrics_validation.csv
│  ├─ segment_audit.csv
│  ├─ permutation_importance.csv
│  └─ ...
├─ src/
│  ├─ app.py
│  ├─ config.py
│  ├─ evaluation.py
│  ├─ explainability.py
│  ├─ inference.py
│  ├─ models.py
│  ├─ preprocessing.py
│  └─ train.py
├─ tests/
│  └─ test_api.py
├─ optimize.py
├─ streamlit_app.py
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

## Kurulum

### 1) Sanal ortam oluştur

```bash
python -m venv .venv
```

### 2) Ortamı aktive et (Windows PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

### 3) Bağımlılıkları yükle

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Not: Dosya upload endpoint’leri için `python-multipart` gereklidir ve `requirements.txt` içinde tanımlıdır.

---

## Uçtan Uca Çalıştırma Akışı

### A) Model eğitimi

```bash
python -m src.train --data-path data/raw/telco.csv
```

Bu adım sonunda temel artifact’lar üretilir:

- `models/champion_pipeline.joblib`
- `models/champion_metadata.json`
- `models/decision_threshold.json`
- `reports/model_metrics_validation.csv`
- `reports/segment_audit.csv`

### B) API başlat

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### C) Streamlit başlat

```bash
streamlit run streamlit_app.py
```

---

## Modelleme ve Threshold Stratejisi

Projede birden fazla model eğitilip karşılaştırılmıştır (lojistik regresyon, ağaç tabanlı yöntemler, boosting ailesi vb.).

Threshold yönetimi üç seviyede ele alınır:

- `default` (genel 0.5 yaklaşımı)
- `cost_optimal` (iş maliyeti odaklı eşik)
- `f1_optimal` (denge odaklı eşik)

Güncel production eşiği metadata üzerinden yönetilir ve API çağrılarında istenirse `threshold_override` ile anlık değiştirilebilir.

---

## FastAPI Kullanımı

### Önemli Endpoint’ler

- `GET /health` → servis ve model yüklenme durumu
- `GET /metadata` → model metadata + aktif threshold
- `POST /predict` → tekil JSON veya dosya upload ile tahmin
- `POST /predict/batch` → JSON liste ile batch tahmin
- `POST /recommend` → tahmin + aksiyon planı
- `POST /admin/reload` → model/threshold cache yenileme

### `/predict` giriş tipleri

1. `application/json` (tek müşteri)
2. `multipart/form-data` + `file` (`.csv` veya `.json`)

### Threshold Override

Aşağıdaki gibi query param ile çağrı bazında karar eşiği değiştirilebilir:

- `/predict?threshold_override=0.65`
- `/predict/batch?threshold_override=0.55`
- `/recommend?threshold_override=0.70`

---

## Streamlit Kullanımı

Arayüzde aşağıdaki özellikler bulunur:

- Tekil müşteri skorlaması
- Batch CSV/JSON upload
- Sonuçları CSV dışa aktarma
- API hata detayını kullanıcıya gösterme
- **Manuel threshold slider (0.01–0.99)**
- Gelişmiş öneri planı görüntüleme

Streamlit varsayılan API adresi:

- `http://127.0.0.1:8000`

Docker Compose içinde otomatik olarak servis içi adres (`http://api:8000`) kullanılır.

---

## Batch Skorlama Kuralları

Batch tahmin için beklenen temel alanlar:

`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

Opsiyonel alan:

- `customerID` (varsa çıktıda `customer_id` olarak döner)

Notlar:

- `TotalCharges` boş gelebilir (pipeline imputer ele alır)
- Eksik kolon durumunda API 422 döner ve eksik alanı `detail` içinde belirtir

---

## Recommendation Motoru

`/recommend` endpoint’i iki seviyeli çıktı sağlar:

1. Geriye dönük uyumluluk alanları:
   - `actions`
   - `rationale`

2. Zengin öneri planı:
   - `recommendation_plan[]`
     - `priority`
     - `action`
     - `rationale`
     - `expected_impact`
     - `campaign_type`

Bu yapı, önerileri operasyonel kampanyalara dönüştürmeyi kolaylaştırır.

---

## Testler

Tüm testleri çalıştır:

```bash
pytest -q
```

Test kapsamı:

- API health/metadata davranışı
- single + batch tahmin
- CSV/JSON upload
- çoklu müşteri desteği
- boş `TotalCharges` edge-case
- threshold override davranışı
- recommendation endpoint davranışı

---

## Docker

Image build:

```bash
docker build -t telco-churn-api .
```

### API modu (varsayılan)

```bash
docker run -p 8000:8000 telco-churn-api
```

Health kontrol:

- `http://localhost:8000/health`

### Streamlit modu

```bash
docker run -e APP_MODE=streamlit -p 8501:8501 telco-churn-api
```

UI:

- `http://localhost:8501`

---

## Docker Compose

API + Streamlit’i birlikte kaldır:

```bash
docker compose up --build
```

Servisler:

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

Kapat:

```bash
docker compose down
```

---

## Sık Karşılaşılan Sorunlar ve Çözümler

### 1) Batch scoring 422 hatası

Nedenler:

- Eksik kolon
- Bozuk JSON
- Hatalı dosya formatı

Çözüm:

- Streamlit’te dönen `detail` mesajını kontrol edin
- Kolon adlarını “Batch Skorlama Kuralları” bölümüne göre doğrulayın

### 2) `Model artifact not found` / 503

Önce eğitim çalıştırın:

```bash
python -m src.train --data-path data/raw/telco.csv
```

### 3) Eşik değişti ama API eski davranıyor

Cache yenileyin:

```bash
curl -X POST http://127.0.0.1:8000/admin/reload
```

---

## Teslim Kontrolü (YZTA Dokümanı Uyum)

YZTA 5.0 P2P 2 dokümanındaki ana beklentilerle uyum durumu:

- Python ile geliştirme: **Tamam**
- Veri analizi ve model geliştirme: **Tamam**
- Birden fazla model deneme/karşılaştırma: **Tamam**
- Predict endpoint’i olan API: **Tamam**
- Çalışan uçtan uca proje: **Tamam**
- Dokümantasyon: **Tamam**
- Docker (opsiyonel): **Tamam**
- Basit arayüz (opsiyonel): **Tamam**

Süreç tarafında ayrıca beklenen madde:

- Git tabanlı depoda teslim/push adımı (kod dışı operasyonel adım)

---

## Sonuç

Bu proje, veri bilimi çözümünü sadece model eğitimi seviyesinde bırakmayıp üretime taşır:

- analiz,
- modelleme,
- açıklanabilirlik,
- servisleme,
- UI,
- containerization

katmanlarını birlikte sunar.

İstersen bir sonraki adımda README’ye örnek `curl` istekleri ve örnek batch CSV şablonunu da ekleyebilirim.
