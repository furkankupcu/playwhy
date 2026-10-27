# ⚽ Goal Explainer

> **"Give it a short television broadcast clip of a football goal, and it explains why the goal happened — the way a good analyst would, with every claim tied to a moment in the clip."**

Goal Explainer combines **spatial positioning data** with **visual player cue analysis** using only the **Meta Model API** (SAM 3.1 & Muse Spark). No models are trained or fine-tuned.

---

## 🏗️ Mimari ve Akış (Architecture)

```
clip.mp4 ─► [1 track] ─► [2 calibrate] ─► [3 geometry] ─► [4 read] ─► [5 explain] ─► UI
              SAM 3.1      piksel→metre     olaylar/anlar    Muse Spark   Muse Spark
```

Tüm adımlar çıktılarını `runs/<clip_id>/` dizini altına JSON olarak kaydeder:

| Adım | Modül | Girdi | Çıktı | Açıklama |
|---|---|---|---|---|
| **1. Track** | `goal_explainer.track` | `video.mp4` | `tracks.json`, `tracks_overlay.mp4` | Muse Spark ilk frame'den iki takımın forma rengini okur; SAM 3.1 her sınıf için ayrı prompt'la (top, kaleci, "player in red shirt", …, hakem) takip eder. Takım prompt'tan gelir. Kayıp top interpolasyonu. |
| **2. Calibrate** | `goal_explainer.calibrate` | `video.mp4` + çizgiler | `homography.json` | Pikselden saha metrelerine homografi ($H$). Sırayla: a) Muse Spark keypoint tespiti, b) Hough çizgi + şablon eşleme, c) Manuel 4 nokta yedek kalibrasyon. ORB + RANSAC kare takibi ve kayma düzeltmesi. |
| **3. Geometry** | `goal_explainer.geometry` | `tracks.json` + $H$ | `pitch.json`, `events.json` | Savitzky–Golay filtreleme. Topa sahip olma tespiti, paslar, pas zinciri, ofsayt çizgisi analizi, şut mesafesi/açısı, açık pas koridorları ve adamını kaybetme anları. |
| **4. Read** | `goal_explainer.read` | Önemli anlar + kırpıntılar | `cues.json` | Muse Spark ile vücut yönü, bakış açısı, el/kol hareketi (ofsayt itirazı), kayma/denge kaybı, reaksiyon gecikmesi çıkarımı. |
| **5. Explain** | `goal_explainer.explain` | `events.json` + `cues.json` | `explanation.json` | Kanıta dayalı taktiksel çıkarımlar (3–6 iddia). Her iddia kesinlikle somut bir olay/ipucu referansına dayanmak zorundadır (kod düzeyinde doğrulanır). |
| **6. Replay** | `goal_explainer.replay` | `pitch.json` + `events.json` | `replay.mp4` | 2D kuşbakışı radar/taktik tahtası animasyonu, hareket izleri (trails), pas okları ve ofsayt çizgisi. |

---

## 📁 Proje Yapısı

```
MetaHack/
├── goal_explainer/
│   ├── __init__.py
│   ├── __main__.py      # python -m goal_explainer clip.mp4
│   ├── api.py           # Meta Model API istemcisi + SHA256 disk önbelleği
│   ├── track.py         # 1. Adım: SAM 3.1 video takibi
│   ├── calibrate.py     # 2. Adım: Kamera kalibrasyonu (Homografi)
│   ├── pitch.py         # FIFA standart saha şablonu (105 × 68 m)
│   ├── geometry.py      # 3. Adım: Metrik uzay analizi ve olay tespiti
│   ├── read.py          # 4. Adım: Muse Spark ile oyuncu görsel ipuçları
│   ├── explain.py       # 5. Adım: Kanıta dayalı taktiksel açıklama
│   ├── replay.py        # 6. Adım: Kuşbakışı 2D radar videosu
│   └── pipeline.py      # Pipeline orkestrasyonu (CLI)
├── eval/
│   ├── __init__.py
│   └── gsr_eval.py      # SoccerNet-GSR üzerinde metre hatası değerlendirmesi (Hungarian)
├── tests/               # Pytest birim ve entegrasyon testleri
├── server.py            # FastAPI web uygulaması (API + static/index.html)
├── static/index.html    # Arayüz: yükleme, ilerleme, 4 noktalı kalibrasyon, sonuç
├── docs/
│   ├── application.md   # Proje hackathon başvurusu
│   └── PLAN.md          # 10 günlük geliştirme ve test planı
└── requirements.txt
```

---

## 🚀 Kurulum

```bash
git clone https://github.com/your-username/GoalExplainer.git
cd GoalExplainer

# Sanal ortam oluşturup aktifleştirin
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Ortam Değişkenleri (Meta Model API)

```bash
export MODEL_API_KEY="your-meta-model-api-key"   # dev.meta.ai → API keys
# isteğe bağlı: export META_API_BASE="https://api.meta.ai/v1"
```

---

## 💻 Kullanım

### 1. Komut Satırı Arayüzü (CLI Pipeline)

```bash
# Tüm adımları baştan sona çalıştırır
python -m goal_explainer.pipeline path/to/clip.mp4

# Belirli bir adımdan itibaren devam etmek için (--skip-to)
python -m goal_explainer.pipeline path/to/clip.mp4 --skip-to 3

# Manuel kalibrasyon noktaları ile çalıştırma (Yedek Yol C)
python -m goal_explainer.pipeline path/to/clip.mp4 --manual-points '[{"px": [120, 340], "pitch": [16.5, 13.84]}]'
```

### 2. Web arayüzü (FastAPI)

```bash
uvicorn server:app --port 8000
```
Repo kökünden çalıştırın (sonuçlar `./runs` altına yazılır). Tarayıcıda `http://localhost:8000`:
- Orijinal gol klibini yükleyin,
- Otomatik veya manuel kalibrasyon seçeneğini kullanın,
- Solda yayın klibini, sağda videoyla senkron kuşbakışı sahayı izleyin (ofsayt çizgisi, top yolu),
- Her iddianın altında listelenen somut kanıtları (pas, ofsayt, bakış açısı vb.) inceleyin.

---

## 📊 Değerlendirme (SoccerNet-GSR)

Modelin pikselden metreye tahmin başarısını `sn-gamestate` veri setine karşı test etmek için:

```bash
python -m eval.gsr_eval --runs-dir runs --gsr-dir /path/to/soccernet_gsr
```

- **Metrikler:** Ortalama Hata (Mean Error, metre), Medyan Hata (Median Error, metre), Recall (5m eşiği içinde eşleşen oyuncu yüzdesi).
- **Eşleme:** Hungarian Algoritması (`scipy.optimize.linear_sum_assignment`).

---

## 🧪 Testler

Projedeki 34 birim ve entegrasyon testini çalıştırmak için:

```bash
pytest tests/
```
