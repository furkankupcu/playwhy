# PlayWhy — Uygulama Planı

Spec: [application.md](application.md). Teslim: **4 Kasım 21:00 ET** (TR 5 Kasım 04:00).
Kural: yalnızca Meta Model API; hiçbir model eğitilmez veya fine-tune edilmez.

## 1. Mimari

```
clip.mp4 ─► [1 track] ─► [2 calibrate] ─► [3 geometry] ─► [4 read] ─► [5 explain] ─► UI
             SAM 3.1      piksel→metre     olaylar/anlar    Muse Spark   Muse Spark
```

Her adım çıktısını `runs/<clip_id>/` altına JSON olarak yazar. Böylece her adım tek başına yeniden çalıştırılabilir ve API çağrıları cache'lenir (para ve zaman kazandırır).

| Adım | Girdi | Çıktı |
|---|---|---|
| 1 track | video | `tracks.json`: her frame için nesne id, sınıf, maske/bbox |
| 2 calibrate | video + çizgi maskeleri | `homography.json`: her frame için 3×3 H matrisi |
| 3 geometry | tracks + H | `pitch.json` (oyuncuların metre cinsinden konumu), `events.json` (paslar, ofsayt çizgisi, şut, açık koridorlar, önemli anlar) |
| 4 read | önemli anlar + oyuncu kırpıntıları | `cues.json`: oyuncu/an başına vücut yönü, bakış, kol, kayma, tepki |
| 5 explain | events + cues | `explanation.json`: her iddia `{text, t_start, t_end, player_ids, evidence[]}` |

## 2. Repo yapısı

```
playwhy/
  api.py         # Meta Model API istemcisi + disk cache
  track.py       # adım 1
  calibrate.py   # adım 2
  pitch.py       # saha şablonu (105×68 m, çizgi ve keypoint koordinatları)
  geometry.py    # adım 3
  read.py        # adım 4
  explain.py     # adım 5
  replay.py      # kuşbakışı replay videosu
  pipeline.py    # CLI: python -m playwhy.pipeline clip.mp4
app.py           # Gradio arayüzü
eval/gsr_eval.py # SoccerNet-GSR üzerinde metre hatası
tests/
```

Stack: Python 3.11, `opencv-python`, `numpy`, `scipy`, `gradio`, `httpx`. Başka bağımlılık yok.

## 3. Adımlar — detay

### 1. Track (SAM 3.1)
- Text prompt'lar: `"football"`, `"goalkeeper"`, `"player"`, `"referee"`, `"white pitch line"`.
- **Takım ayrımı:** Takım başına prompt (`"player in red shirt"`) dene. Tutmazsa maske içindeki forma renklerini 2'ye k-means ile ayır. Bu eğitim değil, basit kümeleme.
- **Top:** Top küçük ve sık sık kayboluyor. Kaybolduğu frame'leri doğrusal interpolasyonla doldur ve bu frame'leri `interpolated: true` diye işaretle. Açıklama bu frame'lere dayanmasın.
- Oyuncunun sahadaki konumu = maskenin en alt orta noktası (ayakları).

### 2. Calibrate (en riskli adım)
1. **Başlangıç H matrisi:** Dört yolu sırayla dene, ilk çalışanı kullan:
   - a) Muse Spark'a ilk frame'i ver, görünen saha keypoint'lerini (ceza sahası köşeleri, kale direkleri vb.) piksel koordinatı olarak JSON'da iste.
   - b) a) yeterince doğru değilse çizgi maskesinden Hough ile doğruları çıkar ve saha şablonuyla eşleştir.
   - c) Yedek yol: UI'da kullanıcı ilk frame'de 4 noktaya tıklasın. Demo bu yolla her durumda çalışır.
2. **Frame'den frame'e:** Çim ve çizgi üzerinde ORB özellik noktalarını eşleştir, oyuncu maskelerini dışarıda bırak. Ardından RANSAC ile H_t→t+1'i bul ve zincirle.
3. **Kayma düzeltmesi:** Her N frame'de bir, şablon çizgilerini görüntüye geri izdüşür. Çizgi maskesiyle mesafe dönüşümü üzerinden hizalayıp H'yi `scipy.optimize` ile ince ayarla.
4. **Kalite kontrolü:** Frame başına izdüşüm hatasını kaydet. Hata eşiği aşan frame'ler "güvenilmez" işaretlenir.

### 3. Geometry
- Konumları Savitzky–Golay filtresiyle yumuşat.
- **Topa sahip olan:** topa en yakın oyuncu (≤1.5 m, en az 3 frame).
- **Pas:** topa sahip olan oyuncunun aynı takımdan birine geçmesi. **Pas zinciri:** golden geriye doğru giden paslar.
- **Ofsayt çizgisi:** her pas anında savunmanın sondan ikinci oyuncusunun x koordinatı.
- **Şut:** son temas noktası; kaleye mesafe ve iki direk arasındaki açı.
- **Açık pas koridoru:** topu tutandan takım arkadaşına çizilen doğrunun 2 m yakınında rakip olmaması.
- **Önemli anlar:** pas anları, şut anı, ve savunmacı–hücumcu mesafesinin hızla açıldığı anlar ("adamını kaybetti"). Her an için ilgili 2–4 oyuncu seçilir.

### 4. Read (Muse Spark)
- Her önemli an için ±0.5 s'lik kırpıntılar (SAM maskesi + %30 pay) ve tam frame.
- Muse Spark'tan sabit şemalı JSON iste: `{player_id, orientation, gaze, arm_raised, slip, reaction_delay, confidence, visible: bool}`. Görünmüyorsa `visible: false` desin. Uydurmasını şemayla engelle.

### 5. Explain (Muse Spark)
- Girdi: `events.json` ve `cues.json`. Muse Spark'a ham video verme; sadece bu iki dosyayla yazsın ki her iddia bir kanıta dayansın.
- Çıktı: 3–6 iddia. Her iddia `evidence` alanında bir olay id'si veya cue id'si göstermek zorunda.
- **Doğrulayıcı:** Kanıtı olmayan ya da zaman aralığı klibin dışında kalan iddiayı at. Bu kontrol kodda yapılır, modele bırakılmaz.

### UI (Gradio)
- Video yükle → solda klip, sağda kuşbakışı replay. Altta iddialar.
- Bir iddiaya tıklayınca iki video da o ana atlar.
- Yedek giriş: ilk frame'de 4 nokta tıklama.

## 4. Değerlendirme
- Veri: SoccerNet-GSR. 200 klip × 30 s, oyuncuların saha konumları elle işaretlenmiş ([sn-gamestate](https://github.com/SoccerNet/sn-gamestate)). Lisansı ve indirme koşullarını kontrol et.
- `eval/gsr_eval.py`: Tahmin edilen ve gerçek konumlar her frame'de Hungarian algoritmasıyla eşleştirilir (eşik 5 m). Rapor: ortalama ve medyan hata (m), recall, ve kalibrasyon yolu (a/b/c) başına sonuç.
- Hedef: README'ye bir tablo ve hatanın en büyük olduğu 3 örnek.

## 5. Takvim

| Gün | Tarih | İş | Bitti sayılır |
|---|---|---|---|
| 1 | 27 Eki | API spike: SAM 3.1 video takibi ve Muse Spark çağrısını gerçek bir golde dene; `api.py` + cache; 5–10 gol klibi seç | Bir klipte maskeler diske yazılıyor |
| 2 | 28 Eki | `track.py`, takım ayrımı, top interpolasyonu | `tracks.json` + üstüne maske çizilmiş video |
| 3 | 29 Eki | `calibrate.py`: başlangıç H (yollar c ve a), ORB zinciri | Şablon çizgileri klip boyunca sahaya oturuyor |
| 4 | 30 Eki | Kayma düzeltmesi + GSR indir + `gsr_eval.py` | **İlk metre hatası rakamı** |
| 5 | 31 Eki | `geometry.py` + `replay.py` | Doğru pas zinciri ve kuşbakışı video |
| 6 | 1 Kas | `read.py` | 3 klipte mantıklı cue'lar |
| 7 | 2 Kas | `explain.py` + doğrulayıcı | Uçtan uca açıklama |
| 8 | 3 Kas | `app.py`, deploy (Modal veya HF Spaces) | Yabancı biri link üzerinden kullanabiliyor |
| 9 | 4 Kas | Eval tablosu, README, demo videosu, **teslim (TR saatiyle 04:00'ten önce)** | Gönderildi |

**Kesme kuralı:** 4. günün sonunda kalibrasyon kabul edilemezse demo yalnızca yol c (4 tıklama) ile devam eder. Otomatik kalibrasyon "gelecek iş" olarak yazılır.

## 6. Riskler

| Risk | Önlem |
|---|---|
| SAM 3.1 API video takibi beklenenden farklı (frame limiti, fiyat) | 1. gün spike'ı; gerekirse frame'leri tek tek gönderip id'leri IoU ile eşle |
| Top kayboluyor | İnterpolasyon + işaretleme; açıklama o frame'lere dayanmaz |
| Kalibrasyon kayıyor | Düzeltme adımı + yol c yedeği + eval'de açıkça raporla |
| Muse Spark uyduruyor | Şemalı JSON, `visible` alanı, koddaki doğrulayıcı |
| API maliyeti/limit | Disk cache; geliştirmeyi kısa kliplerle yap |
| Model ID'leri değişiyor | Model ID'si tek bir config sabiti; canlı listeden kontrol et |

## 7. Açık sorular (1. gün çözülecek)
- SAM 3.1 tek çağrıda video takibi döndürüyor mu? Video uzunluğu ve frame limiti ne?
- Muse Spark'ın güncel model ID'si (1.2 / 1.3?) ne, ve video girdisinin limiti ne?
- Resmi kurallar: takım büyüklüğü ve teslim formatı (demo video süresi, repo zorunlu mu?).
