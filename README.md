# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

YazLab II – Proje II kapsamında geliştirilmiş, zaman serisi anomali tespitinde
**black-box derin öğrenme** modelleri ile **yorumlanabilir olasılıksal otomata**
yaklaşımını karşılaştıran bir çalışmadır.

> 📄 **Teslim raporu:** Detaylı deney sonuçları ve karşılaştırmalı analiz tabloları
> Word formatındaki `rapor.docx` dosyasındadır.

---

## 2. Proje Amacı

Zaman serisi verileri üzerinde iki farklı modelleme paradigmasını; yalnızca
performans açısından değil, aynı zamanda **genellenebilirlik, gürültüye dayanıklılık
ve açıklanabilirlik** kriterleri çerçevesinde karşılaştırmak:

- **Black-box modeller (LSTM, GRU):** Yüksek doğruluk potansiyeli, sınırlı yorumlanabilirlik.
- **Olasılıksal otomata:** Sembolik temsil ve durum geçişlerine dayalı, her kararı
  matematiksel olarak gerekçelendirebilen yorumlanabilir model.

Amaç tek bir "en iyi" modeli seçmek değil, model davranışlarını bilimsel ve sistematik
biçimde analiz etmektir.

---

## 3. Kullanılan Veri Setleri

| Veri Seti | Kapsam | Hedef Sütun |
|---|---|---|
| **SKAB** | Yalnızca `valve1` ve `valve2` klasörleri, tüm `.csv` dosyaları `concat` ile birleştirilir | `anomaly` |
| **BATADAL** | Yalnızca **Training Dataset 2** (`BATADAL_dataset04.csv`) | **`ATT_FLAG`** |

- SKAB birleştirmesinde `source_group` (valve1/valve2) ve `source_file` (kaynak CSV)
  sütunları **yalnızca takip ve grup-bazlı bölme** için eklenir; model girdisi değildir.
- BATADAL'da etiket dönüşümü: **`ATT_FLAG == 1` → anomali (1)**, `-999` dâhil diğer tüm
  değerler → normal (0).

---

## 4. Kurulum

Python **3.11** önerilir.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Gerekli başlıca paketler: `numpy`, `pandas`, `scikit-learn`, `scipy`,
`tensorflow`, `matplotlib`, `seaborn`, `networkx`.

---

## 5. Veri Klasör Yapısı

Proje kökünde aşağıdaki yapı beklenir:

```
.
├── main-yazlab-v1.py
├── BATADAL_dataset04.csv
└── SKAB_Dataset/
    ├── valve1/
    │   └── *.csv
    └── valve2/
        └── *.csv
```

> Not: İlgili veri dosyaları bulunamazsa kod, akışı doğrulamak için küçük sentetik
> veri üretir; gerçek sonuçlar için yukarıdaki dosyaların mevcut olması gerekir.

---

## 6. Çalıştırma

```bash
python main-yazlab-v1.py
```

Tüm deneyler (birim testler → veri yükleme → SKAB/BATADAL/gürültü/unseen/parametre
duyarlılık → istatistiksel test → görseller) tek komutla çalışır ve çıktılar
`outputs/` klasörüne yazılır.

---

## 7. Metodoloji

- **Eksik veri:** İleri doldurma (ffill) → geri doldurma (bfill) → sütun medyanı → 0.
  BATADAL'da `-999` sentinel değeri "ölçüm yok" kabul edilip eksik veri olarak işlenir.
- **Normalizasyon:** `MinMaxScaler`.
- **Boyut indirgeme:** Otomata yalnızca tek boyutlu veri ile çalıştığı için tüm
  özellikler **PCA ile tek bileşene (PC1)** indirgenir.
- **Girdi seçimi:** `datetime`, `changepoint`, `source_group`, `source_file` (SKAB) ve
  `DATETIME` (BATADAL) gibi zaman/kaynak sütunları model girdisine **dâhil edilmez**;
  yalnızca zaman sırasının korunması ve veri bölme için kullanılır.

---

## 8. Veri Sızıntısı (Leakage) Önlemleri

- **Scaler ve PCA yalnızca `train` verisinde fit edilir**; aynı dönüşüm validation/test'e uygulanır.
- **SKAB:** `source_file` grup değişkeni ile **GroupKFold** — aynı CSV hem eğitim hem
  test kümesinde aynı anda yer almaz. Kayan pencereler CSV sınırını aşmaz.
- **SKAB DL early stopping:** validation, **train fold içindeki `source_file` gruplarından**
  (~%20) ayrılır; early stopping bu validation'ın `val_loss`'unu izler. **Test fold'u early
  stopping'de kullanılmaz** — aynı CSV inner-train / validation / test arasında karışmaz.
- **BATADAL:** Zaman sırası korunarak **%60 eğitim / %20 doğrulama / %20 test** (shuffle yok).
- **Automata sözlüğü ve geçiş olasılıkları yalnızca `train`'den** öğrenilir.
- **Anomali eşiği** train/validation skor dağılımından kalibre edilir; **test yalnızca
  değerlendirme** için kullanılır.

---

## 9. Modeller

| Model | Durum |
|---|---|
| **LSTM** | Aktif |
| **GRU** | Aktif |
| **Probabilistic Automata** | Aktif |
| **1D-CNN** | Kodda parametrik olarak **mevcut**, final koşuda **süre maliyeti** nedeniyle kapalı (config ile yeniden aktive edilebilir) |

PDF §V-A "en az iki derin öğrenme modeli" şartı LSTM + GRU ile karşılanır. DL modelleri
sınıf dengesizliğine karşı `class_weight` ve `EarlyStopping(patience=5)` ile eğitilir.

**Sabit eğitim parametreleri:** `max_epochs=50`, `batch_size=32`,
`early_stopping_patience=5`, `random_seeds=[42, 123, 2026, 7, 999]`.

---

## 10. Probabilistic Automata

Otomata aşağıdaki dönüşümler üzerine kurulur:

1. **PAA (Piecewise Aggregate Approximation):** Pencere ortalama segmentlere indirgenir.
2. **SAX (Symbolic Aggregate approXimation):** Segmentler sembolik harflere dönüştürülür.
3. **Sliding Window:** Örüntü (pattern) dizileri çıkarılır.
4. **State = benzersiz pattern;** durumlar arası **geçiş olasılıkları** frekans tabanlı
   hesaplanır (Laplace / add-1 smoothing ile).
5. **Unseen mapping:** Test sırasında sözlükte olmayan pattern'lar için **Levenshtein
   (edit distance)** ile en yakın bilinen pattern bulunur ve sistem o state üzerinden devam eder.

Geçiş olasılığı: `P(Si → Sj) = Geçiş Sayısı / Toplam Çıkış Sayısı`
Dizi olasılığı: ardışık geçiş olasılıklarının çarpımı (path probability). Düşük olasılıklı
diziler anomali adayı olarak işaretlenir.

---

## 11. Açıklanabilirlik

Her karar `outputs/explainability_sample.json` içine aşağıdaki alanlarla yazılır:

| Alan | Anlamı |
|---|---|
| `time_step` | Zaman adımı |
| `state` | Mevcut (önceki) durum |
| `pattern` | Gözlemlenen örüntü |
| `status` | `seen` / `unseen` |
| `mapped_to` | Unseen ise Levenshtein ile eşlenen pattern |
| `transition_prob` | Geçişin olasılığı |
| `anomaly_score` | `-log(transition_prob)` (yüksek = düşük olasılık = anomali) |
| `path_probability` | Dizinin kümülatif olasılığı |
| `confidence` | Karara ait güven (ilgili geçiş olasılığı) |
| `decision` | `normal` / `anomaly` |

Açıklamalar deterministik ve yeniden üretilebilirdir.

---

## 12. Deney Senaryoları

- **Original:** Temiz veri üzerinde tüm modeller.
- **Gaussian noise:** Test verisine gürültü eklenerek dayanıklılık ölçümü (DL için **5 seed**, mean ± std).
- **Unseen data:** Eğitim SAX sözlüğünde bulunmayan pattern'ların yönetimi (deterministik otomata).
- **Parameter sensitivity:** `window_size` ve `alphabet_size` ∈ {3,4,5,6} taraması.
- **Wilcoxon:** Modeller arası F1 farkının istatistiksel anlamlılığı.

---

## 13. Sonuçlar

**Tablo 1 — Model Performansı (F1, ortalama ± standart sapma)**
*(SKAB: 3-fold × 5 seed = 15 ölçüm; BATADAL: 5 seed)*

| Model | SKAB F1 | BATADAL F1 |
|---|---|---|
| LSTM | 0.8015 ± 0.0588 | 0.0000 ± 0.0000 |
| GRU | 0.8064 ± 0.0622 | 0.0086 ± 0.0173 |
| 1D-CNN | N/A (final koşuda kapalı) | N/A |
| Automata | 0.1429 ± 0.0253 | 0.0685 ± 0.0000 |

> BATADAL'da DL modellerinin Accuracy değeri yüksek (~0.83–0.88) olmasına rağmen recall ≈ 0'dır;
> yani modeller çoğunluk (normal) sınıfını tahmin etmektedir (bkz. Bulgular ve Sınırlılıklar).

**Tablo 1b — SKAB 4-Metrik Özeti (3-fold × 5 seed, ortalama ± std)**

| Model | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| LSTM | 0.8015 ± 0.0588 | 0.8747 ± 0.0306 | 0.8913 ± 0.0580 | 0.7404 ± 0.1085 |
| GRU | 0.8064 ± 0.0622 | 0.8779 ± 0.0316 | 0.8932 ± 0.0495 | 0.7452 ± 0.1069 |
| Automata | 0.1429 ± 0.0253 | 0.6237 ± 0.0064 | 0.3555 ± 0.0205 | 0.0903 ± 0.0202 |

**Tablo 2 — Gürültü Etkisi (BATADAL, Gaussian noise, 5 seed: ortalama ± std)**

| Model | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| LSTM | 0.0142 ± 0.0098 | 0.7776 ± 0.0484 | 0.0124 ± 0.0086 | 0.0175 ± 0.0127 |
| GRU | 0.0133 ± 0.0100 | 0.7875 ± 0.0283 | 0.0124 ± 0.0087 | 0.0150 ± 0.0122 |
| Automata (deterministik) | 0.1103 | 0.8444 | 0.1231 | 0.1000 |

> DL modelleri için gürültü deneyi **5 random seed** ile çalıştırılır ve mean ± std raporlanır.
> Automata model+veri açısından deterministik olduğundan tek kez (gürültü çekimi sabit seed=42)
> üretilir; bu nedenle std verilmez.

**Tablo 2b — Unseen Veri Senaryosu (Automata, deterministik)**

| Metrik | Değer |
|---|---|
| F1 | 0.1165 |
| Precision | 0.0952 |
| Recall | 0.1500 |
| Unseen oranı | 1 / 829 pattern (~%0.1) |

> Unseen senaryosu yalnızca deterministik otomata üzerinde çalışır; seed etkisi yoktur.
> Tek unseen pattern Levenshtein (edit distance) ile en yakın train pattern'ına eşlenmiştir.

**Tablo 3 — Parametre Duyarlılık (F1, SKAB Fold-0)**

| Parametre (diğeri sabit) | Değer=3 | Değer=4 | Değer=5 | Değer=6 |
|---|---|---|---|---|
| Window Size (alphabet=3) | 0.0940 | 0.1363 | 0.3631 | 0.4792 |
| Alphabet Size (window=4) | 0.1363 | 0.2015 | 0.3835 | 0.4383 |

> State sayısı ws/al büyüdükçe hızla artar (ör. ws=6, al=6 → 8229 state, geçiş yoğunluğu ≈ %0.02).
> Tam grid `outputs/sensitivity_table.csv` içindedir.

**Tablo 4 — Çalışma Süreleri (BATADAL, ortalama)**

| Model | Eğitim (sn) | Çıkarım (sn) |
|---|---|---|
| LSTM | 6.195 | 0.2161 |
| GRU | 7.243 | 0.2334 |
| Automata | ~0.00 | ~0.00 |

**İstatistiksel test:** LSTM vs GRU (BATADAL F1, 5 seed) → Wilcoxon **p = 1.0** (anlamlı fark yok;
her iki modelin F1 değerleri sıfıra çok yakındır).

---

## 14. Grafikler

| Görsel | İçerik |
|---|---|
| `outputs/cm_roc_LSTM_BATADAL.png` | LSTM (BATADAL) Confusion Matrix + ROC eğrisi |
| `outputs/automata_BATADAL.png` | Otomata state diagram + geçiş olasılığı heatmap |
| `outputs/parameter_sensitivity.png` | Window/Alphabet size etkisinin F1 üzerindeki grafikleri |

---

## 15. Bulgular ve Tartışma

- **SKAB'de DL modelleri güçlü:** LSTM (F1 ≈ 0.80, Acc ≈ 0.87) ve GRU (F1 ≈ 0.81, Acc ≈ 0.88)
  başarılı; precision yüksek (~0.89), recall ~0.74 ile dengelidir. İki model birbirine benzer davranır.
- **BATADAL'da DL F1 düşük:** Sınıf dengesizliği (yaklaşık %5 anomali) nedeniyle modeller
  çoğunluk sınıfına çöker; yüksek Accuracy yanıltıcıdır, recall sıfıra yakındır.
- **Automata performanstan çok açıklanabilirlik sağlar:** F1 düşüktür ancak her karar
  state/geçiş/path-probability ile gerekçelendirilir ve eğitim/çıkarımı çok hızlıdır.
  Projenin tezi tam da bu black-box ↔ yorumlanabilirlik ödünleşimidir.
- **Parametre etkisi:** `window_size`/`alphabet_size` arttıkça hem state sayısı hem F1
  artar; ancak geçiş matrisi seyrekleşir (ezberleme/aşırı parçalanma riski).

---

## 16. Sınırlılıklar

- **BATADAL sınıf dengesizliği:** DL modelleri anomali sınıfını öğrenmekte zorlanır;
  eşik kalibrasyonu (PR-eğrisi) ile iyileştirme gelecek çalışmadır.
- **Gürültü senaryosu** PDF §IX-A uyarınca **5 random seed** ile çalıştırılır ve mean ± std
  raporlanır. **Unseen senaryosu** yalnızca deterministik otomata üzerinde çalıştığından
  seed etkisi yoktur ve tek koşu yeterlidir.
- **1D-CNN** kodda parametrik olarak mevcuttur ancak final koşuda süre maliyeti nedeniyle
  kapalıdır (config ile yeniden aktive edilebilir).
- **Cross-dataset analizi kapalıdır:** Zorunlu ister değildir ve farklı özellik uzayları
  arasında leakage riski taşıdığı için varsayılan olarak devre dışıdır
  (config bayrağı ile açılabilir).

---

## 17. Outputs Açıklaması

| Dosya | Açıklama |
|---|---|
| `outputs/experiment_log.json` | Tüm deneylerin parametre + metrik kayıtları (SKAB/BATADAL/gürültü/unseen/sensitivity/Wilcoxon) |
| `outputs/explainability_sample.json` | Otomata karar zincirinin örnek adım adım açıklama logu |
| `outputs/unseen_sample_log.json` | Unseen pattern örnekleri ve Levenshtein eşleme kararları |
| `outputs/sensitivity_table.csv` | Window/Alphabet size × F1 / state sayısı / yoğunluk tam tablosu |
| `outputs/cm_roc_LSTM_BATADAL.png` | LSTM (BATADAL) Confusion Matrix + ROC |
| `outputs/automata_BATADAL.png` | Otomata state diagram + geçiş heatmap |
| `outputs/parameter_sensitivity.png` | Parametre duyarlılık grafikleri |

---

## 18. Grup İçi Görev Dağılımı

- İlk kod iskeleti ve bazı modelleme/görselleştirme fonksiyonları ekip arkadaşı tarafından
  başlatılmıştır.
- PDF isterleri kontrolü, metodoloji doğrulama, kritik hata düzeltmeleri, ortam kurulumu,
  GitHub düzeni, test/run kontrolü, outputs doğrulama ve rapor/sunum hazırlığı tarafı ayrıca
  yürütülmüştür.

---

## 19. Not

Bu proje **YazLab II – Proje II** dersi kapsamında hazırlanmıştır.

---

### Hızlı Başlangıç

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main-yazlab-v1.py
```
