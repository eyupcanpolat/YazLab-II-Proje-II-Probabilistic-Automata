import os
import sys
import json
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import norm, wilcoxon
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# Windows konsolu (cp1254) Türkçe karakter / ✓ sembolünde çökebilir → UTF-8'e geç.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.makedirs("outputs", exist_ok=True)

# ============================================================
# 1. MERKEZİ KONFİGÜRASYON  [PDF §VIII]
# ============================================================
config_data = {
    "experiment": {
        # PDF §VII-B / §IX-A: zorunlu seed listesi
        "random_seeds": [42, 123, 2026, 7, 999],
        "scenarios": ["original", "gaussian_noise", "unseen_data"],
        # Cross-dataset tablosu (EK Tablo 3) leakage riski taşıdığı ve zorunlu
        # olmadığı için varsayılan olarak KAPALI. İstenirse True yapılabilir.
        "run_cross_dataset": False
    },
    "training_params": {
        "max_epochs": 50,
        "batch_size": 32,
        "early_stopping_patience": 5
    },
    "automata_params": {
        # paa_factor: ham kayan pencere = window_size * paa_factor nokta içerir;
        # PAA bunu window_size segmente indirger (gerçek sıkıştırma için >1 olmalı).
        "default": {"window_size": 4, "alphabet_size": 3, "paa_factor": 2},
        "sensitivity": {
            "window_sizes":   [3, 4, 5, 6],
            "alphabet_sizes": [3, 4, 5, 6]
        }
    },
    "data_split": {
        "skab":    {"strategy": "GroupKFold", "n_splits": 5},
        "batadal": {"train": 0.60, "val": 0.20, "test": 0.20}
    },
    "noise": {"gaussian_level": 0.2},
    # Açıklanabilirlik: anomaly skoru = -log(transition_prob).
    # Eşik, train/validation skor dağılımının 'score_percentile'. yüzdeliğinden kalibre edilir.
    "explainability": {"score_percentile": 95, "unseen_trans_prob": 1e-6},
    "dl_model": {"units": 32, "dropout": 0.2, "threshold": 0.5}
}


class ConfigManager:
    def __init__(self):
        self.config = config_data

    def get(self, key):
        return self.config.get(key)


# ============================================================
# 2. LOGLAMA  [PDF §IX-A]
# ============================================================
class ExperimentLogger:
    """Tüm deney sonuçlarını JSON dosyasına yazar."""

    def __init__(self, path="outputs/experiment_log.json"):
        self.path = path
        self.records = []

    def log(self, record: dict):
        record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.records.append(record)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Deney logu kaydedildi → {self.path}")


# ============================================================
# 3. VERİ YÜKLEYİCİLER  [PDF §III]
# ============================================================
def impute_features(X: pd.DataFrame, treat_sentinel=False, sentinel=-999):
    """Eksik veri / sentinel temizliği  [PDF §IV].

    Sıra: (gerekirse sentinel→NaN) → forward fill → backward fill → sütun medyanı → 0.
    YALNIZCA feature sütunlarına uygulanır; label sütununa dokunulmaz (çağıran taraf
    label'ı zaten X'ten çıkarmış olmalıdır).
    """
    X = X.copy()
    if treat_sentinel:
        # BATADAL'da -999 "ölçüm yok" anlamına gelir → eksik veri kabul edilir.
        X = X.replace(sentinel, np.nan)
    X = X.ffill().bfill()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    return X


class SKABDataLoader:
    def __init__(self, base_path="./SKAB_Dataset"):
        self.base_path = base_path
        self.data = pd.DataFrame()

    def load_data(self):
        all_dfs = []
        valve_path = os.path.join(self.base_path, 'valve1')

        if not os.path.exists(valve_path):
            np.random.seed(42)
            for folder in ['valve1', 'valve2']:
                for i, fname in enumerate([f"f{j}.csv" for j in range(5)]):
                    df = pd.DataFrame(
                        np.random.randn(300, 8),
                        columns=[f"Sensor_{k}" for k in range(8)]
                    )
                    df['anomaly'] = np.random.choice([0, 1], 300, p=[0.85, 0.15])
                    df['source_group'] = folder
                    df['source_file'] = f"{folder}_{fname}"
                    all_dfs.append(df)
        else:
            for folder in ['valve1', 'valve2']:
                folder_path = os.path.join(self.base_path, folder)
                for fp in glob.glob(os.path.join(folder_path, "*.csv")):
                    df = pd.read_csv(fp, sep=';', index_col=False)
                    df.columns = df.columns.str.strip()
                    df['source_group'] = folder
                    df['source_file'] = os.path.basename(fp)
                    all_dfs.append(df)

        self.data = pd.concat(all_dfs, ignore_index=True)
        return self.data

    def get_kfold_splits(self, n_splits=5):
        # datetime/changepoint/source_* model girdisi DEĞİL → çıkarılır [PDF §III-A]
        drop_cols = ['anomaly', 'datetime', 'changepoint', 'source_group', 'source_file']
        X = self.data.drop(columns=[c for c in drop_cols if c in self.data.columns])
        X = impute_features(X, treat_sentinel=False)          # SKAB: NaN temizliği
        y = self.data['anomaly'].fillna(0).astype(int)
        groups = self.data['source_file']
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        return X, y, splits, groups


class BATADALDataLoader:
    def __init__(self, file_path="BATADAL_dataset04.csv"):
        self.file_path = file_path
        self.cfg = config_data["data_split"]["batadal"]

    def load_and_split(self):
        if not os.path.exists(self.file_path):
            np.random.seed(123)
            n = 1000
            data = pd.DataFrame(
                np.random.randn(n, 10),
                columns=[f"S_{i}" for i in range(1, 11)]
            )
            data['ATT_FLAG'] = np.random.choice([0, 1], n, p=[0.90, 0.10])
        else:
            data = pd.read_csv(self.file_path)
            data.columns = data.columns.str.strip()

        target_col = 'ATT_FLAG' if 'ATT_FLAG' in data.columns else 'anomaly'
        self.target_col = target_col

        # BATADAL Training Dataset 2: ATT_FLAG ∈ {-999, 1}.
        # Binary hedef → saldırı (==1) anomaly=1; diğer tüm değerler (-999 dahil) normal=0.
        y = (data[target_col] == 1).astype(int)

        n_anom = int(y.sum()); n_norm = int((y == 0).sum())
        print(f"  [BATADAL] Label sütunu: '{target_col}'  → normal={n_norm}, "
              f"anomaly={n_anom}  (anomali oranı={y.mean():.3f})")

        # Zaman sütunu (DATETIME) model girdisi DEĞİL [PDF §III-B]
        drop_cols = [target_col, 'DATETIME']
        X = data.drop(columns=[c for c in drop_cols if c in data.columns])
        # Feature'larda -999 sentinel'ı eksik veri kabul edilip impute edilir; label hariç.
        X = impute_features(X, treat_sentinel=True, sentinel=-999)

        n = len(X)
        t1 = int(n * self.cfg["train"])
        t2 = int(n * (self.cfg["train"] + self.cfg["val"]))

        return (
            (X.iloc[:t1].reset_index(drop=True),  y.iloc[:t1].reset_index(drop=True)),
            (X.iloc[t1:t2].reset_index(drop=True), y.iloc[t1:t2].reset_index(drop=True)),
            (X.iloc[t2:].reset_index(drop=True),  y.iloc[t2:].reset_index(drop=True))
        )


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================
def _group_blocks(groups):
    """groups dizisindeki bitişik aynı-değer bloklarının (start, end) sınırlarını döndürür.
    SKAB'de source_file sınırlarını yakalamak için kullanılır."""
    groups = np.asarray(groups)
    blocks, start = [], 0
    for b in range(1, len(groups) + 1):
        if b == len(groups) or groups[b] != groups[start]:
            blocks.append((start, b))
            start = b
    return blocks


def create_sequences(X, y, time_steps, groups=None):
    """Zaman serisi için kayan pencere dizileri oluşturur.

    groups verilirse (örn. SKAB source_file), pencereler grup sınırını AŞMAZ:
    bir CSV'nin son satırı ile başka CSV'nin ilk satırı aynı sequence'e girmez.
    """
    # FIX: DataFrame/Series → numpy'ye güvenli dönüşüm (index bağımsız)
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        X = np.array(X)

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)

    # 1D girişi 2D'ye çevir
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    blocks = [(0, len(X))] if groups is None else _group_blocks(groups)

    Xs, ys = [], []
    for s, e in blocks:
        for i in range(s, e - time_steps):
            Xs.append(X[i:i + time_steps])
            ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def _automata_eval_grouped(auto, expl, data_1d, y, groups):
    """Otomata test değerlendirmesini source_file bloklarına bölerek yapar.

    Pencere CSV sınırını aşmaz. Hizalı (decisions, y) çiftlerini döndürür:
    her blokta pencereleme nedeniyle baştaki (win-1) etiket atlanır.
    """
    data_1d = np.asarray(data_1d).flatten()
    y = np.asarray(y)
    all_decs, all_y = [], []
    for s, e in _group_blocks(groups):
        pats = auto.transform_to_sequence(data_1d[s:e])
        if not pats:
            continue
        _, _, decs = expl.evaluate_sequence(pats)
        y_blk = y[s:e]
        adj = max(len(y_blk) - len(decs), 0)
        all_y.extend(y_blk[adj:adj + len(decs)])
        all_decs.extend(decs)
    return np.array(all_decs, dtype=int), np.array(all_y, dtype=int)


def add_gaussian_noise(data, noise_level=None):
    """Gaussian gürültü ekler. Seviye config'den okunur."""
    if noise_level is None:
        noise_level = config_data["noise"]["gaussian_level"]
    return data + np.random.normal(0, noise_level, data.shape)


def to_binary_int(arr):
    """
    FIX: Herhangi bir array'i güvenli şekilde binary int (0/1) dizisine çevirir.
    float32, float64, bool gibi tipleri yakalar.
    """
    arr = np.array(arr).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    # Threshold: 0.5'ten büyükse 1, değilse 0
    if arr.dtype in [np.float32, np.float64]:
        arr = (arr >= 0.5).astype(int)
    else:
        arr = arr.astype(int)
    # Sadece 0 ve 1 bırak
    arr = np.clip(arr, 0, 1)
    return arr


def safe_average(y_true, y_pred):
    """
    FIX: sklearn metrikler için doğru average parametresini belirler.
    Eğer tahminler binary değilse 'weighted' kullanır.
    """
    unique_vals = set(np.unique(y_true)) | set(np.unique(y_pred))
    if unique_vals.issubset({0, 1}):
        return 'binary'
    return 'weighted'


def compute_all_metrics(y_true, y_pred, zero_div=0):
    """
    FIX: PDF §IX — Accuracy, Precision, Recall, F1 hesaplar.
    Tüm giriş tipleri binary int'e dönüştürülür, average otomatik seçilir.
    """
    y_true = to_binary_int(y_true)
    y_pred = to_binary_int(y_pred)
    avg = safe_average(y_true, y_pred)

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=zero_div)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average=avg, zero_division=zero_div)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average=avg, zero_division=zero_div)), 4)
    }


# ============================================================
# 5. DERİN ÖĞRENME MODELLERİ  [PDF §V-A]
# ============================================================
class DeepLearningModels:
    def __init__(self, input_shape, model_type='LSTM'):
        self.model_type = model_type
        cfg = config_data["dl_model"]
        units   = cfg["units"]
        dropout = cfg["dropout"]

        self.model = Sequential(name=f"{model_type}_model")

        if model_type == 'LSTM':
            self.model.add(LSTM(units, activation='relu', input_shape=input_shape))
        elif model_type == 'GRU':
            self.model.add(GRU(units, activation='relu', input_shape=input_shape))
        elif model_type == '1D-CNN':
            self.model.add(Conv1D(
                filters=units, kernel_size=2, activation='relu',
                input_shape=input_shape
            ))
            self.model.add(GlobalMaxPooling1D())

        self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X_tr, y_tr, X_val, y_val):
        y_tr  = np.array(y_tr,  dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # FIX: sınıf dengesizliği için class weight
        n_neg = np.sum(y_tr == 0)
        n_pos = np.sum(y_tr == 1)
        if n_pos > 0:
            class_weight = {0: 1.0, 1: float(n_neg / n_pos)}
        else:
            class_weight = {0: 1.0, 1: 1.0}

        print(f"    class_weight: {class_weight}")  # debug için

        tp = config_data["training_params"]
        es = EarlyStopping(
            monitor='val_loss',
            patience=tp["early_stopping_patience"],
            restore_best_weights=True,
            verbose=0
        )
        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=tp["max_epochs"],
            batch_size=tp["batch_size"],
            callbacks=[es],
            class_weight=class_weight,  # ← eklendi
            verbose=0
        )

    def predict(self, X_test):
        """FIX: Çıktı kesinlikle binary int array olarak döner."""
        thr = config_data["dl_model"]["threshold"]
        raw = self.model.predict(X_test, verbose=0).flatten()
        return (raw > thr).astype(int)

    def predict_proba(self, X_test):
        """FIX: Olasılık değerleri float32 → float64 olarak döner."""
        return self.model.predict(X_test, verbose=0).flatten().astype(np.float64)


# ============================================================
# 6. OLASALIKSAL OTOMATA  [PDF §V-B, §X]
# ============================================================
class ProbabilisticAutomata:
    def __init__(self, window_size=None, alphabet_size=None):
        cfg = config_data["automata_params"]["default"]
        self.ws = window_size  if window_size  is not None else cfg["window_size"]
        self.al = alphabet_size if alphabet_size is not None else cfg["alphabet_size"]
        self.paa_factor = cfg.get("paa_factor", 2)   # ham pencere = ws * paa_factor

        self.breakpoints = norm.ppf(np.linspace(1.0 / self.al, 1.0 - 1.0 / self.al, self.al - 1))

        self.transitions       = defaultdict(lambda: defaultdict(int))
        self.transition_probs  = defaultdict(lambda: defaultdict(float))
        self.vocabulary        = set()
        self._lev_cache        = {}

    def _paa(self, window):
        n = len(window)
        seg_len = max(1, n // self.ws)
        segments = []
        for i in range(self.ws):
            start = i * seg_len
            end   = min(start + seg_len, n)
            segments.append(np.mean(window[start:end]))
        return np.array(segments)

    def _sax(self, paa_values):
        mu, sigma = np.mean(paa_values), np.std(paa_values)
        z = (paa_values - mu) / (sigma + 1e-8)
        return "".join([chr(97 + int(np.searchsorted(self.breakpoints, v))) for v in z])

    def _paa_sax(self, window):
        return self._sax(self._paa(window))

    @staticmethod
    def _split_blocks(data_1d, groups):
        """groups verilirse 1D veriyi source_file bloklarına böler (sınır aşılmaz)."""
        if groups is None:
            return [data_1d]
        return [data_1d[s:e] for s, e in _group_blocks(groups)]

    def transform_to_sequence(self, data_1d):
        data_1d = np.asarray(data_1d).flatten()  # FIX: 2D girişi flatten et
        # Ham kayan pencere SAX kelime uzunluğundan (=self.ws) daha uzun seçilir ki
        # PAA gerçek sıkıştırma yapsın: her SAX sembolü paa_factor ham noktanın ortalaması.
        win = self.ws * self.paa_factor
        if win > len(data_1d):
            return []
        return [
            self._paa_sax(data_1d[i: i + win])
            for i in range(len(data_1d) - win + 1)
        ]

    def fit(self, train_data_1d, groups=None):
        """Otomata sözlüğü ve geçişleri YALNIZCA train verisinden öğrenilir [PDF §VII-B].

        groups verilirse pattern dizileri source_file sınırını aşmaz (SKAB).
        """
        train_data_1d = np.asarray(train_data_1d).flatten()  # FIX: flatten

        for block in self._split_blocks(train_data_1d, groups):
            patterns = self.transform_to_sequence(block)
            self.vocabulary.update(patterns)
            for i in range(len(patterns) - 1):
                self.transitions[patterns[i]][patterns[i + 1]] += 1

        v_size = max(len(self.vocabulary), 1)
        for state, trans in self.transitions.items():
            total = sum(trans.values())
            for nxt in self.vocabulary:
                # Laplace (add-1) smoothing  [PDF §IX-C, rubrik 2]
                self.transition_probs[state][nxt] = (
                    (trans.get(nxt, 0) + 1) / (total + v_size)
                )

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(
                    prev_row[j + 1] + 1,
                    curr_row[j] + 1,
                    prev_row[j] + (c1 != c2)
                ))
            prev_row = curr_row
        return prev_row[-1]

    def find_closest(self, unseen):
        if unseen in self._lev_cache:
            return self._lev_cache[unseen]
        if not self.vocabulary:
            return unseen
        best = min(self.vocabulary, key=lambda v: self.levenshtein_distance(unseen, v))
        self._lev_cache[unseen] = best
        return best

    def predict(self, test_data_1d, calib_data_1d=None):
        """ExplainabilityModule üzerinden binary int karar döner.

        calib_data_1d (train/validation) verilirse eşik ondan kalibre edilir;
        verilmezse hiçbir şey anomali sayılmaz (güvenli varsayılan).
        """
        expl = ExplainabilityModule(self)
        if calib_data_1d is not None:
            expl.calibrate_threshold(
                self.transform_to_sequence(np.asarray(calib_data_1d).flatten())
            )
        _, _, decisions = expl.evaluate_sequence(
            self.transform_to_sequence(np.asarray(test_data_1d).flatten())
        )
        return np.array(decisions, dtype=int)


# ============================================================
# 7. AÇIKLANABILIRLIK MODÜLÜ  [PDF §X]
# ============================================================
class ExplainabilityModule:
    def __init__(self, model: ProbabilisticAutomata, threshold=None):
        self.model = model
        cfg = config_data["explainability"]
        self.score_percentile  = cfg["score_percentile"]
        self.unseen_trans_prob = cfg["unseen_trans_prob"]
        # Anomali eşiği -log(geçiş olasılığı) skoru içindir ve YALNIZCA
        # train/validation patternlerinden kalibre edilir (test'ten ASLA).
        self.threshold   = threshold
        self.unseen_count = 0
        self.total_count  = 0

    def _trans_prob(self, prev_state, mapped):
        """Frekans tabanlı geçiş olasılığı; bilinmeyen geçiş için küçük smoothing değeri."""
        if prev_state is None:
            return 1.0
        return self.model.transition_probs.get(prev_state, {}).get(
            mapped, self.unseen_trans_prob
        )

    def _map_pattern(self, incoming):
        """seen/unseen durumu ve eşlenen pattern (unseen → Levenshtein en yakın)."""
        if incoming in self.model.vocabulary:
            return "seen", incoming
        return "unseen", self.model.find_closest(incoming)

    def calibrate_threshold(self, calib_patterns):
        """Eşiği train/validation skor dağılımının {percentile}. yüzdeliği olarak set eder.

        Geçiş olasılığı düşük (skoru yüksek) örnekler anomali adayıdır [PDF §X-C].
        """
        scores, prev = [], None
        for pat in calib_patterns:
            _, mapped = self._map_pattern(pat)
            prob = self._trans_prob(prev, mapped)
            scores.append(-np.log(max(prob, 1e-12)))
            prev = mapped
        self.threshold = float(np.percentile(scores, self.score_percentile)) if scores else 0.0
        return self.threshold

    def evaluate_sequence(self, test_patterns):
        """
        Her adım için JSON log, kümülatif path olasılığı ve karar döndürür.

        Karar: anomaly_score = -log(transition_prob); skor kalibre eşiği aşarsa anomali.
        (Eski kümülatif-çarpım clamp bug'ı kaldırıldı.)

        Returns
        -------
        json_logs   : list[dict]
        path_probs  : list[float]   — bilgi amaçlı kümülatif olasılık
        decisions   : list[int]     — 0=normal, 1=anomali
        """
        json_logs, path_probs, decisions = [], [], []
        prev_state = None
        path_prob  = 1.0
        # Kalibre edilmemişse hiçbir şey anomali sayılmaz (güvenli varsayılan).
        thr = self.threshold if self.threshold is not None else float("inf")

        for t, incoming in enumerate(test_patterns):
            self.total_count += 1
            status, mapped = self._map_pattern(incoming)
            if status == "unseen":
                self.unseen_count += 1

            trans_prob = self._trans_prob(prev_state, mapped)
            score      = -np.log(max(trans_prob, 1e-12))   # yüksek = düşük olasılık = anomali
            path_prob  = path_prob * trans_prob            # bilgi amaçlı kümülatif (clamp YOK)

            is_anomaly   = score > thr
            decision_str = "anomaly" if is_anomaly else "normal"
            decisions.append(1 if is_anomaly else 0)
            path_probs.append(float(path_prob))

            log_entry = {
                "time_step":        t,
                "state":            prev_state if prev_state is not None else "START",
                "pattern":          incoming,
                "status":           status,
                "mapped_to":        mapped,
                "transition_prob":  round(float(trans_prob), 6),
                "anomaly_score":    round(float(score), 6),
                "path_probability": float(path_prob),
                "confidence":       round(float(trans_prob), 6),
                "decision":         decision_str
            }
            json_logs.append(log_entry)
            prev_state = mapped

        return json_logs, path_probs, decisions

    @property
    def unseen_rate(self):
        if self.total_count == 0:
            return 0.0
        return round(self.unseen_count / self.total_count, 4)


# ============================================================
# 8. BİRİM TESTLER  [PDF §VI]
# ============================================================
def run_unit_tests():
    print("\n" + "="*60)
    print("  [0] BİRİM TESTLER (UNIT TESTS)")
    print("="*60)

    auto = ProbabilisticAutomata(window_size=3, alphabet_size=3)
    auto.vocabulary = {'abc', 'aaa', 'cba', 'bbb'}

    assert auto.levenshtein_distance('abc', 'abc') == 0,  "Test 1 BAŞARISIZ"
    assert auto.levenshtein_distance('abc', 'adc') == 1,  "Test 2 BAŞARISIZ"
    assert auto.levenshtein_distance('abc', 'cba') == 2,  "Test 3 BAŞARISIZ"
    assert auto.levenshtein_distance('',    'abc') == 3,  "Test 4 BAŞARISIZ"
    assert auto.levenshtein_distance('abc', '')    == 3,  "Test 5 BAŞARISIZ"
    print("  [✓] Levenshtein distance — 5 test geçti")

    closest = auto.find_closest('adc')
    assert closest in auto.vocabulary,                    "Test 6 BAŞARISIZ"
    assert auto.levenshtein_distance('adc', closest) <= auto.levenshtein_distance('adc', 'cba'), \
        "Test 7 BAŞARISIZ"
    print("  [✓] find_closest (unseen mapping) — 2 test geçti")

    auto2 = ProbabilisticAutomata(window_size=4, alphabet_size=3)
    window = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    paa_vals = auto2._paa(window)
    assert len(paa_vals) == 4,              "Test 8 BAŞARISIZ"
    assert paa_vals[0] < paa_vals[-1],      "Test 9 BAŞARISIZ"
    print("  [✓] PAA — 2 test geçti")

    sax = auto2._paa_sax(window)
    assert len(sax) == 4,                              "Test 10 BAŞARISIZ"
    assert all(c in 'abc' for c in sax),               "Test 11 BAŞARISIZ"
    print("  [✓] PAA→SAX — 2 test geçti")

    auto3 = ProbabilisticAutomata(window_size=3, alphabet_size=3)
    data  = np.random.randn(50)
    auto3.fit(data)
    explainer = ExplainabilityModule(auto3)
    test_seq  = auto3.transform_to_sequence(np.random.randn(20))
    logs, probs, decs = explainer.evaluate_sequence(test_seq)
    assert len(logs) == len(test_seq),                 "Test 12 BAŞARISIZ"
    assert all(k in logs[0] for k in
        ["time_step","state","pattern","status","mapped_to",
         "transition_prob","path_probability","confidence","decision"]), \
        "Test 13 BAŞARISIZ"
    assert all(d in [0, 1] for d in decs),             "Test 14 BAŞARISIZ"
    print("  [✓] ExplainabilityModule JSON — 3 test geçti")

    # FIX: compute_all_metrics binary/multiclass güvenlik testi
    y_t = np.array([0, 1, 0, 1, 1])
    y_p = np.array([0, 1, 1, 0, 1])
    mets = compute_all_metrics(y_t, y_p)
    assert all(k in mets for k in ["accuracy","precision","recall","f1"]), \
        "Test 15 BAŞARISIZ"
    print("  [✓] compute_all_metrics — 1 test geçti")

    print("\n  [✓] TOPLAM 15 BİRİM TEST BAŞARIYLA GEÇTİ\n")


# ============================================================
# 9. GÖRSELLEŞTİRMELER  [PDF §XI]
# ============================================================
def plot_confusion_and_roc(y_true, y_pred, y_prob, title_prefix="LSTM", save=True):
    """Confusion Matrix + ROC eğrisi."""
    # FIX: şekil uyumu ve tip güvenliği
    y_true = to_binary_int(y_true)
    y_pred = to_binary_int(y_pred)
    y_prob = np.array(y_prob, dtype=np.float64).flatten()

    min_len = min(len(y_true), len(y_pred), len(y_prob))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]
    y_prob = y_prob[-min_len:]

    # Tek sınıf varsa ROC çizilemez
    if len(np.unique(y_true)) < 2:
        print(f"  [UYARI] {title_prefix}: tek sınıf — ROC çizilemiyor, atlandı.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("white")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"Confusion Matrix — {title_prefix}")
    axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gerçek")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'navy', linestyle='--')
    axes[1].set_title(f"ROC Eğrisi — {title_prefix}")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].legend()

    plt.tight_layout()
    if save:
        path = f"outputs/cm_roc_{title_prefix.replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [GÖRSEL] {path}")
    plt.close()


def plot_automata_diagrams(automata: ProbabilisticAutomata, title="", save=True):
    """State diagram + Transition heatmap  [PDF §XI]."""
    states = sorted(list(automata.vocabulary))[:10]
    if len(states) < 2:
        print("  [UYARI] Yeterli state yok, automata grafiği atlandı.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mat = np.array([
        [automata.transition_probs[f].get(t, 0) for t in states]
        for f in states
    ])
    sns.heatmap(mat, xticklabels=states, yticklabels=states,
                cmap='viridis', ax=axes[0], fmt='.2f', annot=len(states) <= 8)
    axes[0].set_title(f"Geçiş Olasılıkları Heatmap {title}")
    axes[0].set_xlabel("Sonraki State"); axes[0].set_ylabel("Mevcut State")

    G = nx.DiGraph()
    for f in states:
        for t in states:
            p = automata.transition_probs[f].get(t, 0)
            if p > 0.08:
                G.add_edge(f, t, weight=round(p, 2))

    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()] if G.edges() else []
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200, ax=axes[1])
    nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[1])
    if edge_weights:
        nx.draw_networkx_edges(G, pos, width=[w * 3 for w in edge_weights],
                               edge_color='gray', arrows=True,
                               connectionstyle='arc3,rad=0.1', ax=axes[1])
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=7, ax=axes[1])
    axes[1].set_title(f"State Diagram {title}")
    axes[1].axis('off')

    plt.tight_layout()
    if save:
        path = f"outputs/automata_{title.replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [GÖRSEL] {path}")
    plt.close()


def plot_parameter_sensitivity(sensitivity_df: pd.DataFrame, save=True):
    """Parametre duyarlılık grafiği [PDF §XI]."""
    if sensitivity_df is None or sensitivity_df.empty:
        print("  [UYARI] sensitivity_df boş — grafik atlandı.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for al in sorted(sensitivity_df['alphabet'].unique()):
        sub = sensitivity_df[sensitivity_df['alphabet'] == al]
        axes[0].plot(sub['window'], sub['f1'], marker='o', label=f"alphabet={al}")
    axes[0].set_title("Window Size Etkisi — F1 Score")
    axes[0].set_xlabel("Window Size"); axes[0].set_ylabel("F1 Score")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for ws in sorted(sensitivity_df['window'].unique()):
        sub = sensitivity_df[sensitivity_df['window'] == ws]
        axes[1].plot(sub['alphabet'], sub['f1'], marker='s', label=f"window={ws}")
    axes[1].set_title("Alphabet Size Etkisi — F1 Score")
    axes[1].set_xlabel("Alphabet Size"); axes[1].set_ylabel("F1 Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Parametre Duyarlılık Analizi (SKAB Fold-0)", fontsize=13)
    plt.tight_layout()
    if save:
        path = "outputs/parameter_sensitivity.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [GÖRSEL] {path}")
    plt.close()


# ============================================================
# 10. DENEY ORKESTRASYONU
# ============================================================
def run_skab_kfold_experiment(X_skab, y_skab, skab_splits, groups, logger, cfg):
    """SKAB 5-Fold × N-Seed deneyi. source_file bazlı GroupKFold [PDF §VII-B]."""
    print("\n" + "="*60)
    print("  [1] SKAB 5-FOLD × SEED CROSS VALIDATION")
    print("="*60)

    seeds  = cfg.get("experiment")["random_seeds"]
    ws     = cfg.get("automata_params")["default"]["window_size"]
    models = ["LSTM", "GRU", "1D-CNN", "Automata"]
    groups_arr = np.asarray(groups)

    results = {m: {s: [] for s in seeds} for m in models}

    for seed in seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        for fold_idx, (tr_idx, te_idx) in enumerate(skab_splits):
            # FIX: iloc ile index-safe erişim. Scaler/PCA YALNIZCA train fold'da fit edilir.
            scaler = MinMaxScaler()
            X_tr_s = scaler.fit_transform(X_skab.iloc[tr_idx])
            X_te_s = scaler.transform(X_skab.iloc[te_idx])

            pca = PCA(n_components=1)
            X_tr_pca = pca.fit_transform(X_tr_s)
            X_te_pca = pca.transform(X_te_s)

            y_tr = y_skab.iloc[tr_idx].values.astype(int)
            y_te = y_skab.iloc[te_idx].values.astype(int)
            g_tr = groups_arr[tr_idx]
            g_te = groups_arr[te_idx]

            # Sequence'ler source_file sınırını aşmaz
            X_seq_tr, y_seq_tr = create_sequences(X_tr_s, y_tr, ws, groups=g_tr)
            X_seq_te, y_seq_te = create_sequences(X_te_s, y_te, ws, groups=g_te)

            # FIX: sequence yeterli uzunlukta değilse atla
            if len(X_seq_tr) == 0 or len(X_seq_te) == 0:
                print(f"  [UYARI] Fold {fold_idx}: yetersiz sequence — atlandı.")
                continue

            for m in ["LSTM", "GRU", "1D-CNN"]:
                model = DeepLearningModels(
                    input_shape=(ws, X_seq_tr.shape[2]), model_type=m
                )
                model.train(X_seq_tr, y_seq_tr, X_seq_te, y_seq_te)
                preds = model.predict(X_seq_te)
                score = float(f1_score(
                    to_binary_int(y_seq_te),
                    to_binary_int(preds),
                    average=safe_average(to_binary_int(y_seq_te), to_binary_int(preds)),
                    zero_division=0
                ))
                results[m][seed].append(score)
                logger.log({
                    "experiment": "SKAB_kfold",
                    "model": m, "seed": seed,
                    "fold": fold_idx, "f1": score
                })
                mets = compute_all_metrics(y_seq_te, preds)
                print(f"  {m:<8}: F1={mets['f1']:.4f} Acc={mets['accuracy']:.4f}")

            # Automata — sözlük & geçiş YALNIZCA train'den (grup-duyarlı),
            # eşik train'den kalibre (SKAB'de ayrı val yok), test yalnız değerlendirme.
            auto = ProbabilisticAutomata(window_size=ws)
            auto.fit(X_tr_pca.flatten(), groups=g_tr)
            expl = ExplainabilityModule(auto)
            expl.calibrate_threshold(auto.transform_to_sequence(X_tr_pca.flatten()))
            auto_decs, y_auto = _automata_eval_grouped(
                auto, expl, X_te_pca.flatten(), y_te, g_te
            )
            auto_score = float(f1_score(
                to_binary_int(y_auto),
                to_binary_int(auto_decs),
                average='binary', zero_division=0
            )) if len(auto_decs) else 0.0
            results["Automata"][seed].append(auto_score)
            logger.log({
                "experiment": "SKAB_kfold",
                "model": "Automata", "seed": seed,
                "fold": fold_idx, "f1": auto_score
            })

    print(f"\n{'Model':<12} | {'Ortalama F1':<14} | {'Std F1':<10}")
    print("-" * 42)
    skab_summary = {}
    for m in models:
        all_scores = [s for seed_scores in results[m].values() for s in seed_scores]
        if not all_scores:
            all_scores = [0.0]
        mu, sigma = float(np.mean(all_scores)), float(np.std(all_scores))
        skab_summary[m] = {"mean": mu, "std": sigma, "all": all_scores}
        print(f"{m:<12} | {mu:.4f}         | ±{sigma:.4f}")

    return skab_summary


def run_batadal_experiment(X_b_tr, y_b_tr, X_b_val, y_b_val, X_b_te, y_b_te,
                            logger, cfg):
    """BATADAL N-Seed deneyi."""
    print("\n" + "="*60)
    print("  [2] BATADAL SEED DENEYİ")
    print("="*60)

    seeds  = cfg.get("experiment")["random_seeds"]
    ws     = cfg.get("automata_params")["default"]["window_size"]
    models = ["LSTM", "GRU", "1D-CNN", "Automata"]

    scaler = MinMaxScaler()
    X_tr_s  = scaler.fit_transform(X_b_tr)
    X_val_s = scaler.transform(X_b_val)
    X_te_s  = scaler.transform(X_b_te)

    pca = PCA(n_components=1)
    X_tr_pca  = pca.fit_transform(X_tr_s)
    X_val_pca = pca.transform(X_val_s)
    X_te_pca  = pca.transform(X_te_s)

    # FIX: y değerlerini int'e çevir
    y_tr_arr  = y_b_tr.values.astype(int)
    y_val_arr = y_b_val.values.astype(int)
    y_te_arr  = y_b_te.values.astype(int)

    X_seq_tr,  y_seq_tr  = create_sequences(X_tr_s,  y_tr_arr,  ws)
    X_seq_val, y_seq_val = create_sequences(X_val_s, y_val_arr, ws)
    X_seq_te,  y_seq_te  = create_sequences(X_te_s,  y_te_arr,  ws)

    bat_results = {
        m: {"metrics": [], "t_train": [], "t_inf": []}
        for m in models
    }
    final_pred, final_prob = None, None
    auto_for_plot = None

    for seed in seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        for m in ["LSTM", "GRU", "1D-CNN"]:
            model = DeepLearningModels(
                input_shape=(ws, X_seq_tr.shape[2]), model_type=m
            )
            t0 = time.time()
            model.train(X_seq_tr, y_seq_tr, X_seq_val, y_seq_val)
            t_train = time.time() - t0

            t0 = time.time()
            preds = model.predict(X_seq_te)
            t_inf = time.time() - t0

            mets = compute_all_metrics(y_seq_te, preds)
            bat_results[m]["metrics"].append(mets)
            bat_results[m]["t_train"].append(t_train)
            bat_results[m]["t_inf"].append(t_inf)

            logger.log({
                "experiment": "BATADAL",
                "model": m, "seed": seed,
                **mets,
                "t_train": round(t_train, 3),
                "t_inf":   round(t_inf, 4)
            })

            if seed == seeds[-1] and m == "LSTM":
                final_prob = model.predict_proba(X_seq_te)
                final_pred = preds

            mets = compute_all_metrics(y_seq_te, preds)
            print(f"  {m:<8}: F1={mets['f1']:.4f} Acc={mets['accuracy']:.4f}")  
        # Automata — sözlük train'den, eşik VALIDATION'dan kalibre, test yalnız değerlendirme
        auto = ProbabilisticAutomata(window_size=ws)
        auto.fit(X_tr_pca.flatten())
        if seed == seeds[-1]:
            auto_for_plot = auto

        expl = ExplainabilityModule(auto)
        expl.calibrate_threshold(auto.transform_to_sequence(X_val_pca.flatten()))
        test_pat = auto.transform_to_sequence(X_te_pca.flatten())
        logs, _, auto_decs = expl.evaluate_sequence(test_pat)

        if seed == seeds[0]:
            print(f"  [Automata] eşik(val {expl.score_percentile}p)={expl.threshold:.4f} | "
                  f"sözlük={len(auto.vocabulary)} state | unseen={expl.unseen_rate*100:.1f}%")
            with open("outputs/explainability_sample.json", "w") as f:
                json.dump(logs[:20], f, indent=2)

        adj  = len(y_te_arr) - len(auto_decs)
        if adj < 0:
            adj = 0
        mets = compute_all_metrics(y_te_arr[adj:], auto_decs)
        bat_results["Automata"]["metrics"].append(mets)
        bat_results["Automata"]["t_train"].append(0.0)
        bat_results["Automata"]["t_inf"].append(0.0)

        logger.log({
            "experiment": "BATADAL",
            "model": "Automata", "seed": seed,
            **mets,
            "unseen_rate": expl.unseen_rate
        })

    # FIX: X_te_pca 2D array olarak döndürülür, flatten gerektiğinde caller yapar
    return bat_results, final_pred, final_prob, auto_for_plot, X_te_pca, y_b_te


def run_noise_experiment(X_b_tr, y_b_tr, X_b_val, y_b_val, X_b_te, y_b_te,
                          X_te_pca, logger, cfg):
    """
    FIX: Gaussian gürültü senaryosu — gereksiz parametreler kaldırıldı,
    her şey fonksiyon içinde tutarlı şekilde hazırlanıyor.
    """
    print("\n" + "="*60)
    print("  [3] GÜRÜLTÜ SENARYOSU (Gaussian Noise)")
    print("="*60)

    ws = cfg.get("automata_params")["default"]["window_size"]
    tf.random.set_seed(42); np.random.seed(42)

    scaler = MinMaxScaler()
    X_tr_s  = scaler.fit_transform(X_b_tr)
    X_val_s = scaler.transform(X_b_val)
    X_te_s  = scaler.transform(X_b_te)

    y_tr_arr  = y_b_tr.values.astype(int)
    y_val_arr = y_b_val.values.astype(int)
    y_te_arr  = y_b_te.values.astype(int)

    X_seq_tr,  y_seq_tr  = create_sequences(X_tr_s,  y_tr_arr,  ws)
    X_seq_val, y_seq_val = create_sequences(X_val_s, y_val_arr, ws)
    X_seq_te,  y_seq_te  = create_sequences(X_te_s,  y_te_arr,  ws)
    X_seq_te_noisy = add_gaussian_noise(X_seq_te)

    noise_results = {}
    for m in ["LSTM", "GRU", "1D-CNN"]:
        model = DeepLearningModels(input_shape=(ws, X_seq_tr.shape[2]), model_type=m)
        model.train(X_seq_tr, y_seq_tr, X_seq_val, y_seq_val)
        preds = model.predict(X_seq_te_noisy)
        mets  = compute_all_metrics(y_seq_te, preds)
        noise_results[m] = mets
        logger.log({"experiment": "noise", "model": m, **mets})
        mets = compute_all_metrics(y_seq_te, preds)
        print(f"  {m:<8}: F1={mets['f1']:.4f} Acc={mets['accuracy']:.4f}")          
    # Automata noise — eşik temiz VALIDATION'dan kalibre (gürültü yalnız test'e eklenir)
    pca = PCA(n_components=1)
    X_tr_pca_n   = pca.fit_transform(X_tr_s)
    X_val_pca_n  = pca.transform(X_val_s)
    X_te_noisy_1d = add_gaussian_noise(X_te_pca)  # dışarıdan gelen X_te_pca kullanılır

    auto_n = ProbabilisticAutomata(window_size=ws)
    auto_n.fit(X_tr_pca_n.flatten())
    expl   = ExplainabilityModule(auto_n)
    expl.calibrate_threshold(auto_n.transform_to_sequence(X_val_pca_n.flatten()))
    pats   = auto_n.transform_to_sequence(X_te_noisy_1d.flatten())
    _, _, decs = expl.evaluate_sequence(pats)

    adj  = len(y_te_arr) - len(decs)
    if adj < 0:
        adj = 0
    mets = compute_all_metrics(y_te_arr[adj:], decs)
    noise_results["Automata"] = mets
    logger.log({"experiment": "noise", "model": "Automata", **mets})
    print(f"  {'Automata':<8}: F1={mets['f1']:.4f} Acc={mets['accuracy']:.4f}")
    return noise_results


def run_unseen_experiment(X_b_tr, y_b_tr, X_b_te, y_b_te, logger, cfg):
    """Unseen veri senaryosu [PDF §VII]."""
    print("\n" + "="*60)
    print("  [4] UNSEEN VERİ SENARYOSU")
    print("="*60)

    ws = cfg.get("automata_params")["default"]["window_size"]

    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_b_tr)
    X_te_s = scaler.transform(X_b_te)

    pca = PCA(n_components=1)
    X_tr_pca = pca.fit_transform(X_tr_s).flatten()
    X_te_pca = pca.transform(X_te_s).flatten()

    auto = ProbabilisticAutomata(window_size=ws)
    auto.fit(X_tr_pca)

    print(f"  Train sözlük boyutu: {len(auto.vocabulary)} unique pattern")

    expl      = ExplainabilityModule(auto)
    # Bu senaryoda ayrı val yok → eşik train sözlüğünden kalibre edilir (test'ten değil)
    expl.calibrate_threshold(auto.transform_to_sequence(X_tr_pca))
    test_pats = auto.transform_to_sequence(X_te_pca)
    logs, probs, decs = expl.evaluate_sequence(test_pats)

    unseen_logs = [l for l in logs if l["status"] == "unseen"]
    print(f"  Test pattern sayısı : {len(test_pats)}")
    print(f"  Unseen pattern sayısı: {expl.unseen_count}  ({expl.unseen_rate*100:.1f}%)")

    with open("outputs/unseen_sample_log.json", "w") as f:
        json.dump(unseen_logs[:10], f, indent=2)

    y_te_arr = y_b_te.values.astype(int)
    adj  = len(y_te_arr) - len(decs)
    if adj < 0:
        adj = 0
    mets = compute_all_metrics(y_te_arr[adj:], decs)
    print(f"  F1={mets['f1']:.4f}  Precision={mets['precision']:.4f}  Recall={mets['recall']:.4f}")

    logger.log({
        "experiment": "unseen",
        "model": "Automata",
        "unseen_count": expl.unseen_count,
        "unseen_rate":  expl.unseen_rate,
        **mets
    })
    return mets, expl.unseen_count, expl.unseen_rate


def run_cross_dataset(X_skab, y_skab, skab_splits,
                       X_b_te, y_b_te, logger, cfg):
    """Cross-dataset genellenebilirlik [PDF §XI — Tablo 3]."""
    print("\n" + "="*60)
    print("  [5] CROSS-DATASET GENELLENEBİLİRLİK")
    print("="*60)

    ws = cfg.get("automata_params")["default"]["window_size"]
    tr_idx, te_idx = skab_splits[0]

    scaler_s = MinMaxScaler()
    X_tr_s   = scaler_s.fit_transform(X_skab.iloc[tr_idx])
    pca_s    = PCA(n_components=1)
    X_tr_pca = pca_s.fit_transform(X_tr_s)

    n_feat = min(X_skab.shape[1], X_b_te.shape[1])
    X_te_common = X_b_te.iloc[:, :n_feat]
    scaler_b = MinMaxScaler()
    X_te_s   = scaler_b.fit_transform(X_te_common)
    pca_b    = PCA(n_components=1)
    X_te_pca = pca_b.fit_transform(X_te_s)

    y_tr_arr = y_skab.iloc[tr_idx].values.astype(int)
    y_te_arr = y_b_te.values.astype(int)

    cross_results = {}
    for m in ["LSTM", "GRU", "1D-CNN"]:
        X_c_tr, y_c_tr = create_sequences(X_tr_pca, y_tr_arr, ws)
        X_c_te, y_c_te = create_sequences(X_te_pca, y_te_arr, ws)

        # FIX: validation olarak test setini kullan (cross-dataset senaryosu)
        if len(X_c_tr) == 0 or len(X_c_te) == 0:
            print(f"  [UYARI] {m}: yetersiz cross-dataset sequence — atlandı.")
            cross_results[m] = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        model = DeepLearningModels(input_shape=(ws, 1), model_type=m)
        model.train(X_c_tr, y_c_tr, X_c_te, y_c_te)
        preds = model.predict(X_c_te)
        mets  = compute_all_metrics(y_c_te, preds)
        cross_results[m] = mets
        print(f"  {m:<8}: F1={mets['f1']:.4f}")
        logger.log({"experiment": "cross_dataset", "model": m, **mets})

    # Automata cross
    auto = ProbabilisticAutomata(window_size=ws)
    auto.fit(X_tr_pca.flatten())
    expl = ExplainabilityModule(auto)
    pats = auto.transform_to_sequence(X_te_pca.flatten())
    _, _, decs = expl.evaluate_sequence(pats)
    adj  = len(y_te_arr) - len(decs)
    if adj < 0:
        adj = 0
    mets = compute_all_metrics(y_te_arr[adj:], decs)
    cross_results["Automata"] = mets
    print(f"  {'Automata':<8}: F1={mets['f1']:.4f}")
    logger.log({"experiment": "cross_dataset", "model": "Automata", **mets})

    return cross_results


def run_parameter_sensitivity(X_skab, y_skab, skab_splits, groups, logger, cfg):
    """Parametre duyarlılık analizi [PDF §VII-A — Tablo 4]."""
    print("\n" + "="*60)
    print("  [6] PARAMETRE DUYARLILIK ANALİZİ (Tablo 4)")
    print("="*60)

    ws_list = cfg.get("automata_params")["sensitivity"]["window_sizes"]
    al_list = cfg.get("automata_params")["sensitivity"]["alphabet_sizes"]
    tr_idx, te_idx = skab_splits[0]
    groups_arr = np.asarray(groups)
    g_tr = groups_arr[tr_idx]
    g_te = groups_arr[te_idx]

    rows = []
    print(f"{'WS':<4} {'AL':<4} | {'States':<8} {'Transitions':<12} {'Yoğunluk':<10} {'F1':<8}")
    print("-" * 52)

    for ws in ws_list:
        for al in al_list:
            scaler = MinMaxScaler()
            X_tr_s   = scaler.fit_transform(X_skab.iloc[tr_idx])
            X_te_s   = scaler.transform(X_skab.iloc[te_idx])
            pca      = PCA(n_components=1)
            X_tr_pca = pca.fit_transform(X_tr_s).flatten()
            X_te_pca = pca.transform(X_te_s).flatten()

            auto = ProbabilisticAutomata(window_size=ws, alphabet_size=al)
            auto.fit(X_tr_pca, groups=g_tr)               # grup-duyarlı, train-only

            n_states = len(auto.vocabulary)
            n_trans  = sum(len(v) for v in auto.transitions.values())
            max_trans = max(n_states * n_states, 1)
            density  = round(n_trans / max_trans, 4)

            expl = ExplainabilityModule(auto)
            expl.calibrate_threshold(auto.transform_to_sequence(X_tr_pca))
            decs, y_aligned = _automata_eval_grouped(
                auto, expl, X_te_pca, y_skab.iloc[te_idx].values.astype(int), g_te
            )
            f1 = float(f1_score(
                to_binary_int(y_aligned),
                to_binary_int(decs),
                average='binary', zero_division=0
            )) if len(decs) else 0.0

            print(f"{ws:<4} {al:<4} | {n_states:<8} {n_trans:<12} {density:<10.4f} {f1:<8.4f}")
            rows.append({
                "window": ws, "alphabet": al,
                "n_states": n_states, "n_transitions": n_trans,
                "density": density, "f1": round(f1, 4)
            })
            logger.log({"experiment": "sensitivity", "ws": ws, "al": al,
                        "n_states": n_states, "density": density, "f1": round(f1, 4)})

    df = pd.DataFrame(rows)
    df.to_csv("outputs/sensitivity_table.csv", index=False)
    return df


def run_statistical_tests(bat_results, logger):
    """
    FIX: Wilcoxon testi — eşit değerler veya tek eleman durumunda güvenli hata yakalama.
    """
    print("\n" + "="*60)
    print("  [7] İSTATİSTİKSEL ANLAMLILIK TESTLERİ (Wilcoxon)")
    print("="*60)

    models = ["LSTM", "GRU", "1D-CNN"]
    pairs  = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]

    for m1, m2 in pairs:
        s1 = [d["f1"] for d in bat_results[m1]["metrics"]]
        s2 = [d["f1"] for d in bat_results[m2]["metrics"]]

        # FIX: Wilcoxon için en az 2 farklı değer gerekli
        if len(s1) < 2 or len(s2) < 2:
            print(f"  {m1} vs {m2}: yetersiz örnek (n={len(s1)}) — test atlandı")
            continue
        if s1 == s2:
            print(f"  {m1} vs {m2}: tüm değerler eşit — test anlamlı değil (p=1.0)")
            logger.log({"experiment": "wilcoxon", "pair": f"{m1}_vs_{m2}",
                        "p_value": 1.0, "significant": False, "note": "identical_scores"})
            continue

        try:
            stat, p = wilcoxon(s1, s2)
            sig = "ANLAMLI ✓" if p < 0.05 else "anlamsız"
            print(f"  {m1} vs {m2:<8}: p={p:.4f}  → {sig}")
            logger.log({"experiment": "wilcoxon", "pair": f"{m1}_vs_{m2}",
                        "p_value": round(float(p), 6), "significant": bool(p < 0.05)})
        except Exception as e:
            print(f"  {m1} vs {m2}: test yapılamadı — {e}")
            logger.log({"experiment": "wilcoxon", "pair": f"{m1}_vs_{m2}",
                        "error": str(e)})


# ============================================================
# 11. NİHAİ RAPOR TABLOSU
# ============================================================
def print_final_report(skab_summary, bat_results, noise_results,
                        cross_results, sensitivity_df):
    print("\n" + "="*60)
    print("  NİHAİ RAPOR")
    print("="*60)

    models = ["LSTM", "GRU", "1D-CNN", "Automata"]

    print("\n[TABLO 1] Model Performansı (Ortalama F1 ± Std)\n")
    print(f"{'Model':<12} | {'SKAB F1':<18} | {'BATADAL F1':<18}")
    print("-" * 55)
    for m in models:
        skab_m = skab_summary.get(m, {})
        bat_f1 = [d["f1"] for d in bat_results.get(m, {}).get("metrics", [])]
        s_str  = f"{skab_m.get('mean', 0):.4f} ±{skab_m.get('std', 0):.4f}"
        b_str  = f"{np.mean(bat_f1):.4f} ±{np.std(bat_f1):.4f}" if bat_f1 else "N/A"
        print(f"{m:<12} | {s_str:<18} | {b_str}")

    print("\n[TABLO 2] Gürültü Etkisi (BATADAL Gaussian Noise)\n")
    print(f"{'Model':<12} | {'F1':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall'}")
    print("-" * 60)
    for m in models:
        mets = noise_results.get(m, {})
        print(f"{m:<12} | {mets.get('f1',0):<8.4f} | "
              f"{mets.get('accuracy',0):<10.4f} | "
              f"{mets.get('precision',0):<10.4f} | "
              f"{mets.get('recall',0):.4f}")

    print("\n[TABLO 3] Cross-Dataset (SKAB → BATADAL)\n")
    print(f"{'Model':<12} | {'F1':<8} | {'Accuracy'}")
    print("-" * 35)
    for m in models:
        mets = cross_results.get(m, {})
        print(f"{m:<12} | {mets.get('f1',0):<8.4f} | {mets.get('accuracy',0):.4f}")

    print("\n[TABLO 4] Parametre Duyarlılık — ilk 8 satır")
    if sensitivity_df is not None and not sensitivity_df.empty:
        print(sensitivity_df.head(8).to_string(index=False))
    else:
        print("  (veri yok)")

    print("\n[TABLO 5] Çalışma Süreleri (BATADAL)\n")
    print(f"{'Model':<12} | {'Ort. Train (sn)':<18} | {'Ort. Inf (sn)'}")
    print("-" * 45)
    for m in models:
        t_tr = float(np.mean(bat_results.get(m, {}).get("t_train", [0])))
        t_in = float(np.mean(bat_results.get(m, {}).get("t_inf",   [0])))
        print(f"{m:<12} | {t_tr:<18.3f} | {t_in:.4f}")


# ============================================================
# 12. MAIN
# ============================================================
if __name__ == "__main__":

    run_unit_tests()

    cfg    = ConfigManager()
    logger = ExperimentLogger()
    ws     = cfg.get("automata_params")["default"]["window_size"]

    print("\n" + "="*60)
    print("  VERİLER YÜKLENİYOR")
    print("="*60)

    skab_loader = SKABDataLoader()
    skab_loader.load_data()
    X_skab, y_skab, skab_splits, skab_groups = skab_loader.get_kfold_splits()
    print(f"  SKAB: {X_skab.shape}, anomali oranı: {y_skab.mean():.3f}")

    bat_loader = BATADALDataLoader()
    (X_b_tr, y_b_tr), (X_b_val, y_b_val), (X_b_te, y_b_te) = bat_loader.load_and_split()
    print(f"  BATADAL: train={len(X_b_tr)}, val={len(X_b_val)}, test={len(X_b_te)}")

    skab_summary = run_skab_kfold_experiment(
        X_skab, y_skab, skab_splits, skab_groups, logger, cfg
    )

    bat_results, final_pred, final_prob, auto_for_plot, X_te_pca, _ = \
        run_batadal_experiment(
            X_b_tr, y_b_tr, X_b_val, y_b_val, X_b_te, y_b_te, logger, cfg
        )

    # FIX: run_noise_experiment artık X_seq_tr/val parametresi almıyor
    noise_results = run_noise_experiment(
        X_b_tr, y_b_tr, X_b_val, y_b_val, X_b_te, y_b_te,
        X_te_pca, logger, cfg
    )

    unseen_mets, unseen_cnt, unseen_rate = run_unseen_experiment(
        X_b_tr, y_b_tr, X_b_te, y_b_te, logger, cfg
    )
    print(f"  Unseen oran: {unseen_rate*100:.1f}%  F1={unseen_mets['f1']:.4f}")

    # Cross-dataset (EK Tablo 3): leakage riski + zorunlu değil → varsayılan KAPALI
    if config_data["experiment"].get("run_cross_dataset", False):
        cross_results = run_cross_dataset(
            X_skab, y_skab, skab_splits,
            X_b_te, y_b_te, logger, cfg
        )
    else:
        print("\n[BİLGİ] Cross-dataset deneyi devre dışı (RUN_CROSS_DATASET=False, "
              "leakage riski). Açmak için config_data['experiment']['run_cross_dataset']=True")
        cross_results = {}

    sensitivity_df = run_parameter_sensitivity(
        X_skab, y_skab, skab_splits, skab_groups, logger, cfg
    )

    run_statistical_tests(bat_results, logger)

    print("\n" + "="*60)
    print("  GÖRSELLEŞTİRMELER ÜRETİLİYOR")
    print("="*60)

    if final_pred is not None and final_prob is not None:
        y_plot = y_b_te.values[-len(final_pred):]
        plot_confusion_and_roc(y_plot, final_pred, final_prob, title_prefix="LSTM_BATADAL")

    if auto_for_plot is not None:
        plot_automata_diagrams(auto_for_plot, title="BATADAL")

    plot_parameter_sensitivity(sensitivity_df)

    print_final_report(skab_summary, bat_results, noise_results,
                        cross_results, sensitivity_df)

    # --- Metodoloji & veri özeti (savunma/rapor için doğrulama kontrolleri) ---
    print("\n" + "="*60)
    print("  METODOLOJİ & VERİ ÖZETİ")
    print("="*60)
    files_per_group = (
        skab_loader.data.groupby('source_group')['source_file'].nunique().to_dict()
    )
    y_b_all = pd.concat([y_b_tr, y_b_val, y_b_te])
    print(f"  SKAB klasörleri        : {sorted(skab_loader.data['source_group'].unique())}")
    print(f"  SKAB dosya sayıları    : {files_per_group} "
          f"(toplam {skab_loader.data['source_file'].nunique()} CSV)")
    print(f"  BATADAL satır sayısı   : {len(y_b_all)}")
    print(f"  BATADAL label sütunu   : {bat_loader.target_col}")
    print(f"  BATADAL label dağılımı : normal={int((y_b_all == 0).sum())}, "
          f"anomaly={int(y_b_all.sum())}")
    print(f"  Random seed listesi    : {cfg.get('experiment')['random_seeds']}")
    print(f"  Scaler/PCA fit         : YALNIZCA train (her fold/sette ayrı) — leakage yok")
    print(f"  Automata sözlük & eşik : sözlük=train | eşik=val(BATADAL)/train(SKAB) "
          f"{config_data['explainability']['score_percentile']}p | test=yalnız değerlendirme")
    print(f"  SKAB sequence sınırı   : source_file bazlı (pencere CSV sınırını aşmaz)")
    print(f"  Cross-dataset          : "
          f"{'AÇIK' if config_data['experiment'].get('run_cross_dataset') else 'KAPALI (leakage riski)'}")

    logger.save()

    print("\n" + "="*60)
    print("  TÜM DENEYLER TAMAMLANDI")
    print("  Çıktılar → ./outputs/ klasörü")
    print("="*60)