# =============================================================
#  preprocessor.py — Telemetry validation, normalization,
#  feature alignment and breach detection
# =============================================================

import numpy as np
import pandas as pd
from config import FEATURES, FEATURE_RANGES, FEATURE_THRESHOLDS, COLUMN_RENAME_MAP


class TelemetryPreprocessor:
    """
    Converts raw telemetry packets (dicts) into normalized numpy
    feature vectors ready for the Random Forest classifier.
    Also handles CSV column name variations from different hardware sources.
    """

    def __init__(self, scaler=None):
        # scaler: fitted sklearn StandardScaler from training
        self.scaler = scaler

    # ----------------------------------------------------------
    #  Column alignment (handles real-world CSV naming chaos)
    # ----------------------------------------------------------

    @staticmethod
    def align_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename and normalize DataFrame columns to match FEATURES.
        Safe to call even if columns are already correct.
        """
        # Normalize to lowercase + underscores first
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("%", "pct")
        )
        # Apply explicit rename map
        rename = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
        df = df.rename(columns=rename)
        # Add any missing feature columns as 0
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        return df

    @staticmethod
    def align_packet(packet: dict) -> dict:
        """Normalize a single telemetry dict to use FEATURES keys."""
        rename = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
        normalized = {}
        for k, v in packet.items():
            key = k.strip().lower().replace(" ", "_").replace("%", "pct")
            normalized[rename.get(key, key)] = v
        return normalized

    # ----------------------------------------------------------
    #  Validation
    # ----------------------------------------------------------

    def validate(self, packet: dict) -> tuple:
        """
        Check all 8 required features are present and in-range.
        Returns (is_valid: bool, errors: list[str]).
        """
        packet = self.align_packet(packet)
        errors = []
        for feat in FEATURES:
            if feat not in packet:
                errors.append(f"Missing field: {feat}")
                continue
            lo, hi = FEATURE_RANGES[feat]
            val = packet[feat]
            if not isinstance(val, (int, float)):
                errors.append(f"{feat} is not numeric: {val}")
            elif not (lo - abs(lo) * 0.1 <= val <= hi + abs(hi) * 0.1):
                errors.append(f"{feat}={val} out of range [{lo}, {hi}]")
        return len(errors) == 0, errors

    # ----------------------------------------------------------
    #  Vectorization
    # ----------------------------------------------------------

    def to_vector(self, packet: dict) -> np.ndarray:
        """Convert a single packet → (1, 8) scaled numpy array."""
        packet = self.align_packet(packet)
        raw = np.array([[packet[f] for f in FEATURES]], dtype=np.float32)
        if self.scaler is not None:
            return self.scaler.transform(raw)
        # Fallback: min-max normalization
        normed = np.zeros_like(raw)
        for i, feat in enumerate(FEATURES):
            lo, hi = FEATURE_RANGES[feat]
            normed[0, i] = np.clip((raw[0, i] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        return normed

    def batch_to_matrix(self, packets: list) -> tuple:
        """
        Convert list of packets → (N, 8) matrix.
        Invalid packets are silently dropped.
        Returns (matrix, valid_indices).
        """
        vectors, valid_idx = [], []
        for i, pkt in enumerate(packets):
            ok, _ = self.validate(pkt)
            if ok:
                vectors.append(self.to_vector(pkt)[0])
                valid_idx.append(i)
        if not vectors:
            return np.empty((0, len(FEATURES))), []
        return np.vstack(vectors), valid_idx

    # ----------------------------------------------------------
    #  Breach detection
    # ----------------------------------------------------------

    def breached_features(self, packet: dict) -> list:
        """
        Return names of features that breach their warning threshold.
        Used by the Plan stage to explain WHY a node was flagged.
        """
        packet = self.align_packet(packet)
        breached = []
        for feat, (op, thresh) in FEATURE_THRESHOLDS.items():
            val = packet.get(feat)
            if val is None:
                continue
            if op == "lt" and val < thresh:
                breached.append(feat)
            elif op == "gt" and val > thresh:
                breached.append(feat)
        return breached
