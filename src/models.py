"""Reusable model code for the Streamlit analytics suite."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class ClassificationBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    target_column: str
    negative_label: str
    positive_label: str
    X_test: pd.DataFrame
    y_test: np.ndarray
    probabilities: np.ndarray
    numeric_only: bool
    dropped_columns: list[str]


def _binary_target(values: pd.Series) -> tuple[pd.Series, str, str]:
    """Map a binary target to 0/1, using Yes/True/1/Fraud as the positive class."""
    clean = values.astype(str).str.strip()
    unique = list(pd.unique(clean.dropna()))
    if len(unique) != 2:
        raise ValueError(
            f"The selected target must have exactly 2 classes; found {len(unique)}: {unique[:8]}"
        )

    lower_to_original = {v.lower(): v for v in unique}
    positive_candidates = [
        "yes", "true", "1", "1.0", "churn", "fraud", "positive", "default", "active"
    ]
    positive = next((lower_to_original[k] for k in positive_candidates if k in lower_to_original), None)
    if positive is None:
        # For numeric labels, 1 is conventionally positive. Otherwise use the second class.
        try:
            numeric = sorted((float(v), v) for v in unique)
            positive = next((v for n, v in numeric if n == 1.0), numeric[-1][1])
        except (TypeError, ValueError):
            positive = sorted(unique)[-1]
    negative = next(v for v in unique if v != positive)
    mapped = clean.map({negative: 0, positive: 1}).astype(int)
    return mapped, negative, positive


def _prepare_features(
    frame: pd.DataFrame, target_column: str, numeric_only: bool
) -> tuple[pd.DataFrame, list[str]]:
    data = frame.drop(columns=[target_column]).copy()
    dropped: list[str] = []

    # Remove common identifier columns and columns that are effectively unique IDs.
    explicit_ids = {
        "customerid", "customer_id", "transactionid", "transaction_id", "recordid",
        "record_id", "id", "index", "unnamed: 0",
    }
    for col in list(data.columns):
        normalized = str(col).strip().lower()
        non_null = data[col].notna().sum()
        unique_ratio = data[col].nunique(dropna=True) / max(non_null, 1)
        if normalized in explicit_ids or (data[col].dtype == "object" and non_null > 30 and unique_ratio > 0.95):
            dropped.append(col)
            data = data.drop(columns=[col])

    # Parse object columns that are mostly numeric (e.g. Amount / TotalCharges with blanks).
    for col in list(data.columns):
        if data[col].dtype == "object":
            parsed = pd.to_numeric(data[col], errors="coerce")
            non_empty = data[col].notna().sum()
            if non_empty and parsed.notna().sum() / non_empty >= 0.85:
                data[col] = parsed

    if numeric_only:
        non_numeric = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
        dropped.extend(non_numeric)
        data = data.drop(columns=non_numeric)

    # Drop columns with no usable values or no variation.
    for col in list(data.columns):
        if data[col].notna().sum() == 0 or data[col].nunique(dropna=True) <= 1:
            dropped.append(col)
            data = data.drop(columns=[col])

    if data.empty:
        raise ValueError("No usable feature columns remain after excluding IDs and non-numeric columns.")
    return data, dropped


def _build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_features = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical_features = [c for c in features.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_features))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1500, solver="liblinear", random_state=42
    )
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def train_classification_model(
    frame: pd.DataFrame,
    target_column: str,
    *,
    numeric_only: bool = False,
    test_size: float = 0.25,
) -> ClassificationBundle:
    """Train and evaluate a binary classifier with preprocessing inside the pipeline."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    usable = frame.dropna(subset=[target_column]).copy()
    y, negative_label, positive_label = _binary_target(usable[target_column])
    if y.value_counts().min() < 4:
        raise ValueError("Each class needs at least 4 records for a reliable stratified train/test split.")

    X, dropped = _prepare_features(usable, target_column, numeric_only)
    if len(X) != len(y):
        raise ValueError("Feature and target rows do not match after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipeline = _build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return ClassificationBundle(
        pipeline=pipeline,
        feature_columns=list(X.columns),
        target_column=target_column,
        negative_label=negative_label,
        positive_label=positive_label,
        X_test=X_test,
        y_test=y_test.to_numpy(),
        probabilities=probabilities,
        numeric_only=numeric_only,
        dropped_columns=dropped,
    )


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Not positive", "Positive"],
            output_dict=True,
            zero_division=0,
        ),
    }
    result["roc_auc"] = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) == 2 else float("nan")
    result["pr_auc"] = average_precision_score(y_true, probabilities)
    return result


def predict_new_records(bundle: ClassificationBundle, new_frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Predict uploaded records, filling absent training fields with null and ignoring extras."""
    X = new_frame.copy()
    if bundle.target_column in X.columns:
        X = X.drop(columns=[bundle.target_column])
    X = X.drop(columns=[c for c in X.columns if c not in bundle.feature_columns], errors="ignore")
    missing = [c for c in bundle.feature_columns if c not in X.columns]
    for col in missing:
        X[col] = np.nan
    X = X[bundle.feature_columns]
    probabilities = bundle.pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    result = new_frame.copy()
    result["positive_probability"] = probabilities.round(6)
    result["predicted_class"] = np.where(predictions == 1, bundle.positive_label, bundle.negative_label)
    return result


def top_feature_weights(bundle: ClassificationBundle, limit: int = 15) -> pd.DataFrame:
    """Return absolute logistic-regression coefficient magnitudes for an indicative ranking."""
    try:
        names = bundle.pipeline.named_steps["preprocess"].get_feature_names_out()
        coefficients = bundle.pipeline.named_steps["classifier"].coef_[0]
        table = pd.DataFrame({"feature": names, "absolute_weight": np.abs(coefficients)})
        table["feature"] = table["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
        return table.sort_values("absolute_weight", ascending=False).head(limit).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "absolute_weight"])


class MovieRecommender:
    """Content-based movie recommender with optional item-item collaborative similarity."""

    def __init__(self) -> None:
        self.movies: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.content_similarity: np.ndarray | None = None
        self.collaborative_similarity: pd.DataFrame | None = None
        self.movie_id_column = "movieId"

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame | None = None) -> "MovieRecommender":
        required = {"title", "genres", "overview"}
        if not required.issubset(movies.columns):
            raise ValueError("Movie data must contain title, genres and overview columns after mapping.")
        data = movies.copy().reset_index(drop=True)
        data["title"] = data["title"].fillna("").astype(str)
        data["genres"] = data["genres"].fillna("").astype(str)
        data["overview"] = data["overview"].fillna("").astype(str)
        if "movieId" not in data.columns:
            data.insert(0, "movieId", np.arange(1, len(data) + 1))
        data["movieId"] = data["movieId"].astype(str)
        if data["title"].str.strip().eq("").all():
            raise ValueError("The movie title column is empty.")

        text_corpus = (
            data["title"] + " " + data["genres"].str.replace("|", " ", regex=False) + " " + data["overview"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = self.vectorizer.fit_transform(text_corpus)
        self.content_similarity = cosine_similarity(matrix, dense_output=True)
        self.movies = data
        self.collaborative_similarity = None

        if ratings is not None and {"userId", "movieId", "rating"}.issubset(ratings.columns):
            r = ratings[["userId", "movieId", "rating"]].copy()
            r["movieId"] = r["movieId"].astype(str)
            r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
            r = r.dropna(subset=["userId", "movieId", "rating"])
            r = r[r["movieId"].isin(data["movieId"])]
            if not r.empty:
                pivot = r.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
                # Centre each user's ratings so similarity reflects relative preferences.
                centered = pivot.sub(pivot.mean(axis=1), axis=0).fillna(0)
                sim = cosine_similarity(centered.T)
                self.collaborative_similarity = pd.DataFrame(sim, index=pivot.columns, columns=pivot.columns)
        return self

    def recommend(self, title: str, n: int = 8, use_ratings: bool = True) -> pd.DataFrame:
        if self.movies is None or self.content_similarity is None:
            raise RuntimeError("Fit the recommender before requesting recommendations.")
        matches = self.movies.index[self.movies["title"].str.casefold() == title.casefold()].tolist()
        if not matches:
            raise ValueError(f"Movie '{title}' was not found in the catalogue.")
        seed_idx = matches[0]
        seed_id = self.movies.loc[seed_idx, "movieId"]
        scores = self.content_similarity[seed_idx].copy()

        has_collab = (
            use_ratings
            and self.collaborative_similarity is not None
            and seed_id in self.collaborative_similarity.index
        )
        if has_collab:
            collab = np.zeros(len(self.movies), dtype=float)
            for idx, movie_id in enumerate(self.movies["movieId"]):
                if movie_id in self.collaborative_similarity.columns:
                    collab[idx] = self.collaborative_similarity.loc[seed_id, movie_id]
            scores = 0.7 * scores + 0.3 * collab

        scores[seed_idx] = -1
        top_idx = np.argsort(scores)[::-1][:n]
        result = self.movies.iloc[top_idx][["movieId", "title", "genres", "overview"]].copy()
        result["similarity_score"] = scores[top_idx].round(3)
        seed_genres = set(self.movies.loc[seed_idx, "genres"].split("|"))
        reasons = []
        for genres in result["genres"]:
            overlap = seed_genres & set(str(genres).split("|"))
            reasons.append("Shared genres: " + ", ".join(sorted(overlap)) if overlap else "Similar plot or themes")
        result["why_recommended"] = reasons
        return result.reset_index(drop=True)
