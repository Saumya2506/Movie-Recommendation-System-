"""Streamlit UI components for the three demo applications."""
from __future__ import annotations

from pathlib import Path
import hashlib
import io

import pandas as pd
import streamlit as st

from src.models import (
    MovieRecommender,
    classification_metrics,
    predict_new_records,
    top_feature_weights,
    train_classification_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _read_dataset(kind: str, sample_name: str, key: str) -> pd.DataFrame | None:
    source = st.radio(
        "Choose data source",
        ["Bundled demo dataset", "Upload your CSV"],
        horizontal=True,
        key=f"{key}_source",
    )
    if source == "Bundled demo dataset":
        df = pd.read_csv(DATA_DIR / sample_name)
        st.info("You are using a synthetic demo dataset. Upload a real dataset to train a model for your use case.")
        return df
    upload = st.file_uploader("Upload a CSV file", type=["csv"], key=f"{key}_upload")
    if upload is None:
        st.caption("Upload a CSV to continue, or switch back to the bundled demo data.")
        return None
    try:
        return pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return None


def _signature(df: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _show_classification_results(bundle, threshold: float, prefix: str, metric_focus: str) -> None:
    metrics = classification_metrics(bundle.y_test, bundle.probabilities, threshold)
    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 score", metric_focus],
        [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["pr_auc"] if metric_focus == "PR-AUC" else metrics["roc_auc"]],
    ):
        col.metric(label, f"{value:.3f}")

    st.caption(
        f"Positive class: **{bundle.positive_label}** · Negative class: **{bundle.negative_label}** · "
        f"Decision threshold: **{threshold:.2f}**"
    )
    st.markdown("**Confusion matrix**")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {bundle.negative_label}", f"Actual {bundle.positive_label}"],
        columns=[f"Predicted {bundle.negative_label}", f"Predicted {bundle.positive_label}"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Classification report**")
    report = pd.DataFrame(metrics["report"]).T
    st.dataframe(report.round(3), use_container_width=True)

    with st.expander("Top model features (indicative coefficient weights)"):
        st.caption("Larger absolute logistic-regression coefficients indicate stronger model influence after preprocessing; they do not prove causation.")
        importance = top_feature_weights(bundle)
        if importance.empty:
            st.write("Feature weights are unavailable for this dataset.")
        else:
            st.dataframe(importance.round(4), use_container_width=True, hide_index=True)


def render_churn() -> None:
    st.title("Customer Churn Prediction")
    st.write("Train a binary classification model to estimate which customers are more likely to leave.")
    df = _read_dataset("churn", "sample_churn.csv", "churn")
    if df is None:
        return
    st.caption(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(12), use_container_width=True)

    default_target = "Churn" if "Churn" in df.columns else df.columns[-1]
    target = st.selectbox("Target column (the column to predict)", list(df.columns), index=list(df.columns).index(default_target), key="churn_target")
    sig = _signature(df) + target
    state_key = "churn_model_bundle"
    if st.session_state.get(state_key + "_signature") != sig:
        st.session_state.pop(state_key, None)

    if st.button("Train / evaluate churn model", type="primary", key="churn_train"):
        try:
            with st.spinner("Training the churn model..."):
                bundle = train_classification_model(df, target, numeric_only=False)
            st.session_state[state_key] = bundle
            st.session_state[state_key + "_signature"] = sig
            st.success("Model trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    bundle = st.session_state.get(state_key)
    if bundle is None:
        st.info("Choose the target column, then click **Train / evaluate churn model**.")
        return

    st.subheader("Model evaluation")
    threshold = st.slider("Churn probability threshold", 0.10, 0.90, 0.50, 0.05, key="churn_threshold")
    _show_classification_results(bundle, threshold, "churn", "ROC-AUC")
    if bundle.dropped_columns:
        st.caption("Ignored identifier / unusable columns: " + ", ".join(map(str, bundle.dropped_columns[:12])))

    st.subheader("Predict churn for new customers")
    pred_file = st.file_uploader("Upload a CSV of customers without needing a Churn label", type=["csv"], key="churn_predict_upload")
    if pred_file is not None:
        try:
            new_data = pd.read_csv(pred_file)
            if st.button("Generate churn predictions", key="churn_predict_button"):
                output = predict_new_records(bundle, new_data, threshold)
                st.dataframe(output.head(30), use_container_width=True)
                st.download_button(
                    "Download churn predictions CSV",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    key="churn_download",
                )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")


def render_fraud() -> None:
    st.title("Credit Card Fraud Detection")
    st.write("Train an imbalanced binary classifier and tune the probability threshold to control fraud alerts.")
    df = _read_dataset("fraud", "sample_fraud.csv", "fraud")
    if df is None:
        return
    st.caption(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(12), use_container_width=True)

    default_target = "Class" if "Class" in df.columns else df.columns[-1]
    target = st.selectbox("Target column (1 = fraud, 0 = legitimate in the sample)", list(df.columns), index=list(df.columns).index(default_target), key="fraud_target")
    sig = _signature(df) + target
    state_key = "fraud_model_bundle"
    if st.session_state.get(state_key + "_signature") != sig:
        st.session_state.pop(state_key, None)

    if st.button("Train / evaluate fraud model", type="primary", key="fraud_train"):
        try:
            with st.spinner("Training the fraud detection model..."):
                bundle = train_classification_model(df, target, numeric_only=True)
            st.session_state[state_key] = bundle
            st.session_state[state_key + "_signature"] = sig
            st.success("Model trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    bundle = st.session_state.get(state_key)
    if bundle is None:
        st.info("Choose the target column, then click **Train / evaluate fraud model**.")
        return

    st.subheader("Model evaluation")
    st.warning("Fraud detection is highly imbalanced. Review precision, recall and PR-AUC—not accuracy alone. The threshold is a business trade-off between missed fraud and false alerts.")
    threshold = st.slider("Fraud alert probability threshold", 0.05, 0.95, 0.50, 0.05, key="fraud_threshold")
    _show_classification_results(bundle, threshold, "fraud", "PR-AUC")
    if bundle.dropped_columns:
        st.caption("Ignored non-numeric / unusable columns: " + ", ".join(map(str, bundle.dropped_columns[:12])))

    st.subheader("Score new transactions")
    pred_file = st.file_uploader("Upload a CSV of transactions (feature columns should match training data)", type=["csv"], key="fraud_predict_upload")
    if pred_file is not None:
        try:
            new_data = pd.read_csv(pred_file)
            if st.button("Score transactions", key="fraud_predict_button"):
                output = predict_new_records(bundle, new_data, threshold)
                st.dataframe(output.head(30), use_container_width=True)
                st.download_button(
                    "Download transaction scores CSV",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    key="fraud_download",
                )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")


def _map_movie_columns(raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(raw.columns)
    st.markdown("**Map your movie catalogue columns**")
    title_default = columns.index("title") if "title" in columns else 0
    title_col = st.selectbox("Movie title column", columns, index=title_default, key="movie_title_col")
    genre_options = ["(none)"] + columns
    genre_default = genre_options.index("genres") if "genres" in genre_options else 0
    genres_col = st.selectbox("Genres column", genre_options, index=genre_default, key="movie_genres_col")
    overview_options = ["(none)"] + columns
    overview_default = overview_options.index("overview") if "overview" in overview_options else 0
    overview_col = st.selectbox("Overview / description column", overview_options, index=overview_default, key="movie_overview_col")
    id_options = ["(create IDs)"] + columns
    id_default = id_options.index("movieId") if "movieId" in id_options else 0
    id_col = st.selectbox("Movie ID column", id_options, index=id_default, key="movie_id_col")

    result = pd.DataFrame()
    result["title"] = raw[title_col].astype(str)
    result["genres"] = raw[genres_col].fillna("").astype(str) if genres_col != "(none)" else ""
    result["overview"] = raw[overview_col].fillna("").astype(str) if overview_col != "(none)" else ""
    result["movieId"] = raw[id_col].astype(str) if id_col != "(create IDs)" else [str(i + 1) for i in range(len(raw))]
    return result


def render_movies() -> None:
    st.title("Movie Recommendation System")
    st.write("Find movies similar to a title using TF-IDF content similarity, with optional item-item collaborative filtering from ratings.")
    source = st.radio("Movie catalogue source", ["Bundled demo catalogue", "Upload your movie CSV"], horizontal=True, key="movie_source")
    if source == "Bundled demo catalogue":
        raw_movies = pd.read_csv(DATA_DIR / "sample_movies.csv")
        st.info("The included catalogue and ratings are synthetic / curated demo data. Replace them with your own catalogue for a real recommendation experience.")
        movie_data = raw_movies[["title", "genres", "overview", "movieId"]].copy()
    else:
        movie_file = st.file_uploader("Upload movie catalogue CSV", type=["csv"], key="movie_catalogue_upload")
        if movie_file is None:
            st.caption("Upload a movie CSV to continue.")
            return
        try:
            raw_movies = pd.read_csv(movie_file)
        except Exception as exc:
            st.error(f"Could not read the movie CSV: {exc}")
            return
        movie_data = _map_movie_columns(raw_movies)

    st.caption(f"Catalogue: {len(movie_data):,} movies")
    with st.expander("Preview movie catalogue", expanded=False):
        st.dataframe(movie_data.head(12), use_container_width=True)

    ratings_mode = st.radio("Ratings data", ["Use bundled demo ratings", "Upload ratings CSV", "Content-only (no ratings)"], horizontal=True, key="ratings_mode")
    ratings_data = None
    if ratings_mode == "Use bundled demo ratings" and source == "Bundled demo catalogue":
        ratings_data = pd.read_csv(DATA_DIR / "sample_ratings.csv")
    elif ratings_mode == "Use bundled demo ratings" and source != "Bundled demo catalogue":
        st.caption("Bundled ratings refer to the bundled movie IDs. Upload a matching ratings file or choose content-only.")
    elif ratings_mode == "Upload ratings CSV":
        ratings_file = st.file_uploader("Upload ratings CSV (needs userId, movieId, rating or mapped equivalents)", type=["csv"], key="ratings_upload")
        if ratings_file is not None:
            try:
                raw_ratings = pd.read_csv(ratings_file)
                cols = list(raw_ratings.columns)
                user_default = cols.index("userId") if "userId" in cols else 0
                movie_default = cols.index("movieId") if "movieId" in cols else min(1, len(cols)-1)
                rating_default = cols.index("rating") if "rating" in cols else min(2, len(cols)-1)
                c1, c2, c3 = st.columns(3)
                user_col = c1.selectbox("User ID", cols, index=user_default, key="ratings_user_col")
                rate_movie_col = c2.selectbox("Movie ID", cols, index=movie_default, key="ratings_movie_col")
                score_col = c3.selectbox("Rating", cols, index=rating_default, key="ratings_score_col")
                ratings_data = raw_ratings[[user_col, rate_movie_col, score_col]].rename(
                    columns={user_col: "userId", rate_movie_col: "movieId", score_col: "rating"}
                )
            except Exception as exc:
                st.error(f"Could not read ratings CSV: {exc}")
                return

    recommendation_signature = _signature(movie_data) + (_signature(ratings_data) if ratings_data is not None else "no-ratings")

    if st.button("Build recommendation model", type="primary", key="movie_build"):
        try:
            with st.spinner("Building the recommendation model..."):
                recommender = MovieRecommender().fit(movie_data, ratings_data)
            st.session_state["movie_recommender"] = recommender
            st.session_state["movie_recommender_signature"] = recommendation_signature
            st.session_state["movie_recommender_ratings_enabled"] = ratings_data is not None
            st.success("Recommendation model is ready.")
        except Exception as exc:
            st.error(f"Could not build the recommender: {exc}")

    recommender = st.session_state.get("movie_recommender")
    if recommender is None:
        st.info("Click **Build recommendation model** to create the recommender.")
        return
    # Rebuild indicator prevents accidentally applying a stale model after changing the catalogue.
    if st.session_state.get("movie_recommender_signature") != recommendation_signature:
        st.warning("The movie catalogue or ratings data changed. Rebuild the recommendation model before requesting results.")
        return

    st.subheader("Get recommendations")
    titles = recommender.movies["title"].dropna().astype(str).tolist()
    selected_title = st.selectbox("Choose a movie you like", titles, key="movie_seed_title")
    n_recs = st.slider("Number of recommendations", 3, 15, 8, key="movie_count")
    use_ratings = st.checkbox("Use ratings similarity when available", value=True, disabled=recommender.collaborative_similarity is None, key="movie_use_ratings")
    if st.button("Recommend movies", type="primary", key="movie_recommend"):
        try:
            result = recommender.recommend(selected_title, n=n_recs, use_ratings=use_ratings)
            st.session_state["movie_recommendations"] = result
            st.session_state["movie_recommendation_seed"] = selected_title
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("movie_recommendations")
    if result is not None and st.session_state.get("movie_recommendation_seed") == selected_title:
        st.markdown(f"**Recommendations based on: {selected_title}**")
        st.dataframe(result[["title", "genres", "similarity_score", "why_recommended"]], use_container_width=True, hide_index=True)
        with st.expander("Movie descriptions"):
            for _, row in result.iterrows():
                st.markdown(f"**{row['title']}**  \n{row['overview']}")
        st.download_button(
            "Download recommendations CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="movie_recommendations.csv",
            mime="text/csv",
            key="movie_recommend_download",
        )
