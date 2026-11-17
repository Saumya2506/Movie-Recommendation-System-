import streamlit as st
from src.ui import render_movies
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
render_movies()
