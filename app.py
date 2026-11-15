import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="What Digit Did You Draw?")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

st.title("What Digit Did You Draw?")
st.subheader("Handwritten Digit Classifier")

st.write(
    "Draw a single digit from 0 to 9 in the box below. "
    "A neural network trained on the MNIST dataset will "
    "guess which digit you drew and show how confident "
    "it is in its guess."
)

left_col, right_col = st.columns(2)

with left_col:
    canvas_result = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )

    if st.button("Clear Drawing"):
        st.session_state.canvas_key += 1
        st.rerun()