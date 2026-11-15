import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="What Digit Did You Draw?")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

def preprocess_canvas_image(canvas_image):
    image = Image.fromarray(canvas_image.astype("uint8"))
    image = image.convert("L")

    bounding_box = image.getbbox()
    if bounding_box is not None:
        image = image.crop(bounding_box)

    width, height = image.size
    scale = 20 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height))

    centered_image = Image.new("L", (28, 28), 0)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    centered_image.paste(image, (left, top))
    image = centered_image

    image_array = np.array(image) / 255.0
    model_input = image_array.reshape(1, 28, 28, 1)

    return image, image_array, model_input


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

with right_col:
    if canvas_result.image_data is not None:
        processed_image, image_array, model_input = (
            preprocess_canvas_image(canvas_result.image_data)
        )

        if image_array.max() == 0:
            st.info("Draw a digit to see the model's prediction.")
        else:
            st.image(
                processed_image.resize((140, 140), Image.NEAREST),
                caption="What the Model Sees (28 x 28)"
            )

            st.write("Shape:", model_input.shape)
            st.write("Min:", image_array.min())
            st.write("Max:", image_array.max())