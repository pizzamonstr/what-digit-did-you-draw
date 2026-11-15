import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="What Digit Did You Draw?")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_classifier.keras")

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

model = load_model()

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
    brush_size = st.slider(
        "Brush Thickness",
        min_value=10,
        max_value=40,
        value=28,
        step=2
    )
    
    canvas_result = st_canvas(
        stroke_width=brush_size,
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

has_drawing = False

with right_col:
    if canvas_result.image_data is not None:
        processed_image, image_array, model_input = (
            preprocess_canvas_image(canvas_result.image_data)
        )

        if image_array.max() == 0:
            st.info("Draw a digit to see the model's prediction.")
        else:
            has_drawing = True

            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            st.metric("Predicted Digit", predicted_digit)
            st.metric("Confidence", f"{confidence:.1%}")

            chart_data = pd.DataFrame({
                "Digit": np.arange(10),
                "Probability": probabilities
            })

            probability_chart = (
                alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Digit:O",
                            title="Digit", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Probability:Q",
                            title="Probability", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0, 1]))
                )
            )

            st.write("**Probability for Each Digit**")
            st.altair_chart(probability_chart)

            top_three = np.argsort(probabilities)[::-1][:3]

            st.write("**Top 3 Predictions**")
            for rank, digit in enumerate(top_three, start=1):
                st.write(
                    f"{rank}. Digit {digit}: "
                    f"{probabilities[digit]:.1%}"
                )
            
            st.image(
                processed_image.resize((140, 140), Image.NEAREST),
                caption="What the Model Sees (28 x 28)"
            )

if has_drawing:
    with st.expander("View the 28 x 28 Pixel Values"):
        pixel_table = pd.DataFrame(np.round(image_array, 1))
        shaded_table = (
            pixel_table.style
            .background_gradient(cmap="gray", vmin=0, vmax=1)
            .format("{:.1f}")
            .set_table_styles([{
                    "selector": '',
                    "props": "margin: 0 auto;"
                },
                {
                    "selector": "td",
                    "props": (
                        "font-size: 9px; "
                        "padding: 2px; "
                        "text-align: center;"
                    )
                }
            ])
            .hide(axis="index")
            .hide(axis="columns")
        )
        st.html(shaded_table.to_html())
    