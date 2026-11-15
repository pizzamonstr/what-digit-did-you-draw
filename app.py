import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="What Digit Did You Draw?")

st.markdown(
    "<style>"
    "[role='tab'] p {font-size: 1.25rem !important;}"
    "</style>",
    unsafe_allow_html=True
)

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_classifier.keras")


def clear_canvas():
    st.session_state.canvas_key += 1


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


@st.cache_data
def get_confusion_matrix():
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    test_probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(test_probabilities, axis=1)

    return confusion_matrix(y_test, y_pred)


model = load_model()

st.title("What Digit Did You Draw?")
st.subheader("Handwritten Digit Classifier")

draw_tab, how_tab, tips_tab, about_tab, pixel_tab = st.tabs([
    "**Classifier**",
    "**How It Works**",
    "**Tips**",
    "**About the Model**",
    "**Pixel Values**"
])

has_drawing = False

with draw_tab:
    st.write(
        "Draw a single digit from 0 to 9 in the box below. "
        "A neural network trained on the MNIST dataset will "
        "guess which digit you drew and show how confident "
        "it is in its guess."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Your Drawing**")
        
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

        st.button("Clear Drawing", on_click=clear_canvas)

        if canvas_result.image_data is not None:
            processed_image, image_array, model_input = (
                preprocess_canvas_image(canvas_result.image_data)
            )

            if image_array.max() > 0:
                has_drawing = True
                st.markdown("**What the Model Sees (28 x 28)**")
                st.image(
                    processed_image.resize((140, 140), Image.NEAREST)
                )

    with right_col:
        st.markdown("**Model Prediction**")

        if not has_drawing:
            st.info("Draw a digit to see the model's prediction.")
        else:
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))

            digit_col, confidence_col = st.columns(2)
            with digit_col:
                st.metric("Predicted Digit", predicted_digit)
            with confidence_col:
                st.metric("Confidence", f"{confidence:.1%}")

            top_three = np.argsort(probabilities)[::-1][:3]

            st.write("**Top 3 Predictions**")
            guess_labels = ["1st Guess", "2nd Guess", "3rd Guess"]
            top_cols = st.columns(3)

            for i in range(3):
                digit = int(top_three[i])
                with top_cols[i]:
                    with st.container(border=True):
                        st.metric(guess_labels[i], digit)
                        st.caption(f"{probabilities[digit]:.1%}")

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
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition(alt.datum.Digit == predicted_digit,
                                        alt.value("#FF4B4B"),
                                        alt.value("#83C9FF"))
                )
            )

            st.write("**Probability for Each Digit**")
            st.altair_chart(probability_chart)

with how_tab:
    st.markdown("**1. You Draw a Digit**")
    st.write("When you draw in the box, the canvas saves your "
             "drawing as a 280 x 280 pixel image, which is ten "
             "times larger than the images the model was trained on.")

    st.markdown("**2. The Image Is Cleaned Up**")
    st.write("Before the model sees your drawing, it's prepared to  "
        "match the MNIST training images. First, it's converted "
        "to grayscale and the empty space around the digit is "
        "cropped away. Next, the digit is shrunk to fit inside "
        "a 20 x 20 box and centered in a 28 x 28 image. Finally, "
        "each pixel is divided by 255, so every value falls "
        "between 0 (black) and 1 (white).")

    st.markdown("**3. Convolution Layers Find Patterns**")
    st.write("The model then slides small 3 x 3 filters across the "
             "image, one pixel at a time. Each filter has learned to "
             "look for a particular pattern, such as an edge or a "
             "curve, and responds strongly wherever that pattern "
             "appears. This model has two convolutional layers: the "
             "first uses 16 filters to find simple shapes, and the "
             "second uses 32 filters to combine them into more "
             "complex ones.")

    st.caption("Shape: 28 x 28 x 1 becomes 26 x 26 x 16, then "
               "11 x 11 x 32 after the second convolution layer.")

    st.markdown("**4. Pooling Layers Shrink the Image**")
    st.write("After each convolution layer, a pooling layer looks at "
             "every 2 x 2 area and keeps only its strongest value. "
             "This halves the size of the image while holding on to "
             "the most important signals, and it also makes the "
             "model less sensitive to exactly where each stroke "
             "was drawn.")

    st.caption("Shape: 26 x 26 x 16 becomes 13 x 13 x 16, and "
               "11 x 11 x 32 becomes 5 x 5 x 32.")

    st.markdown("**5. Dense Layers Combine the Patterns**")
    st.write("Once the patterns have been found, they're flattened "
             "into a single list of 800 numbers. A layer of 64 "
             "neurons then weighs all of these together to work out "
             "which digit the patterns most likely match.")

    st.caption("Shape: 5 x 5 x 32 is flattened into 800 numbers, "
               "then combined into 64.")

    st.markdown("**6. Softmax Turns Scores Into Probabilities**")
    st.write("The final layer has 10 neurons, one for each digit "
             "from 0 to 9, and each one produces a score. Softmax "
             "converts these scores into probabilities that add up "
             "to 100%, so they can be compared directly.")

    st.caption("Shape: 64 numbers become 10 probabilities.")

    st.markdown("**7. The Model Makes Its Prediction**")
    st.write("The digit with the highest probability becomes the "
             "model's prediction, and that probability is shown as "
             "the confidence score.")

with tips_tab:
    st.markdown("**Tips for Best Results**")
    st.write("For the most accurate predictions, try to draw the way "
             "the MNIST digits were written:")

    st.markdown("- Draw one digit at a time, rather than a whole "
                "number.\n"
                "- Draw it large, so it fills most of the box.\n"
                "- Keep loops open and clear, like the top of a 4 or "
                "the middle of an 8, since small loops can fill in "
                "when the image is shrunk.\n"
                "- If a prediction looks wrong, try adjusting the "
                "Brush Thickness slider and drawing the digit again.")

with about_tab:
    st.markdown("**About the Model**")
    st.write("This app uses a convolutional neural network (CNN) "
             "that I built and trained on the MNIST dataset of "
             "handwritten digits. The model learned from 54,000 "
             "training images, while another 6,000 were held "
             "back to check its progress during training. When "
             "it was tested on 10,000 images it had never seen "
             "before, it classified 98.4% of them correctly.")

    st.markdown("**Confusion Matrix**")
    st.write("Each row is the real digit, and each column is the "
             "model's guess. The diagonal shows correct predictions, "
             "and every other square shows a mistake.")

    with st.spinner("Checking the model on 10,000 test images..."):
        matrix = get_confusion_matrix()

    matrix_rows = []
    for actual in range(10):
        for predicted in range(10):
            matrix_rows.append({
                "Actual": actual,
                "Predicted": predicted,
                "Count": int(matrix[actual, predicted])
            })
    matrix_data = pd.DataFrame(matrix_rows)

    heatmap = alt.Chart(matrix_data).mark_rect().encode(
        x=alt.X("Predicted:O", title="Predicted Digit", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:O", title="Actual Digit"),
        color=alt.Color("Count:Q", legend=None)
    )

    labels = alt.Chart(matrix_data).mark_text(fontSize=10).encode(
        x="Predicted:O",
        y="Actual:O",
        text="Count:Q",
        color=alt.condition(alt.datum.Count > 500,
                            alt.value("black"),
                            alt.value("white"))
    )

    st.altair_chart(heatmap + labels)

with pixel_tab:
    if has_drawing:
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
    else:
        st.info("Draw a digit to see its pixel values.")
    