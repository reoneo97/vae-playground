import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import app_utils
from PIL import Image
st.title("VAE Playground")
# st.sidebar()
st.markdown("This is a simple streamlit app to showcase how different VAEs "
            "function and how the differences in architecture and "
            "hyperparameters will show up in the generated images \n \n"
            "In this example, we will look at two different architectures")
st.write("This is some additional text")
st.write("does this work now?")


@st.cache
def load_model_files():
    files = os.listdir("saved_models/")
    return files


files = load_model_files()
model = st.selectbox("Choose Model:",files)


stroke_width = st.sidebar.slider("Stroke width: ", 5, 10, 3)
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
col1, col2 = st.beta_columns(2)

with col1:
    canvas_result_1 = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color=bg_color,
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas1",
    )
    if st.button("Picture 1"):
        st.write("Button 1")
    else:
        st.write("Not pressed")
with col2:
    canvas_result_2 = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color=bg_color,
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas2",
    )