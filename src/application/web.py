import streamlit as st
import numpy as np
from PIL import Image

st.title('Face recognition App')

owner_image_file = st.file_uploader(
    "Choose Image to remember",
    type=['png', 'jpg']
)

target_img_file = st.file_uploader(
    "Choose target Image",
    type=['png', 'jpg']
)

images = []
caption = []

if owner_image_file is not None:
    owner_image = Image.open(owner_image_file)
    images = [owner_image]
    caption = ["Owner"]

if target_img_file is not None:
    target_img = Image.open(target_img_file)
    images += [target_img]
    caption += ["Target"]

st.image(
    image=images,
    caption=caption,
    width=100
)
