import os
import torch
import numpy as np
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import albumentations as albu

from src.data.preprocessing import (
    augmentation_transforms,
    post_transform,
    pre_transform
)

from src.models.inference_model import FaceRecModel
from src.pipeline.save_load_model import MODELS_PATH, load_model


@st.cache_resource
def load_my_model(model_name: str):
    return load_model(model_name)


@st.cache_resource
def create_recognizer(model_name: str) -> FaceRecModel:
    model = load_my_model(model_name)

    basic_transformation = albu.Compose([
        pre_transform(), post_transform()
    ])

    augmentation_transformation = albu.Compose([
        pre_transform(), augmentation_transforms(), post_transform()
    ])

    recognizer = FaceRecModel(
        model,
        basic_transformation,
        augmentation_transformation
    )

    return recognizer


def remember_owner(
    recognizer: FaceRecModel,
    image: Image.Image
) -> None:
    image = np.array(image)
    st.write(image.shape)
    st.write("Working around with a picture...")
    recognizer.learn_owner(image)
    st.write("Done")


def verify_owner(
    image: Image.Image
) -> int:
    return recognizer.recognize_face(np.array(image))


owner_image_file = st.file_uploader(
    "Provide Image of Owner",
    type=['png', 'jpg']
)

target_img_file = st.file_uploader(
    "Provide image to verify",
    type=['png', 'jpg']
)

images = []
caption = []

if owner_image_file is not None:
    owner_image = Image.open(owner_image_file).convert("RGB")
    images = [owner_image]
    caption = ["Owner"]

if target_img_file is not None:
    target_img = Image.open(target_img_file).convert("RGB")
    images += [target_img]
    caption += ["Target"]


st.image(
    image=images,
    caption=caption,
    width=100
)

model_name = st.selectbox(
    'Choose model',
    os.listdir(MODELS_PATH)
)

if model_name is not None:
    recognizer = create_recognizer(model_name)

st.write('Your selected model: ', model_name)

if owner_image_file is not None:
    is_remember_owner_button_pressed = st.button(
        label="Remember owner"
    )
    if is_remember_owner_button_pressed:
        remember_owner(
            recognizer=recognizer,
            image=owner_image
        )

if target_img_file is not None:
    is_verify_button_pressed = st.button(
        label="Verify owner"
    )
    if is_verify_button_pressed:
        is_owner = verify_owner(target_img)
        if is_owner:
            st.text("Matched")
        else:
            st.text("Not matched")
