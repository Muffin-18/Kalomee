import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import config

# Load model
model = tf.keras.models.load_model("food_cnn_model.keras")
nutrition_df = pd.read_csv(config.CSV_PATH)
nutrition_df.columns = nutrition_df.columns.str.strip()
nutrition_df["food"] = nutrition_df["food"].str.lower().str.replace(" ", "_")

# Class names
class_names = sorted(nutrition_df["food"].unique())

st.title("Food Recognition + Nutrition Analysis")
st.write("Upload a food image to identify it and show nutritional values.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    img = image.load_img(uploaded, target_size=config.IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx]

    predicted_food = class_names[idx]

    st.subheader(f"Predicted Food: **{predicted_food.replace('_',' ').title()}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    match = nutrition_df[nutrition_df['food'] == predicted_food]

    if not match.empty:
        st.subheader("Nutrition Information per 100g:")
        st.write(match.iloc[0].to_frame())
    else:
        st.error("No nutrition data available for this food.")
