from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

app = FastAPI()

# Lazy load model (IMPORTANT for Render)
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("model.h5")
    return model


# GradCAM function
def get_gradcam(img_array, model, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load model safely
        model = load_model()

        # Read image
        image = Image.open(file.file).convert("RGB")
        img = image.resize((160, 160))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = float(model.predict(img_array)[0][0])

        # GradCAM
        heatmap = get_gradcam(img_array, model)

        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = heatmap * 0.4 + np.array(image)

        _, buffer = cv2.imencode(".jpg", superimposed)

        return {
            "prediction": pred,
            "gradcam": buffer.tobytes().hex()
        }

    except Exception as e:
        return {"error": str(e)}
