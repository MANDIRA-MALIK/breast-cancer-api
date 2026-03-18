import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("model.h5", compile=False)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return {"prediction": "Malignant"}
    else:
        return {"prediction": "Benign"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
