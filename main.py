from fastapi import FastAPI
from ultralytics import YOLO
from PIL import Image
import requests
import io
import base64

# Cargar el modelo YOLOv8
model = YOLO('pesos/yolov8s.pt')

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

@app.get("/predict/{image_name}")
async def predict(image_name: str):
    # Abrir la imagen
    img = Image.open(image_name)
    return predict_image(img)

@app.get("/predict_url")
async def predict_url(image_url: str):
    # Descargar la imagen de la URL
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    return predict_image(img)

@app.get("/predict_base64")
async def predict_base64(image_base64: str):
    # Decodificar la imagen de la cadena base64
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data))
    return predict_image(img)

def predict_image(img):
    # Ejecutar la inferencia en la imagen
    results = model.predict(img)

    output = []
    for r in results:
        for box, class_id in zip(r.boxes.xyxy, r.boxes.cls):
            class_name = model.names[int(class_id)]
            x1, y1, x2, y2 = box
            output.append({
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                },
                "class": class_name
            })

    return output