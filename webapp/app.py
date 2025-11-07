from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

# Load your Keras model (change path if needed)
MODEL_PATH = "mask_detector_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# If model.predict expects shape (1,224,224,3)
IMG_SIZE = (224, 224)

# simple preprocess (change according to how you trained)
def preprocess_image_pil(image: Image.Image, img_size=IMG_SIZE):
    image = image.convert("RGB")
    image = image.resize(img_size)
    arr = np.asarray(image) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# optional face detection using Haar cascade (put haar xml in same folder or install opencv default)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# If your model expects face crops, you can use this to extract faces.
def detect_faces_and_predict(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    results = []
    for (x,y,w,h) in faces:
        face = np_img[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        inp = preprocess_image_pil(face_pil)
        preds = model.predict(inp)[0]
        # assume binary: [mask_prob, no_mask_prob] or single sigmoid
        if preds.shape[0] == 2:
            mask_prob = float(preds[0])
            no_mask_prob = float(preds[1])
        else:
            # single prob -> treat as mask_prob
            mask_prob = float(preds[0])
            no_mask_prob = 1.0 - mask_prob
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        confidence = max(mask_prob, no_mask_prob)
        results.append({
            "box": [int(x), int(y), int(w), int(h)],
            "label": label,
            "confidence": round(confidence, 4)
        })
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Expecting form-data file OR JSON {image: "data:image/png;base64,..."}
    if "file" in request.files:
        file = request.files["file"].read()
        image = Image.open(io.BytesIO(file))
        inp = preprocess_image_pil(image)
        preds = model.predict(inp)[0]
        if preds.shape[0] == 2:
            mask_prob = float(preds[0])
            no_mask_prob = float(preds[1])
        else:
            mask_prob = float(preds[0])
            no_mask_prob = 1.0 - mask_prob
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        confidence = max(mask_prob, no_mask_prob)
        return jsonify({"label": label, "confidence": round(confidence,4)})
    else:
        data = request.get_json(force=True)
        img_b64 = data.get("image", "")
        if img_b64.startswith("data:image"):
            header, img_b64 = img_b64.split(",", 1)
        try:
            img_bytes = base64.b64decode(img_b64)
            np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error":"invalid image","details":str(e)}), 400

        # Option A: If model expects whole image
        pil = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        inp = preprocess_image_pil(pil)
        preds = model.predict(inp)[0]
        if preds.shape[0] == 2:
            mask_prob = float(preds[0])
            no_mask_prob = float(preds[1])
        else:
            mask_prob = float(preds[0])
            no_mask_prob = 1.0 - mask_prob
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        confidence = max(mask_prob, no_mask_prob)

        # Optionally: run face detection and give per-face results
        faces_results = detect_faces_and_predict(np_img)
        return jsonify({
            "label": label,
            "confidence": round(confidence,4),
            "faces": faces_results
        })

if __name__ == "__main__":
    app.run(debug=True)
