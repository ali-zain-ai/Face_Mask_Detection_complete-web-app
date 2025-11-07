# 🧠 Face Mask Detection Web App

This is a **Face Mask Detection System** built using **Deep Learning** and **Flask**.
The model detects whether a person is wearing a face mask or not from an image — and provides real-time predictions through a simple **web interface**.

---

## 🚀 Features

* Detects **Mask / No Mask** using a trained CNN model
* Supports both **image uploads** and **base64 inputs** (for webcam or API use)
* Built with **TensorFlow**, **Keras**, and **OpenCV**
* Web interface made using **Flask (Python)**
* Optional **face detection** using OpenCV’s Haar Cascade

---

## 📁 Project Structure

```
face-mask-detection/
│
├── model/
│   ├── mask_detector_model.h5         # Trained model (download link below)
│   ├── Face_Mask_Detection.ipynb      # Model training notebook
│   └── sample_data/                   # Optional example images
│
├── webapp/
│   ├── app.py                         # Flask app file
│   ├── templates/
│   │   └── index.html                 # Frontend template
│   ├── static/                        # CSS, JS, and images
│   └── requirements.txt               # Python dependencies
│
├── README.md
└── .gitignore
```


### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

After downloading:

```
face-mask-detection/
└── model/
    └── mask_detector_model.h5
```

---

## ▶️ Run the App

From the `webapp` directory:

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

Upload an image — the app will predict whether the person is wearing a **mask** or **no mask**, and show confidence levels.

---

## 🧩 Technologies Used

* **Python 3.9+**
* **TensorFlow / Keras**
* **Flask**
* **OpenCV**
* **NumPy**
* **Pillow (PIL)**

---

## 📊 Model Overview

The CNN model was trained on a labeled dataset of face images with and without masks.
It outputs probabilities for each class:

* **Mask**
* **No Mask**

You can retrain or fine-tune it using the Jupyter notebook:

```
Face_Mask_Detection.ipynb
```

---

## 📸 Sample Output

*(You can add example screenshots here later)*
![Mask Detection Example](static/sample_output.jpg)

---

## 🧾 License

This project is released under the [MIT License](LICENSE).

---

## 👤 Author

**Ali Zain**
📧 mindfuelbyali@gmail.com
📱 Instagram: [@mindfuelbyali](https://instagram.com/mindfuelbyali)
