# 🚗 AI Driver Drowsiness & Fatigue Detector

An interactive web application built with Streamlit and TensorFlow that detects driver drowsiness and fatigue in real-time. This project utilizes a **Soft Voting Ensemble** of two deep learning models (EfficientNetB3) to provide a highly accurate, unbiased prediction of a driver's alert state.

## 🌟 Features

* **Dual Input Methods:** Users can either upload an existing image (`.jpg`, `.png`) or capture a live photo directly from their device's webcam/mobile camera.
* **Model Ensembling:** Instead of relying on a single neural network, this app loads two separate EfficientNetB3 models—one trained specifically on drowsiness features and another on general fatigue. 
* **Bias Reduction:** The application uses a soft-voting mechanism to mathematically average the confidence scores from both models, acting as a fail-safe against false positives.
* **Interactive UI:** Built entirely in Python using Streamlit for a clean, responsive, and user-friendly interface.

## 🧠 Technical Architecture

1. **Base Model:** EfficientNetB3 (Pre-trained on ImageNet, fine-tuned for facial state classification).
2. **Input Processing:** Images are dynamically resized to `300x300` pixels and converted to tensor arrays to match the models' expected input shapes.
3. **Inference Logic:**
   * Model A (Drowsiness) -> Outputs `[Danger_Prob, Safe_Prob]`
   * Model B (Fatigue) -> Outputs `[Danger_Prob, Safe_Prob]`
   * The app averages the `Danger` probabilities and `Safe` probabilities independently to calculate the final ensemble confidence.

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Machine Learning:** TensorFlow / Keras
* **Data Manipulation:** NumPy
* **Image Processing:** Pillow (PIL)

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed, then clone this repository to your local machine.

### 1. Install Dependencies
Navigate to the project folder and install the required libraries:
```bash
pip install -r requirements.txt
