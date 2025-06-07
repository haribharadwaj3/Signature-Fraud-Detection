# ✍️ Signature Fraud Detection 

This project detects whether a signature is **genuine or forged** using a deep learning approach that combines **Convolutional Neural Networks (CNN)** and **Scale-Invariant Feature Transform (SIFT)** for enhanced feature extraction.

## 📌 Objective

To automate the verification of handwritten signatures by comparing two input images and classifying them as **genuine** or **forged** with great accuracy.

---

## 🧠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- SIFT (Scale-Invariant Feature Transform)
- CNN (Convolutional Neural Network)
- Flask (for basic UI)

---

## 📁 Project Structure

```
Signature-Fraud-Detection/
├── app.py                # Flask-based UI to upload and compare signatures
├── templates/            # HTML templates for web UI
├── requirements.txt      # All required Python libraries
└── README.md             # Project documentation
```



## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/signature-fraud-detection.git
   cd signature-fraud-detection
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # For Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and go to:
   ```
   http://localhost:5000
   ```

---

## 🧪 Dataset

- **CEDAR Dataset** was used for training and testing.
- Publicly available at [CEDAR Signature Dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures)

---

## 📊 Features

- Upload two signature images and compare them
- Uses CNN for feature learning
- Integrates SIFT for fine-grained signature characteristics
- Classifies input as **Genuine** or **Forged**
- Flask UI for ease of interaction

