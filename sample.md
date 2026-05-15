# ♻️ Garbage Image Classification

Website Link: [Garbage Classifier](https://garbageclassifier.duckdns.org/)

### 📌 About the Project

Garbage Image Classification is a deep learning-based application designed to automatically identify and categorize waste materials from images. The system classifies waste into categories such as:

* Plastic
* Metal
* Glass
* Paper
* Organic

The goal of this project is to assist in automating waste segregation, improving recycling efficiency, and reducing human effort through an intelligent image classification model deployed via a simple user interface.

## 🧪 Project Approach

### 1️⃣ Data Collection

* Gather a labeled dataset of garbage images across categories:

  * Plastic, Metal, Glass, Paper, Organic
* Ensure diversity in lighting, angles, and backgrounds for better generalization

---

### 2️⃣ Data Cleaning & Preprocessing

* Resize all images to **180 × 180**
* Normalize pixel values

  
---

### 3️⃣ Exploratory Data Analysis (EDA)

* Visualize the number of images per class to check class balance
* Display sample images from each category
* Analyze pixel intensity and color distribution patterns

---

### 4️⃣ Model Development

* Use **Transfer Learning** with pre-trained models:

  * ResNet50
  * MobileNetV2
  * VGG16 / VGG19

* Approach:

  * Load pre-trained model (ImageNet weights)
  * Freeze base layers
  * Add custom classification layers:

    * Dense layers
    * Dropout (to prevent overfitting)
    * Softmax output layer

---

### 5️⃣ Model Evaluation

* Evaluate models using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

* Generate and analyze:

  * Confusion Matrix
  * Misclassification patterns

---

### 6️⃣ Best Model Selection

* Select the model with:

  * Highest **F1-Score**
  * Balanced performance across all classes

* ✅ Final selected model: **ResNet50**

---

### 7️⃣ Application Development

* Build an interactive **Streamlit** application with:

  * 📤 Image upload feature
  * 🤖 Real-time prediction
  * 📊 Display predicted class with confidence score

---

📈 Experiment Tracking with MLflow & DagsHub

To ensure reproducibility, experiment comparison, and proper model management, this project integrates MLflow with DagsHub for tracking experiments, metrics, and artifacts.


### `.env` variables to setup

`MLFLOW_USERNAME` - Dagshub Username

`MLFLOW_PASSWORD` - Dagshub Password Token

`MLFLOW_EXPERIMENT_NAME` - Garbage Image Classifier

`MLFLOW_TRACKING_URI` - Dagshub MLFlow Tracking URI

