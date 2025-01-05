# 🩺 Skin Cancer Diagnosis System using Deep Learning

A **computer-aided diagnosis system** built with deep learning to distinguish between benign and malignant skin cancer cells. This project employs a two-stage approach, leveraging **U-Net** for image segmentation and **InceptionV3** for classification, further enhanced with an attention module for improved accuracy and reliability.

---

## 🛠️ Features

- **Two-Stage Diagnosis Pipeline**:
  - **Stage 1**: Image segmentation using **U-Net** to isolate regions of interest (ROIs).
  - **Stage 2**: Classification using **InceptionV3** to identify benign or malignant cancer cells.

- **Attention Mechanism**:
  - Integrated an attention module to enhance the model’s focus on critical areas, improving overall performance.

- **User-Friendly Interface**:
  - Built a web application using **Flask** for easy interaction and visualization of results.

- **Preprocessing**:
  - Utilized **OpenCV** for image preprocessing, including resizing, normalization, and augmentation.

---

## 📚 Technologies Used

### Deep Learning Frameworks:
- **TensorFlow**: For building and training U-Net and InceptionV3 models.
- **Scikit-Learn**: For data preprocessing and performance evaluation.

### Image Processing:
- **OpenCV**: For image preprocessing and augmentation.

### Web Application:
- **Flask**: To deploy the trained model as a user-friendly web application.
