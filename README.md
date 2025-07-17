# CNN-Based Defect Detection in Metal Additive Manufacturing

This project focuses on applying deep learning techniques, specifically Convolutional Neural Networks (CNNs), to detect and classify common defects in metal additive manufacturing (AM) processes, including **Gas Porosity** and **Lack of Fusion (LoF)**.

## 🔍 Objective
To develop a robust image processing model that can automatically inspect and identify defects from metallurgical cross-section images or weld bead layers, improving quality control in AM workflows.

## 🛠️ Features
- Preprocessed labeled images of gas and LoF defects
- CNN architecture for binary/multi-class classification
- Training and validation with high accuracy
- Support for model evaluation using precision, recall, and F1-score
- Lightweight and scalable implementation using TensorFlow/Keras

## 📦 Dataset
- Contains real or simulated bead images from metal additive manufacturing processes.
- Defect categories:
  - `gas`: Spherical pores due to trapped gas
  - `lof`: Incomplete fusion between layers

## 🚀 Technologies Used
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas / Matplotlib

## 📈 Results
The CNN achieved high classification accuracy and proved effective in identifying microscopic defects that may not be visible to the naked eye.

## 🤝 Contributions & Extensions
You can extend this by:
- Adding segmentation for defect localization
- Integrating with real-time monitoring systems
- Using transfer learning for faster convergence

---

Feel free to fork or contribute!
