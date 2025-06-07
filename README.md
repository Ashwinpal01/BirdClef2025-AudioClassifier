# 🐦 BirdCLEF 2025 – Birdsong Classification with CNNs (Keras)

This project is built for classifying bird species from their audio recordings, using deep learning techniques inspired by the [BirdCLEF 2025 competition on Kaggle](https://www.kaggle.com/competitions/birdclef-2025/data). The system takes raw `.ogg` audio files, processes them into Mel spectrogram images, and feeds them into a Convolutional Neural Network (CNN) built with Keras and TensorFlow.

---

## 💡 Ideation
Birdsong identification has ecological and conservation significance but poses challenges due to overlapping sounds, noise, and class imbalance. This project explores the feasibility of an audio-to-image classification pipeline using spectrograms and CNNs to address these issues in a scalable way.

---

## ⚙️ Technologies Used
- **Languages**: Python
- **Frameworks**: Keras, TensorFlow
- **Audio Processing**: Librosa
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **IDE**: Jupyter Notebook

---

## 🧪 Methodology
1. **Audio Preprocessing**:  
   - Converted audio files into Mel spectrograms using Librosa  
   - Resized and normalized spectrograms for CNN input

2. **Model Architecture**:  
   - Built a custom CNN with Conv2D, MaxPooling, Dropout, and Dense layers in Keras  
   - Used EarlyStopping and ModelCheckpoint callbacks to prevent overfitting

3. **Training & Evaluation**:  
   - Monitored performance on training and validation sets  
   - Evaluated with accuracy and confusion matrix; future plans include augmentation and transfer learning

---

## 🧠 Model Used
- Custom-built **Convolutional Neural Network (CNN)**  
- Trained from scratch on Mel spectrogram images  
- Potential future extension to use **pretrained CNNs** (e.g., MobileNet, ResNet)

---

## 📊 Current Status
- The model demonstrates low accuracy (~3.4%) on noisy, real-world audio
- Performance is limited by overlapping sounds and imbalanced classes
- Ongoing improvements planned with better augmentation and hyperparameter tuning

---

## 📁 Dataset

The dataset is available at the official [BirdCLEF 2025 Kaggle page](https://www.kaggle.com/competitions/birdclef-2025/data).  
Due to size and licensing restrictions, it is not included in this repository.  

---


