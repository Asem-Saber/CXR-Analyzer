# 🫁 CXR-Analyzer: Dual-Task COVID-19 Radiography AI
> **Deep Learning-powered Chest X-Ray Analysis for COVID-19 Detection and Lung Segmentation**  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

An end-to-end deep learning pipeline and interactive web application for the simultaneous classification and lesion segmentation of Chest X-Rays (CXRs). 

This project utilizes a custom dual-headed U-Net architecture built in PyTorch to predict four lung conditions while simultaneously generating a precise pixel-level mask of the affected lung regions.


---

## 📂 Project Structure

```text
CXR-Analyzer/
│
├── app.py              # Streamlit web interface for deployment
├── dataset.py          # Kaggle dataset parsing, DataLoaders, and v2 Transforms
├── evaluate.py         # Validation loop and Binary Jaccard Index (IoU) calculation
├── inference.py        # Single-image processing and mask overlay generation
├── main.py             # CLI entry point for training and inference
├── model.py            # Custom Dual-Head U-Net architecture
├── train.py            # Training loop and custom combined loss function
├── best_model.pth      # Saved model weights (Generated after training)
└── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## 💾 Dataset

This project is trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle.

The dataset contains Chest X-Rays for 4 distinct classes along with their corresponding segmentation masks:

| Class | Description |
|-------|-------------|
| **COVID** | COVID-19 positive cases |
| **Lung_Opacity** | Non-COVID lung infections |
| **Normal** | Healthy lungs |
| **Viral Pneumonia** | Viral pneumonia cases |

--- 

## 🚀 How to Run the Entire Project

### 1. Prerequisites

- Python 3.8+
- pip or conda

### 2. Setup
```bash
git clone https://github.com/yourusername/CXR-Analyzer.git
cd CXR-Analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt
```

### 3. Training the Model
```bash
python main.py --mode train --data_dir COVID-19_Radiography_Dataset --epochs 10 --batch_size 16 --lr 0.005
```

### 4. Run Inference
```bash
python main.py --mode infer --image path/to/your/test/image.png --weights best_model.pth
```

### 5. Launching the Web Application
```bash
streamlit run app.py
```
Once running, open <http://localhost:8051> in your web browser.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
⭐ Star this repo if you found it helpful! ⭐
</div>
