# Violence Detection using OpenPose and DIFEM Features

A deep learning-based violence detection system that extracts pose-based features from videos using OpenPose and classifies them using the Distance-based Interaction Feature Extraction Method (DIFEM).

## 📋 Overview

This project implements a violence detection pipeline for video surveillance that:
- Extracts human pose keypoints from videos using OpenPose
- Computes distance-based interaction features (DIFEM) including joint velocities, person proximity, and wrist overlaps
- Trains machine learning classifiers (Random Forest, SVM, Logistic Regression, etc.) for violence classification
- Achieves robust performance on the RWF-2000 dataset

## 🎯 Features

### DIFEM Feature Extraction
The Distance-based Interaction Feature Extraction Method (DIFEM) computes the following features:

1. **Joint Velocity Features**
   - Weighted velocity calculation for 11 key joints (wrists, elbows, hips, knees, ankles, neck)
   - Mean, max, and variance of velocities across video frames

2. **Proximity-based Features**
   - **Closeness Factor**: Measures spatial proximity between people in a frame
   - **Wrist Overlap**: Detects potential physical interactions through wrist position overlaps
   - Statistical measures (mean and variance) for both metrics

### Supported Classifiers
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Gradient Boosting
- AdaBoost
- Decision Tree

## 🗂️ Dataset

This project uses the **RWF-2000 (Real-world Fight) dataset**:
- 2,000 video clips collected from YouTube
- Two classes: Fight and Non-Fight
- Split: 1,600 training videos, 400 validation videos
- Real-world surveillance scenarios with varying conditions

## 🚀 Getting Started

### Prerequisites

```bash
# Python dependencies
pip install numpy opencv-python scikit-learn matplotlib seaborn

# For OpenPose extraction (Google Colab recommended)
# See openpose_extraction.py for detailed setup
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/violence-detection-difem.git
cd violence-detection-difem
```

2. Download the RWF-2000 dataset and organize it as follows:
```
data/
├── videos/
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
```

## 📖 Usage

### Step 1: Extract OpenPose Features

Run the OpenPose extraction script (recommended on Google Colab with GPU):

```python
# See openpose_extraction.py for complete implementation
python openpose_extraction.py
```

This will generate JSON files containing pose keypoints for each video frame.

### Step 2: Extract DIFEM Features and Train Classifiers

```python
# See difem_features.py for complete implementation
python difem_features.py
```

This script will:
1. Process OpenPose JSON files
2. Compute DIFEM features (velocity, proximity, wrist overlap)
3. Train multiple classifiers
4. Generate performance metrics and visualizations

## 🏗️ Architecture

### Pipeline Overview

```
Video Input → OpenPose → Pose Keypoints → DIFEM Features → ML Classifier → Violence/Non-Violence
```

### DIFEM Feature Computation

1. **Joint Selection**: 11 key joints identified as most relevant for violence detection
2. **Weighted Velocity**: Distance traveled by each joint weighted by importance
3. **Proximity Analysis**: Euclidean distance between people's center points
4. **Interaction Detection**: Overlap detection for wrist positions indicating physical contact

## 📊 Results

The models are evaluated on multiple metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curves

Sample performance (will vary based on dataset split):
- Random Forest: ~85-90% accuracy
- SVM: ~82-87% accuracy
- Logistic Regression: ~80-85% accuracy

## 📁 Project Structure

```
violence-detection-difem/
├── README.md
├── LICENSE
├── requirements.txt
├── openpose_extraction.py      # OpenPose pose extraction from videos
├── difem_features.py           # DIFEM feature extraction and training
├── data/                        # Dataset directory (not included)
│   └── videos/
├── outputs/                     # Generated features and models
│   ├── train_features.pkl
│   ├── val_features.pkl
│   └── models/
└── notebooks/                   # Jupyter notebooks for analysis
    └── exploratory_analysis.ipynb
```

## 🔬 Methodology

### OpenPose Extraction
- Uses CMU's OpenPose BODY_25 model
- Extracts 25 body keypoints per person per frame
- Processes videos frame-by-frame
- Outputs JSON files with keypoint coordinates and confidence scores

### DIFEM Features
- **Velocity-based**: Captures motion intensity and patterns
- **Distance-based**: Measures spatial relationships between people
- **Interaction-based**: Detects potential physical contact through wrist overlaps

### Classification
- Features normalized using StandardScaler
- Multiple classifiers trained for comparison
- Hyperparameter tuning using GridSearchCV (optional)
- Stratified cross-validation for robust evaluation

## 🛠️ Technical Details

### Selected Joints and Weights
```python
selected_joints = [
    "right_wrist", "left_wrist",      # Weight: 1.0
    "right_elbow", "left_elbow",      # Weight: 0.8
    "right_hip", "left_hip",          # Weight: 1.0
    "right_knee", "left_knee",        # Weight: 1.0
    "right_ankle", "left_ankle",      # Weight: 1.0
    "neck"                             # Weight: 1.0
]
```

### Feature Vector
For each video, a 7-dimensional feature vector is computed:
1. Mean velocity
2. Max velocity
3. Variance of velocity
4. Mean closeness factor
5. Variance of closeness factor
6. Mean wrist overlap
7. Variance of wrist overlap

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{violence-detection-difem,
  author = {Your Name},
  title = {Violence Detection using OpenPose and DIFEM Features},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/violence-detection-difem}
}
```

## 🙏 Acknowledgments

- OpenPose by CMU Perceptual Computing Lab
- RWF-2000 Dataset creators
- scikit-learn community

## 📧 Contact

Mailto - [himanshumittal.hbti@gmail.com]



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project was developed as part of research work on violence detection in surveillance videos. The DIFEM approach provides an interpretable, efficient alternative to deep learning methods while maintaining competitive performance.
