# Violence Detection Project - Setup and Usage Guide

## Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- Google Colab account (recommended for OpenPose extraction)
- RWF-2000 Dataset

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/himanshuhbti/violence-detection-difem.git
cd violence-detection-difem
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download and Organize Dataset
Download the RWF-2000 dataset and organize it as follows:
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

## Step-by-Step Workflow

### Step 1: Extract OpenPose Features (Google Colab)

1. Open `openpose_extraction.py` in Google Colab
2. Install OpenPose (first-time setup):
   ```python
   # Run the installation commands in the script
   # This will download and build OpenPose (~10-15 minutes)
   ```

3. Download pretrained models and place them in:
   - `openpose/models/face/pose_iter_116000.caffemodel`
   - `openpose/models/hand/pose_iter_102000.caffemodel`
   - `openpose/models/pose/body_25/pose_iter_584000.caffemodel`
   - `openpose/models/pose/coco/pose_iter_440000.caffemodel`
   - `openpose/models/pose/mpi/pose_iter_160000.caffemodel`

4. Update paths in the configuration section:
   ```python
   VIDEO_FOLDER = "./data/videos/train/Fight"
   OUTPUT_FOLDER = "./outputs/openpose_json/train/Fight"
   ```

5. Run the script to extract pose keypoints
   - This will create JSON files for each video frame
   - Processing time: ~1-2 minutes per video (GPU)

6. Repeat for all categories (train/Fight, train/NonFight, val/Fight, val/NonFight)

### Step 2: Extract DIFEM Features and Train Models

1. Ensure OpenPose JSON files are organized:
   ```
   data/
   ├── train/
   │   ├── Fight/
   │   │   ├── video1/
   │   │   │   ├── 000000000000_keypoints.json
   │   │   │   ├── 000000000001_keypoints.json
   │   │   │   └── ...
   │   │   └── video2/
   │   └── NonFight/
   └── val/
       ├── Fight/
       └── NonFight/
   ```

2. Run the DIFEM feature extraction:
   ```bash
   python difem_features_clean.py
   ```

3. The script will:
   - Process all videos and extract DIFEM features
   - Save features to `outputs/train_features_enhanced.pkl` and `outputs/val_features_enhanced.pkl`
   - Train multiple classifiers (Random Forest, SVM, Logistic Regression, etc.)
   - Display performance metrics for each classifier

## Understanding DIFEM Features

### Feature Vector (7 dimensions)

1. **Mean Velocity**: Average movement speed of all joints across the video
2. **Max Velocity**: Maximum movement speed observed
3. **Variance of Velocity**: Variation in movement patterns
4. **Mean Closeness Factor**: Average proximity between people (1/distance)
5. **Variance of Closeness Factor**: Variation in proximity
6. **Mean Wrist Overlap**: Average frequency of wrist overlaps (indicating physical contact)
7. **Variance of Wrist Overlap**: Variation in wrist overlap patterns

### Selected Joints (11 key points)
- **Wrists** (left & right): High-importance for detecting strikes
- **Elbows** (left & right): Medium-importance for arm movements
- **Hips** (left & right): Core body position
- **Knees** (left & right): Leg movements (kicks)
- **Ankles** (left & right): Footwork patterns
- **Neck**: Upper body orientation

## Customization Options

### Modify Joint Selection
Edit `SELECTED_JOINTS` in `difem_features_clean.py`:
```python
SELECTED_JOINTS = [
    "right_wrist", "left_wrist",
    # Add or remove joints as needed
]
```

### Adjust Joint Weights
Edit `JOINT_WEIGHTS` to change importance:
```python
JOINT_WEIGHTS = {
    "right_wrist": 1.5,  # Increase importance
    "left_wrist": 1.5,
    "right_elbow": 0.6,  # Decrease importance
    # ...
}
```

### Change Classifier
Modify the `classifiers_to_test` list:
```python
classifiers_to_test = [
    'random_forest',
    'svm',
    # Add or remove classifiers
]
```

### Hyperparameter Tuning
Use scikit-learn's GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## Expected Performance

| Classifier | Accuracy Range | Notes |
|------------|---------------|-------|
| Random Forest | 85-90% | Best overall performance |
| SVM | 82-87% | Good for smaller datasets |
| Logistic Regression | 80-85% | Fast, interpretable |
| Gradient Boosting | 83-88% | Slower training |
| AdaBoost | 81-86% | Good with weak learners |
| Decision Tree | 75-82% | Baseline model |

*Note: Actual performance depends on dataset quality and hyperparameters*

## Troubleshooting

### Issue: OpenPose installation fails
**Solution**: Use Google Colab with GPU runtime. Ensure all system dependencies are installed.

### Issue: "No people detected" in many frames
**Solution**: 
- Check video quality
- Adjust OpenPose confidence threshold
- Verify model files are correctly placed

### Issue: Low classification accuracy
**Solution**:
- Increase training data
- Tune hyperparameters
- Try different feature combinations
- Check for class imbalance

### Issue: Memory error during processing
**Solution**:
- Process videos in smaller batches
- Reduce video resolution
- Use progressive saving (already implemented)

## Output Files

```
outputs/
├── train_features_enhanced.pkl    # Training features (7D vectors)
├── val_features_enhanced.pkl      # Validation features (7D vectors)
├── openpose_json/                 # Raw OpenPose outputs
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
└── models/                        # Saved trained models (optional)
```

## Performance Optimization Tips

1. **GPU Acceleration**: Always use GPU for OpenPose extraction
2. **Batch Processing**: Process multiple videos in parallel if resources allow
3. **Feature Caching**: Save extracted features to avoid recomputation
4. **Model Selection**: Start with Random Forest for best accuracy/speed tradeoff

## Next Steps

After getting the baseline working:

1. **Experiment with features**:
   - Add temporal features (velocity changes over time)
   - Include more joints or body parts
   - Try different distance metrics

2. **Improve models**:
   - Ensemble multiple classifiers
   - Use deep learning (LSTM, CNN) on raw keypoints
   - Implement sliding window for temporal context

3. **Deploy**:
   - Create a REST API for real-time inference
   - Build a web interface for video upload and classification
   - Optimize for edge devices

## Citation

If you use this project, please cite:

```bibtex
@misc{violence-detection-difem-2024,
  author = {Himanshu},
  title = {Violence Detection using OpenPose and DIFEM Features},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/himanshuhbti/violence-detection-difem}
}
```

## Contact and Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: [himanshumittal.hbti@gmail.com]


## License

MIT License - See LICENSE file for details
