#!/bin/bash
# setup.sh - Project setup script for Violence Detection using DIFEM

echo "=================================================="
echo "Violence Detection Project - Setup Script"
echo "=================================================="
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/videos/train/Fight
mkdir -p data/videos/train/NonFight
mkdir -p data/videos/val/Fight
mkdir -p data/videos/val/NonFight

mkdir -p data/train/Fight
mkdir -p data/train/NonFight
mkdir -p data/val/Fight
mkdir -p data/val/NonFight

mkdir -p outputs/openpose_json/train/Fight
mkdir -p outputs/openpose_json/train/NonFight
mkdir -p outputs/openpose_json/val/Fight
mkdir -p outputs/openpose_json/val/NonFight

mkdir -p outputs/models
mkdir -p notebooks

echo "✓ Directory structure created"
echo ""

# Create placeholder README files
echo "Creating placeholder files..."

cat > data/README.md << 'EOF'
# Data Directory

## Structure

```
data/
├── videos/              # Original video files
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
└── train/               # OpenPose JSON outputs for training
    ├── Fight/
    │   ├── video1/
    │   │   ├── 000000000000_keypoints.json
    │   │   ├── 000000000001_keypoints.json
    │   │   └── ...
    │   └── video2/
    └── NonFight/
```

## Dataset

Download the RWF-2000 dataset and place videos in the appropriate folders:
- Training Fight videos → `videos/train/Fight/`
- Training NonFight videos → `videos/train/NonFight/`
- Validation Fight videos → `videos/val/Fight/`
- Validation NonFight videos → `videos/val/NonFight/`

After running OpenPose extraction, JSON files will be organized in:
- `train/Fight/` and `train/NonFight/` for training data
- `val/Fight/` and `val/NonFight/` for validation data
EOF

cat > outputs/README.md << 'EOF'
# Outputs Directory

This directory contains:

## Feature Files
- `train_features_enhanced.pkl` - Training features (7D vectors)
- `val_features_enhanced.pkl` - Validation features (7D vectors)
- `train_features_basic.pkl` - Basic features (3D vectors)
- `val_features_basic.pkl` - Basic validation features

## OpenPose JSON Files
- `openpose_json/` - Raw pose keypoint data from OpenPose

## Trained Models
- `models/` - Saved classifier models (optional)

## Analysis Results
- Classification reports
- Confusion matrices
- Performance visualizations
EOF

echo "✓ Placeholder files created"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠ pip not found. Please install dependencies manually:"
    echo "  pip install -r requirements.txt"
fi
echo ""

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Download RWF-2000 dataset and place in data/videos/"
echo "2. Run OpenPose extraction (use Google Colab)"
echo "3. Run DIFEM feature extraction and training:"
echo "   python difem_features_clean.py"
echo ""
echo "See USAGE_GUIDE.md for detailed instructions"
echo ""
