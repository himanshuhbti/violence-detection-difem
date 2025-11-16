"""
Distance-based Interaction Feature Extraction Method (DIFEM)
=============================================================

This script implements the DIFEM approach for violence detection in videos.
It processes OpenPose JSON outputs and extracts the following features:

1. Joint Velocity Features:
   - Weighted velocity for 11 key joints (wrists, elbows, hips, knees, ankles, neck)
   - Mean, max, and variance across video frames

2. Proximity-based Features:
   - Closeness Factor: Spatial proximity between people
   - Wrist Overlap: Potential physical interactions

3. Classification:
   - Trains multiple ML classifiers (Random Forest, SVM, Logistic Regression, etc.)
   - Evaluates performance with accuracy, precision, recall, F1-score

Author: Himanshu
Date: 2024
"""

import os
import json
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the joints you want to include in the features
SELECTED_JOINTS = [
    "right_wrist", "left_wrist",
    "right_elbow", "left_elbow",
    "right_hip", "left_hip",
    "right_knee", "left_knee",
    "right_ankle", "left_ankle",
    "neck"
]

# Joint importance weights for velocity calculation
JOINT_WEIGHTS = {
    "right_wrist": 1.0,
    "left_wrist": 1.0,
    "right_elbow": 0.8,
    "left_elbow": 0.8,
    "right_hip": 1.0,
    "left_hip": 1.0,
    "right_knee": 1.0,
    "left_knee": 1.0,
    "right_ankle": 1.0,
    "left_ankle": 1.0,
    "neck": 1.0
}

# OpenPose BODY_25 model joint names (25 keypoints)
JOINT_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
    "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: Tuple (x1, y1)
        point2: Tuple (x2, y2)
    
    Returns:
        float: Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_person_center(person_keypoints):
    """
    Calculate the center point of a person using hip keypoints.
    
    Args:
        person_keypoints: List of keypoint coordinates from OpenPose
    
    Returns:
        tuple: (center_x, center_y) or None if hips not detected
    """
    mid_hip_idx = JOINT_NAMES.index("mid_hip")
    mid_hip_x = person_keypoints[mid_hip_idx * 3]
    mid_hip_y = person_keypoints[mid_hip_idx * 3 + 1]
    
    if mid_hip_x == 0 and mid_hip_y == 0:
        return None
    
    return (mid_hip_x, mid_hip_y)


# ============================================================================
# FEATURE EXTRACTION - VERSION 1: BASIC VELOCITY FEATURES
# ============================================================================

def calculate_velocity_features(frame_data, joint_weights):
    """
    Calculate joint velocity features for a single frame.
    
    Args:
        frame_data: JSON data for one frame from OpenPose
        joint_weights: Dictionary of joint importance weights
    
    Returns:
        numpy.array: Velocity features for all people in the frame
    """
    frame_velocities = []

    for person in frame_data['people']:
        person_velocity = []
        prev_x, prev_y = None, None

        for joint_name in SELECTED_JOINTS:
            joint_idx = JOINT_NAMES.index(joint_name)
            curr_x = person['pose_keypoints_2d'][joint_idx * 3]
            curr_y = person['pose_keypoints_2d'][joint_idx * 3 + 1]

            if prev_x is not None and prev_y is not None:
                weight = joint_weights.get(joint_name, 1.0)
                joint_velocity = weight * calculate_distance((curr_x, curr_y), (prev_x, prev_y))
            else:
                joint_velocity = 0

            person_velocity.append(joint_velocity)
            prev_x, prev_y = curr_x, curr_y

        frame_velocities.append(person_velocity)

    return np.array(frame_velocities)


def process_video_basic(video_path, joint_weights):
    """
    Process a video and extract basic velocity features.
    
    Args:
        video_path: Path to directory containing OpenPose JSON files
        joint_weights: Dictionary of joint importance weights
    
    Returns:
        tuple: (video_frames, mean_velocity, max_velocity, variance_velocity)
    """
    video_frames = []
    max_velocity = 0
    all_velocities = []

    for json_file in sorted(os.listdir(video_path)):
        file_path = os.path.join(video_path, json_file)
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        # Skip frames with no detected people
        if 'people' not in frame_data or not frame_data['people']:
            continue

        frame_velocities = calculate_velocity_features(frame_data, joint_weights)
        video_frames.append(frame_velocities)

        max_frame_velocity = np.max(frame_velocities)
        max_velocity = max(max_velocity, max_frame_velocity)
        all_velocities.extend(frame_velocities.flatten())

    # Calculate statistics
    mean_velocity = np.mean(all_velocities) if all_velocities else 0
    variance_velocity = np.var(all_velocities) if all_velocities else 0

    return video_frames, mean_velocity, max_velocity, variance_velocity


# ============================================================================
# FEATURE EXTRACTION - VERSION 2: ENHANCED WITH PROXIMITY FEATURES
# ============================================================================

def calculate_enhanced_features(frame_data, joint_weights):
    """
    Calculate enhanced features including velocity, proximity, and wrist overlap.
    
    Args:
        frame_data: JSON data for one frame from OpenPose
        joint_weights: Dictionary of joint importance weights
    
    Returns:
        tuple: (frame_velocities, closeness_factors, wrist_overlaps)
    """
    frame_velocities = []
    closeness_factors = []
    wrist_overlaps = []

    people = frame_data['people']
    num_people = len(people)

    # Calculate velocity features for each person
    for person in people:
        person_velocity = []
        prev_x, prev_y = None, None

        for joint_name in SELECTED_JOINTS:
            joint_idx = JOINT_NAMES.index(joint_name)
            curr_x = person['pose_keypoints_2d'][joint_idx * 3]
            curr_y = person['pose_keypoints_2d'][joint_idx * 3 + 1]

            if prev_x is not None and prev_y is not None:
                weight = joint_weights.get(joint_name, 1.0)
                joint_velocity = weight * calculate_distance((curr_x, curr_y), (prev_x, prev_y))
            else:
                joint_velocity = 0

            person_velocity.append(joint_velocity)
            prev_x, prev_y = curr_x, curr_y

        frame_velocities.append(person_velocity)

    # Calculate proximity features (closeness factor and wrist overlap)
    if num_people > 1:
        for i in range(num_people):
            for j in range(i + 1, num_people):
                person1 = people[i]['pose_keypoints_2d']
                person2 = people[j]['pose_keypoints_2d']

                # Closeness factor (distance between centers)
                center1 = get_person_center(person1)
                center2 = get_person_center(person2)

                if center1 and center2:
                    distance = calculate_distance(center1, center2)
                    closeness_factor = 1 / (distance + 1e-5)  # Avoid division by zero
                    closeness_factors.append(closeness_factor)

                # Wrist overlap
                right_wrist_idx = JOINT_NAMES.index("right_wrist")
                left_wrist_idx = JOINT_NAMES.index("left_wrist")

                p1_right_wrist = (person1[right_wrist_idx * 3], person1[right_wrist_idx * 3 + 1])
                p1_left_wrist = (person1[left_wrist_idx * 3], person1[left_wrist_idx * 3 + 1])
                p2_right_wrist = (person2[right_wrist_idx * 3], person2[right_wrist_idx * 3 + 1])
                p2_left_wrist = (person2[left_wrist_idx * 3], person2[left_wrist_idx * 3 + 1])

                # Check for wrist overlaps (distance threshold = 50 pixels)
                overlap_threshold = 50
                overlaps = [
                    calculate_distance(p1_right_wrist, p2_right_wrist) < overlap_threshold,
                    calculate_distance(p1_right_wrist, p2_left_wrist) < overlap_threshold,
                    calculate_distance(p1_left_wrist, p2_right_wrist) < overlap_threshold,
                    calculate_distance(p1_left_wrist, p2_left_wrist) < overlap_threshold
                ]
                wrist_overlap = 1 if any(overlaps) else 0
                wrist_overlaps.append(wrist_overlap)

    return np.array(frame_velocities), closeness_factors, wrist_overlaps


def process_video_enhanced(video_path, joint_weights):
    """
    Process a video and extract enhanced features including proximity metrics.
    
    Args:
        video_path: Path to directory containing OpenPose JSON files
        joint_weights: Dictionary of joint importance weights
    
    Returns:
        tuple: (video_frames, mean_velocity, max_velocity, variance_velocity,
                mean_closeness_factor, variance_closeness_factor,
                mean_wrist_overlap, variance_wrist_overlap)
    """
    video_frames = []
    max_velocity = 0
    all_velocities = []
    all_closeness_factors = []
    all_wrist_overlaps = []

    for json_file in sorted(os.listdir(video_path)):
        file_path = os.path.join(video_path, json_file)
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        if 'people' not in frame_data or not frame_data['people']:
            continue

        frame_velocities, frame_closeness, wrist_overlaps = calculate_enhanced_features(
            frame_data, joint_weights
        )
        video_frames.append(frame_velocities)

        max_frame_velocity = np.max(frame_velocities)
        max_velocity = max(max_velocity, max_frame_velocity)

        all_velocities.extend(frame_velocities.flatten())
        all_closeness_factors.extend(frame_closeness)
        all_wrist_overlaps.extend(wrist_overlaps)

    # Calculate statistics
    mean_velocity = np.mean(all_velocities) if all_velocities else 0
    variance_velocity = np.var(all_velocities) if all_velocities else 0
    mean_closeness = np.mean(all_closeness_factors) if all_closeness_factors else 0
    variance_closeness = np.var(all_closeness_factors) if all_closeness_factors else 0
    mean_wrist = np.mean(all_wrist_overlaps) if all_wrist_overlaps else 0
    variance_wrist = np.var(all_wrist_overlaps) if all_wrist_overlaps else 0

    return (
        video_frames, mean_velocity, max_velocity, variance_velocity,
        mean_closeness, variance_closeness, mean_wrist, variance_wrist
    )


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def save_dataset(dataset, save_path):
    """Save dataset to disk using pickle."""
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to: {save_path}")


def load_dataset(dataset_path):
    """Load dataset from disk."""
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def process_dataset(dataset_path, output_path, use_enhanced=True):
    """
    Process entire dataset and extract features.
    
    Args:
        dataset_path: Path to root directory containing Fight/NonFight folders
        output_path: Path to save processed features
        use_enhanced: If True, use enhanced features; else use basic features
    
    Returns:
        list: Processed dataset
    """
    # Load existing dataset if available
    if os.path.exists(output_path):
        dataset = load_dataset(output_path)
        processed_videos = set([os.path.basename(item[0][0][0]) for item in dataset])
        print(f"Loaded existing dataset with {len(dataset)} videos")
    else:
        dataset = []
        processed_videos = set()

    start_time = time.time()
    video_count = 0

    for category in ['Fight', 'NonFight']:
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category path not found: {category_path}")
            continue

        for video_folder in os.listdir(category_path):
            video_path = os.path.join(category_path, video_folder)
            
            if os.path.isdir(video_path) and video_folder not in processed_videos:
                # Process video
                if use_enhanced:
                    video_data = process_video_enhanced(video_path, JOINT_WEIGHTS)
                else:
                    video_data = process_video_basic(video_path, JOINT_WEIGHTS)
                
                label = 1 if category == "Fight" else 0
                dataset.append((video_data, label))

                # Save progress periodically
                if video_count % 10 == 0 and video_count > 0:
                    save_dataset(dataset, output_path)
                    elapsed = time.time() - start_time
                    print(f"Processed {video_count} videos ({category}). Elapsed: {elapsed:.2f}s")

                video_count += 1

    # Final save
    save_dataset(dataset, output_path)
    total_time = time.time() - start_time
    print(f"Completed! Total videos: {video_count}, Time: {total_time:.2f}s")

    return dataset


# ============================================================================
# FEATURE PREPARATION FOR CLASSIFICATION
# ============================================================================

def extract_features_labels_basic(dataset):
    """Extract features and labels for basic (3-feature) model."""
    features = {
        'mean_velocity': {1: [], 0: []},
        'max_velocity': {1: [], 0: []},
        'variance': {1: [], 0: []}
    }

    for item in dataset:
        mean_vel, max_vel, var_vel = item[0][1:4]
        label = item[1]
        
        features['mean_velocity'][label].append(mean_vel)
        features['max_velocity'][label].append(max_vel)
        features['variance'][label].append(var_vel)

    return features['mean_velocity'], features['max_velocity'], features['variance']


def extract_features_labels_enhanced(dataset):
    """Extract features and labels for enhanced (7-feature) model."""
    features = {
        'mean_velocity': {1: [], 0: []},
        'max_velocity': {1: [], 0: []},
        'variance_velocity': {1: [], 0: []},
        'mean_closeness_factor': {1: [], 0: []},
        'variance_closeness_factor': {1: [], 0: []},
        'mean_wrist_overlap': {1: [], 0: []},
        'variance_wrist_overlap': {1: [], 0: []}
    }

    for item in dataset:
        mean_vel, max_vel, var_vel = item[0][1:4]
        mean_close, var_close = item[0][4:6]
        mean_wrist, var_wrist = item[0][6:8]
        label = item[1]

        features['mean_velocity'][label].append(mean_vel)
        features['max_velocity'][label].append(max_vel)
        features['variance_velocity'][label].append(var_vel)
        features['mean_closeness_factor'][label].append(mean_close)
        features['variance_closeness_factor'][label].append(var_close)
        features['mean_wrist_overlap'][label].append(mean_wrist)
        features['variance_wrist_overlap'][label].append(var_wrist)

    return (features['mean_velocity'], features['max_velocity'], features['variance_velocity'],
            features['mean_closeness_factor'], features['variance_closeness_factor'],
            features['mean_wrist_overlap'], features['variance_wrist_overlap'])


def prepare_dataset_basic(mean_fight, mean_nonfight, max_fight, max_nonfight, 
                         var_fight, var_nonfight):
    """Prepare dataset for training (basic 3-feature model)."""
    X = np.concatenate([mean_fight, mean_nonfight, max_fight, max_nonfight, 
                       var_fight, var_nonfight])
    y = np.concatenate([[1] * len(mean_fight) + [0] * len(mean_nonfight)])
    return X.reshape(-1, 3), y


def prepare_dataset_enhanced(mean_fight, mean_nonfight, max_fight, max_nonfight,
                            var_fight, var_nonfight, mean_close_fight, mean_close_nonfight,
                            var_close_fight, var_close_nonfight, mean_wrist_fight,
                            mean_wrist_nonfight, var_wrist_fight, var_wrist_nonfight):
    """Prepare dataset for training (enhanced 7-feature model)."""
    X = np.concatenate([
        mean_fight, mean_nonfight, max_fight, max_nonfight, var_fight, var_nonfight,
        mean_close_fight, mean_close_nonfight, var_close_fight, var_close_nonfight,
        mean_wrist_fight, mean_wrist_nonfight, var_wrist_fight, var_wrist_nonfight
    ])
    y = np.concatenate([[1] * len(mean_fight) + [0] * len(mean_nonfight)])
    return X.reshape(-1, 7), y


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_classifier(X_train, y_train, classifier_name='random_forest'):
    """
    Train a classifier on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        classifier_name: Name of classifier to use
    
    Returns:
        Trained model
    """
    # Impute missing values
    X_train = SimpleImputer(strategy='mean').fit_transform(X_train)
    
    # Select classifier
    classifiers = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'adaboost': AdaBoostClassifier(random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    
    model = classifiers.get(classifier_name, RandomForestClassifier())
    model.fit(X_train, y_train)
    
    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a trained classifier.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (accuracy, classification_report, confusion_matrix)
    """
    X_test = SimpleImputer(strategy='mean').fit_transform(X_test)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['NonFight', 'Fight'])
    confusion = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, confusion


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    TRAIN_DATASET_PATH = './data/train'
    VAL_DATASET_PATH = './data/val'
    
    # Output paths for basic features
    OUTPUT_TRAIN_BASIC = './outputs/train_features_basic.pkl'
    OUTPUT_VAL_BASIC = './outputs/val_features_basic.pkl'
    
    # Output paths for enhanced features
    OUTPUT_TRAIN_ENHANCED = './outputs/train_features_enhanced.pkl'
    OUTPUT_VAL_ENHANCED = './outputs/val_features_enhanced.pkl'
    
    # Create outputs directory
    os.makedirs('./outputs', exist_ok=True)
    
    # Process datasets
    print("=" * 70)
    print("PROCESSING DATASETS WITH ENHANCED FEATURES")
    print("=" * 70)
    
    print("\nProcessing training dataset...")
    train_dataset = process_dataset(TRAIN_DATASET_PATH, OUTPUT_TRAIN_ENHANCED, use_enhanced=True)
    
    print("\nProcessing validation dataset...")
    val_dataset = process_dataset(VAL_DATASET_PATH, OUTPUT_VAL_ENHANCED, use_enhanced=True)
    
    # Extract features
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIERS")
    print("=" * 70)
    
    (train_mean, train_max, train_var, train_mean_close, train_var_close,
     train_mean_wrist, train_var_wrist) = extract_features_labels_enhanced(train_dataset)
    
    X_train, y_train = prepare_dataset_enhanced(
        train_mean[1], train_mean[0], train_max[1], train_max[0],
        train_var[1], train_var[0], train_mean_close[1], train_mean_close[0],
        train_var_close[1], train_var_close[0], train_mean_wrist[1], train_mean_wrist[0],
        train_var_wrist[1], train_var_wrist[0]
    )
    
    (test_mean, test_max, test_var, test_mean_close, test_var_close,
     test_mean_wrist, test_var_wrist) = extract_features_labels_enhanced(val_dataset)
    
    X_test, y_test = prepare_dataset_enhanced(
        test_mean[1], test_mean[0], test_max[1], test_max[0],
        test_var[1], test_var[0], test_mean_close[1], test_mean_close[0],
        test_var_close[1], test_var_close[0], test_mean_wrist[1], test_mean_wrist[0],
        test_var_wrist[1], test_var_wrist[0]
    )
    
    # Train and evaluate multiple classifiers
    classifiers_to_test = [
        'random_forest', 'svm', 'logistic_regression',
        'gradient_boosting', 'adaboost', 'decision_tree'
    ]
    
    print("\nTraining and evaluating classifiers...")
    print("-" * 70)
    
    for clf_name in classifiers_to_test:
        print(f"\n{clf_name.upper().replace('_', ' ')}")
        print("-" * 70)
        
        model = train_classifier(X_train, y_train, clf_name)
        accuracy, report, confusion = evaluate_classifier(model, X_test, y_test)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"\nConfusion Matrix:\n{confusion}\n")
    
    print("=" * 70)
    print("COMPLETED!")
    print("=" * 70)
