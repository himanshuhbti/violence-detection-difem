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

# Define the joints you want to include in the features
selected_joints = ["right_wrist", "left_wrist", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle", "neck"]

joint_weights = {
    "right_wrist": 1,
    "left_wrist": 1,
    "right_elbow": 0.8,
    "left_elbow": 0.8,
    "right_hip": 1,
    "left_hip": 1,
    "right_knee": 1,
    "left_knee": 1,
    "right_ankle": 1,
    "left_ankle": 1,
    "neck": 1.0
}

joint_names = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
    "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"
]

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_features(curr_frame_data, joint_weights):
    frame_velocities = []

    for curr_person in curr_frame_data['people']:
        person_velocity = []
        prev_x, prev_y = None, None  # Initialize previous coordinates for each person

        for joint_name in selected_joints:
            i = joint_names.index(joint_name)
            curr_x, curr_y = curr_person['pose_keypoints_2d'][i * 3], curr_person['pose_keypoints_2d'][i * 3 + 1]

            if prev_x is not None and prev_y is not None:  # Calculate joint velocity
                weight = joint_weights.get(joint_name, 1.0)
                joint_velocity = weight * calculate_distance((curr_x, curr_y), (prev_x, prev_y))
            else:
                joint_velocity = 0

            person_velocity.append(joint_velocity)

            prev_x, prev_y = curr_x, curr_y

        frame_velocities.append(person_velocity)

    return np.array(frame_velocities)

def process_video(video_path, joint_weights):
    video_frames = []
    max_velocity = 0  # Variable to store max velocity for the video
    all_velocities = []  # To store all velocities for later calculation of mean and variance

    for json_file in sorted(os.listdir(video_path)):  # Sort files to process in order
        file_path = os.path.join(video_path, json_file)
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        # Skip processing if no person is detected in the frame
        if 'people' not in frame_data or not frame_data['people']:
            continue

        frame_velocities = calculate_features(frame_data, joint_weights)
        video_frames.append(frame_velocities)

        # Calculate max velocity for the current frame
        max_frame_velocity = np.max(frame_velocities)
        max_velocity = max(max_velocity, max_frame_velocity)

        # Store all velocities for later calculation of mean and variance
        all_velocities.extend(frame_velocities)

    # Calculate mean and variance of velocities for the whole video
    mean_velocity = np.mean(all_velocities)
    variance_velocity = np.var(all_velocities)

    return video_frames, mean_velocity, max_velocity, variance_velocity

def process_dataset(dataset_path, output_dataset_path):
    # Load the existing dataset if it exists
    if os.path.exists(output_dataset_path):
        with open(output_dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        processed_videos = set([os.path.basename(item[0][0][0]) for item in dataset])  # Extract processed videos
    else:
        dataset = []
        processed_videos = set()

    start_time = time.time()
    video_count = 0

    for category in ['Fight', 'NonFight']:
        category_path = os.path.join(dataset_path, category)
        for video_folder in os.listdir(category_path):
            video_folder_path = os.path.join(category_path, video_folder)
            if os.path.isdir(video_folder_path) and video_folder not in processed_videos:
                video_frames, mean_velocity, max_velocity, variance_velocity = process_video(video_folder_path, joint_weights)
                label = 1 if category == "Fight" else 0  # 1 for Fight, 0 for NonFight
                dataset.append(([video_frames, mean_velocity, max_velocity, variance_velocity], label))

                # Save progress periodically
                if video_count % 10 == 0:
                    save_dataset_to_drive(dataset, output_dataset_path)
                    print(f"Processed {video_count} videos. Time elapsed: {time.time() - start_time:.2f} seconds")

                video_count += 1

    # Final save after all processing is complete
    save_dataset_to_drive(dataset, output_dataset_path)
    print(f"Completed processing. Total time: {time.time() - start_time:.2f} seconds")

    return dataset

# Function to save dataset
def save_dataset_to_drive(dataset, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved successfully to {save_path}")

# Example usage
train_dataset_path = './train'
val_dataset_path = './val'

output_train_dataset_path = './last_train_features.pkl'
output_val_dataset_path = './last_val_features.pkl'

# Process both train and validation datasets
process_dataset(train_dataset_path, output_train_dataset_path)
process_dataset(val_dataset_path, output_val_dataset_path)

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset from a given path
def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

# Extract features and labels from the dataset
def extract_features_labels(dataset):
    features = {'mean_velocity': {1: [], 0: []},
                'max_velocity': {1: [], 0: []},
                'variance': {1: [], 0: []}}

    for item in dataset:
        mean_velocity, max_velocity, variance = item[0][1:4]
        label = item[1]  # Label is now expected to be numeric (1 for "fight", 0 for "nonfight")
        features['mean_velocity'][label].append(mean_velocity)
        features['max_velocity'][label].append(max_velocity)
        features['variance'][label].append(variance)

    return features['mean_velocity'], features['max_velocity'], features['variance']

# Prepare the dataset for model training/testing
def prepare_dataset(mean_fight, mean_nonfight, max_fight, max_nonfight, var_fight, var_nonfight):
    X = np.concatenate([mean_fight, mean_nonfight, max_fight, max_nonfight, var_fight, var_nonfight])
    y = np.concatenate([[1] * len(mean_fight) + [0] * len(mean_nonfight)])  # 1 for fight, 0 for nonfight
    return X.reshape(-1, 3), y

# Train the Random Forest model
def train_model(X_train, y_train):
    X_train = SimpleImputer(strategy='mean').fit_transform(X_train)
    model = RandomForestClassifier().fit(X_train, y_train)
    return model

# Evaluate the trained model
def evaluate_model(model, X_test, y_test):
    X_test = SimpleImputer(strategy='mean').fit_transform(X_test)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

# Example usage
train_dataset_path = './last_train_features.pkl'
test_dataset_path = './last_val_features.pkl'

train_dataset = load_dataset(train_dataset_path)
test_dataset = load_dataset(test_dataset_path)

train_mean, train_max, train_var = extract_features_labels(train_dataset)
X_train, y_train = prepare_dataset(train_mean[1], train_mean[0],
                                   train_max[1], train_max[0],
                                   train_var[1], train_var[0])

test_mean, test_max, test_var = extract_features_labels(test_dataset)
X_test, y_test = prepare_dataset(test_mean[1], test_mean[0],
                                 test_max[1], test_max[0],
                                 test_var[1], test_var[0])

model = train_model(X_train, y_train)
accuracy, report, confusion = evaluate_model(model, X_test, y_test)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

"""With Proximity"""

import os
import json
import numpy as np
import pickle
import time

# Define the joints you want to include in the features
selected_joints = ["right_wrist", "left_wrist", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle", "neck"]

joint_weights = {
    "right_wrist": 1,
    "left_wrist": 1,
    "right_elbow": 0.8,
    "left_elbow": 0.8,
    "right_hip": 1,
    "left_hip": 1,
    "right_knee": 1,
    "left_knee": 1,
    "right_ankle": 1,
    "left_ankle": 1,
    "neck": 1.0
}

joint_names = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
    "right_big_toe", "left_small_toe", "right_small_toe", "left_heel", "right_heel"
]

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_closeness_factor(person_data):
    keypoints = person_data['pose_keypoints_2d']
    num_keypoints = len(keypoints) // 3
    total_distance = 0
    count = 0

    # Calculate average distance between all pairs of keypoints
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x1, y1 = keypoints[i * 3], keypoints[i * 3 + 1]
            x2, y2 = keypoints[j * 3], keypoints[j * 3 + 1]
            distance = calculate_distance((x1, y1), (x2, y2))
            total_distance += distance
            count += 1

    if count == 0:
        return 0  # No keypoints found

    return total_distance / count

def check_wrist_overlap(person1_data, person2_data):
    wrist_indices = [4, 7]  # Indices of right and left wrists

    # Get bounding box for person 2's body
    x_min_p2 = min(person2_data['pose_keypoints_2d'][::3])
    x_max_p2 = max(person2_data['pose_keypoints_2d'][::3])
    y_min_p2 = min(person2_data['pose_keypoints_2d'][1::3])
    y_max_p2 = max(person2_data['pose_keypoints_2d'][1::3])

    # Check if any wrist coordinates of person 1 are within bounding box of person 2
    wrist_overlap_count = 0
    for idx in wrist_indices:
        x_wrist, y_wrist = person1_data['pose_keypoints_2d'][idx * 3], person1_data['pose_keypoints_2d'][idx * 3 + 1]
        if x_min_p2 <= x_wrist <= x_max_p2 and y_min_p2 <= y_wrist <= y_max_p2:
            wrist_overlap_count += 1

    return wrist_overlap_count

def calculate_features(curr_frame_data, joint_weights):
    frame_velocities = []
    frame_closeness_factors = []
    wrist_overlaps = []

    for i, curr_person in enumerate(curr_frame_data['people']):
        person_velocity = []
        prev_x, prev_y = None, None  # Initialize previous coordinates for each person

        for joint_name in selected_joints:
            i = joint_names.index(joint_name)
            curr_x, curr_y = curr_person['pose_keypoints_2d'][i * 3], curr_person['pose_keypoints_2d'][i * 3 + 1]

            if prev_x is not None and prev_y is not None:  # Calculate joint velocity
                weight = joint_weights.get(joint_name, 1.0)
                joint_velocity = weight * calculate_distance((curr_x, curr_y), (prev_x, prev_y))
            else:
                joint_velocity = 0

            person_velocity.append(joint_velocity)
            prev_x, prev_y = curr_x, curr_y

        # Calculate closeness factor
        closeness_factor = calculate_closeness_factor(curr_person)
        frame_closeness_factors.append(closeness_factor)

        # Calculate wrist overlap with other persons in the frame
        wrist_overlap_count = 0
        for j, other_person in enumerate(curr_frame_data['people']):
            if i != j:
                wrist_overlap_count += check_wrist_overlap(curr_person, other_person)

        wrist_overlaps.append(wrist_overlap_count)

        frame_velocities.append(person_velocity)

    return np.array(frame_velocities), np.array(frame_closeness_factors), np.array(wrist_overlaps)

def process_video(video_path, joint_weights):
    video_frames = []
    max_velocity = 0
    all_velocities = []
    all_closeness_factors = []
    all_wrist_overlaps = []

    for json_file in sorted(os.listdir(video_path)):  # Sort files to process in order
        file_path = os.path.join(video_path, json_file)
        with open(file_path, 'r') as f:
            frame_data = json.load(f)

        # Skip processing if no person is detected in the frame
        if 'people' not in frame_data or not frame_data['people']:
            continue

        frame_velocities, frame_closeness_factors, wrist_overlaps = calculate_features(frame_data, joint_weights)
        video_frames.append(frame_velocities)

        # Calculate max velocity for the current frame
        max_frame_velocity = np.max(frame_velocities)
        max_velocity = max(max_velocity, max_frame_velocity)

        # Store all velocities and new features for later calculation of mean and variance
        all_velocities.extend(frame_velocities)
        all_closeness_factors.extend(frame_closeness_factors)
        all_wrist_overlaps.extend(wrist_overlaps)

    # Calculate mean and variance of velocities and new features for the whole video
    mean_velocity = np.mean(all_velocities)
    variance_velocity = np.var(all_velocities)
    mean_closeness_factor = np.mean(all_closeness_factors)
    variance_closeness_factor = np.var(all_closeness_factors)
    mean_wrist_overlap = np.mean(all_wrist_overlaps)
    variance_wrist_overlap = np.var(all_wrist_overlaps)

    return (
        video_frames,
        mean_velocity, max_velocity, variance_velocity,
        mean_closeness_factor, variance_closeness_factor,
        mean_wrist_overlap, variance_wrist_overlap
    )

def process_dataset(dataset_path, output_dataset_path):
    if os.path.exists(output_dataset_path):
        with open(output_dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        processed_videos = set([os.path.basename(item[0][0][0]) for item in dataset])  # Extract processed videos
    else:
        dataset = []
        processed_videos = set()

    start_time = time.time()
    video_count = 0

    for category in ['Fight', 'NonFight']:
        category_path = os.path.join(dataset_path, category)
        for video_folder in os.listdir(category_path):
            video_folder_path = os.path.join(category_path, video_folder)
            if os.path.isdir(video_folder_path) and video_folder not in processed_videos:
                video_data = process_video(video_folder_path, joint_weights)
                label = 1 if category == "Fight" else 0  # 1 for Fight, 0 for NonFight
                dataset.append((video_data, label))

                # Save progress periodically
                if video_count % 10 == 0:
                    save_dataset_to_drive(dataset, output_dataset_path)
                    print(f"Processed {video_count} videos. Time elapsed: {time.time() - start_time:.2f} seconds")

                video_count += 1

    # Final save after all processing is complete
    save_dataset_to_drive(dataset, output_dataset_path)
    print(f"Completed processing. Total time: {time.time() - start_time:.2f} seconds")

    return dataset

# Function to save dataset
def save_dataset_to_drive(dataset, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved successfully to {save_path}")

# Example usage
train_dataset_path = './train'
val_dataset_path = './val'

output_train_dataset_path = './proximity_last_train_features.pkl'
output_val_dataset_path = './proximity_last_val_features.pkl'

# Process both train and validation datasets
process_dataset(train_dataset_path, output_train_dataset_path)
process_dataset(val_dataset_path, output_val_dataset_path)

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset from a given path
def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

# Extract features and labels from the dataset
def extract_features_labels(dataset):
    features = {'mean_velocity': {1: [], 0: []},
                'max_velocity': {1: [], 0: []},
                'variance_velocity': {1: [], 0: []},
                'mean_closeness_factor': {1: [], 0: []},
                'variance_closeness_factor': {1: [], 0: []},
                'mean_wrist_overlap': {1: [], 0: []},
                'variance_wrist_overlap': {1: [], 0: []}}

    for item in dataset:
        mean_velocity, max_velocity, variance_velocity = item[0][1:4]
        mean_closeness_factor, variance_closeness_factor = item[0][4:6]
        mean_wrist_overlap, variance_wrist_overlap = item[0][6:8]
        label = item[1]  # Label is numeric (1 for "fight", 0 for "nonfight")

        features['mean_velocity'][label].append(mean_velocity)
        features['max_velocity'][label].append(max_velocity)
        features['variance_velocity'][label].append(variance_velocity)
        features['mean_closeness_factor'][label].append(mean_closeness_factor)
        features['variance_closeness_factor'][label].append(variance_closeness_factor)
        features['mean_wrist_overlap'][label].append(mean_wrist_overlap)
        features['variance_wrist_overlap'][label].append(variance_wrist_overlap)

    return (features['mean_velocity'], features['max_velocity'], features['variance_velocity'],
            features['mean_closeness_factor'], features['variance_closeness_factor'],
            features['mean_wrist_overlap'], features['variance_wrist_overlap'])

# Prepare the dataset for model training/testing
def prepare_dataset(mean_fight, mean_nonfight, max_fight, max_nonfight,
                    var_fight, var_nonfight, mean_close_fight, mean_close_nonfight,
                    var_close_fight, var_close_nonfight, mean_wrist_fight,
                    mean_wrist_nonfight, var_wrist_fight, var_wrist_nonfight):
    X = np.concatenate([mean_fight, mean_nonfight, max_fight, max_nonfight, var_fight, var_nonfight,
                        mean_close_fight, mean_close_nonfight, var_close_fight, var_close_nonfight,
                        mean_wrist_fight, mean_wrist_nonfight, var_wrist_fight, var_wrist_nonfight])
    y = np.concatenate([[1] * len(mean_fight) + [0] * len(mean_nonfight)])  # 1 for fight, 0 for nonfight
    return X.reshape(-1, 7), y  # Reshape with 7 features now

# Train the Random Forest model
def train_model(X_train, y_train):
    X_train = SimpleImputer(strategy='mean').fit_transform(X_train)
    model = RandomForestClassifier().fit(X_train, y_train)
    return model

# Evaluate the trained model
def evaluate_model(model, X_test, y_test):
    X_test = SimpleImputer(strategy='mean').fit_transform(X_test)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

# Example usage
train_dataset_path = './proximity_last_train_features.pkl'
test_dataset_path = './proximity_last_val_features.pkl'

train_dataset = load_dataset(train_dataset_path)
test_dataset = load_dataset(test_dataset_path)

(train_mean, train_max, train_var, train_mean_close, train_var_close,
 train_mean_wrist, train_var_wrist) = extract_features_labels(train_dataset)

X_train, y_train = prepare_dataset(train_mean[1], train_mean[0],
                                   train_max[1], train_max[0],
                                   train_var[1], train_var[0],
                                   train_mean_close[1], train_mean_close[0],
                                   train_var_close[1], train_var_close[0],
                                   train_mean_wrist[1], train_mean_wrist[0],
                                   train_var_wrist[1], train_var_wrist[0])

(test_mean, test_max, test_var, test_mean_close, test_var_close,
 test_mean_wrist, test_var_wrist) = extract_features_labels(test_dataset)

X_test, y_test = prepare_dataset(test_mean[1], test_mean[0],
                                 test_max[1], test_max[0],
                                 test_var[1], test_var[0],
                                 test_mean_close[1], test_mean_close[0],
                                 test_var_close[1], test_var_close[0],
                                 test_mean_wrist[1], test_mean_wrist[0],
                                 test_var_wrist[1], test_var_wrist[0])

model = train_model(X_train, y_train)
accuracy, report, confusion = evaluate_model(model, X_test, y_test)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

