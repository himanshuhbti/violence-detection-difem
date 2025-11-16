"""
OpenPose Feature Extraction for Violence Detection
===================================================

This script extracts human pose keypoints from video datasets using CMU's OpenPose.
It processes videos from the RWF-2000 dataset and generates JSON files containing
pose keypoints for each frame.

Requirements:
- Google Colab environment (recommended for GPU support)
- OpenPose pretrained models
- RWF-2000 dataset videos

Author: Himanshu
Date: 2024
"""


"""
OpenPose Installation (for Google Colab)
=========================================

Run these commands in a Colab cell to install OpenPose:

# Install CMake
!wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
!tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local

# Clone OpenPose repository
!git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Fix CMakeLists.txt for Caffe version
!sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt

# Install system dependencies
!apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev

# Build OpenPose with Python support
!cd openpose && rm -rf build || true && mkdir build && cd build && cmake -DBUILD_PYTHON=ON .. && make -j`nproc`

# Download pretrained models
# You need to download these models and place them in the correct directories:
# - pose_iter_116000.caffemodel → models/face/
# - pose_iter_102000.caffemodel → models/hand/
# - pose_iter_584000.caffemodel → models/pose/body_25/
# - pose_iter_440000.caffemodel → models/pose/coco/
# - pose_iter_160000.caffemodel → models/pose/mpi/
"""

import os

# Configuration
# Update these paths according to your setup
VIDEO_FOLDER = "./data/videos/test/Fight"  # Path to input videos
OPENPOSE_PATH = "./openpose/build/examples/openpose/openpose.bin"  # Path to OpenPose binary
OUTPUT_FOLDER = "./outputs/openpose_json/val/Fight"  # Path for JSON outputs
OPENPOSE_PARAMS = "--display 0 --model_pose BODY_25 --render_pose 0"  # OpenPose parameters

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processed_count = 0

# Loop through all video files in the folder
for file_name in os.listdir(VIDEO_FOLDER):
    processed_count += 1
    print(f"Processing video {processed_count}: {file_name}")
    
    # Check if the file is a video
    if file_name.endswith((".avi", ".mp4", ".mkv")):
        # Construct full paths
        video_path = os.path.join(VIDEO_FOLDER, file_name)
        json_output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(file_name)[0])

        # Check if already processed (skip if JSON files exist)
        if os.path.exists(json_output_path):
            if any(fname.endswith('.json') for fname in os.listdir(json_output_path)):
                print(f"  → Skipping (already processed): {video_path}")
                continue

        # Create output directory for this video
        os.makedirs(json_output_path, exist_ok=True)

        print(f"  → Processing: {video_path}")
        print(f"  → Output: {json_output_path}")

        # Construct and execute OpenPose command
        command = f"cd openpose && ./build/examples/openpose/openpose.bin --video {video_path} --write_json {json_output_path} {OPENPOSE_PARAMS}"
        os.system(command)
        
print(f"\nCompleted! Processed {processed_count} videos.")

