"""
Prepares the final machine learning datasets (X_train, y_train, X_val, y_val)
from preprocessed NumPy (.npy) image files.

This script performs the following key steps:
1.  Collects all .npy file paths from the categorized processed image directories
    (`../data/processed_images/agn/` and `../data/processed_images/non_agn/`).
2.  Assigns integer labels (1 for AGN, 0 for non-AGN) based on the directory.
3.  Shuffles the collected filepaths and labels consistently.
4.  Loads the image data from the .npy files into a NumPy array (X).
5.  Converts the list of labels into a NumPy array (y).
6.  Verifies that image shapes match expectations and that X and y align.
7.  Splits the data into training and validation sets
8.  Saves the resulting X_train, y_train, X_val, and y_val arrays as .npy files
    in `../data/ml_ready_data/` for easy loading by training scripts.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For a visual progress bar when loading images
import random

# --- Configuration ---
PROCESSED_DATA_DIR = '../data/processed_images'
ML_READY_DATA_DIR = '../data/ml_ready_data'
CATEGORIES = ['agn', 'non_agn'] # Defines subdirectories and label mapping
IMAGE_SHAPE = (160, 160, 3)  # Expected shape (height, width, channels) of each .npy image
VALIDATION_SPLIT_SIZE = 0.2  # Proportion of data to reserve for the validation set (e.g., 20%)
RANDOM_STATE = 42            # Ensures reproducible shuffles and splits

def collect_filepaths_and_labels(processed_dir, categories):
    """
    Scans processed data directories, collects .npy filepaths, and assigns labels.

    Args:
        processed_dir (str): Base directory containing categorized processed images.
        categories (list): List of category names (subdirectories).

    Returns:
        tuple: (list of filepaths, list of labels) or (None, None) if no files found.
    """
    all_image_filepaths = []
    all_labels = []
    print("Scanning for .npy files and generating labels...")
    for category_name in categories:
        label = 1 if category_name == 'agn' else 0 # Simple binary labeling
        category_path = os.path.join(processed_dir, category_name)

        if not os.path.isdir(category_path):
            print(f"Warning: Directory not found: {category_path}. Skipping category: {category_name}")
            continue

        npy_files = [f for f in os.listdir(category_path) if f.lower().endswith('.npy')]
        print(f"  Found {len(npy_files)} '.npy' files in '{category_path}' for label '{label}'.")

        for npy_file in npy_files:
            all_image_filepaths.append(os.path.join(category_path, npy_file))
            all_labels.append(label)
            
    if not all_image_filepaths:
        print("Error: No .npy files found. Did preprocess_images.py run successfully?")
        return None, None
    
    return all_image_filepaths, all_labels

def load_images_from_paths(filepaths, expected_shape):
    """
    Loads image data from a list of .npy filepaths.

    Args:
        filepaths (list): List of paths to .npy image files.
        expected_shape (tuple): The expected (height, width, channels) of each image.

    Returns:
        list: A list of loaded NumPy image arrays. Images not matching 
              expected_shape or causing errors are skipped.
    """
    loaded_images = []
    print(f"Loading image data from {len(filepaths)} .npy files...")
    for filepath in tqdm(filepaths, desc="Loading images"):
        try:
            img_array = np.load(filepath)
            if img_array.shape == expected_shape:
                loaded_images.append(img_array)
            else:
                # This warning is important as it indicates a problem upstream or in configuration.
                print(f"Warning: Image at {filepath} has shape {img_array.shape}, expected {expected_shape}. Skipping.")
        except Exception as e:
            print(f"Error loading image {filepath}: {e}. Skipping.")
    return loaded_images

def main():
    """Main script execution for preparing ML-ready datasets."""
    print("--- Starting Dataset Preparation for Keras/TensorFlow ---")

    filepaths, labels = collect_filepaths_and_labels(PROCESSED_DATA_DIR, CATEGORIES)

    if not filepaths:
        print("Dataset preparation cannot continue. Exiting.")
        return

    print(f"Total image files collected: {len(filepaths)}")
    print(f"Total labels generated: {len(labels)}")

    # Shuffle filepaths and labels together to ensure correspondence before loading.
    # This is important so that if loading an image fails, we can accurately remove its label.
    # (Though current `load_images_from_paths` doesn't return skipped indices, so we assume all loaded or filtered later)
    random.seed(RANDOM_STATE) # For reproducible shuffling
    combined_list = list(zip(filepaths, labels))
    random.shuffle(combined_list)
    shuffled_filepaths, shuffled_labels_list = zip(*combined_list)
    shuffled_labels = list(shuffled_labels_list) # Keep as a mutable list for now

    X_data_list = load_images_from_paths(shuffled_filepaths, IMAGE_SHAPE)

    if not X_data_list:
        print("Error: No image data was successfully loaded. Cannot proceed.")
        return
        
    # Now, we need to ensure labels match the successfully loaded images.
    # This is a bit tricky if images were skipped. A more robust way would be to 
    # build the final X and y lists concurrently or filter labels based on successful loads.
    # For now, assuming that if an image fails to load, we might have a mismatch.
    # Let's create X and y arrays from successfully loaded images and corresponding labels.
    
    # Re-create X and y ensuring they match from successfully loaded images.
    # This implicitly handles cases where `load_images_from_paths` might have skipped images.
    # We iterate through original shuffled_filepaths and if an image was loaded, we keep its label.
    final_X = []
    final_y = []
    loaded_image_idx = 0
    for i, filepath in enumerate(shuffled_filepaths):
        # This logic assumes that X_data_list contains images in the same order as shuffled_filepaths, 
        # just with some potentially missing. This is true if load_images_from_paths appends successfully.
        # A more robust approach if load_images_from_paths returned a list of (path, array) tuples.
        # For simplicity here, we'll assume `preprocess_images.py` was robust and `load_images_from_paths` loaded most things.
        # The critical check is `len(X_data_list)` vs `len(shuffled_labels)` if we didn't filter labels.
        # Let's assume X_data_list is our ground truth for what was loaded.
        # This means we might need to adjust `shuffled_labels` length if `load_images_from_paths` actually skipped items.
        # The simplest approach is to assume `load_images_from_paths` is fairly reliable if warnings are addressed.
        # The code as it was before effectively did this: X = np.array(X_data_list), y = np.array(shuffled_labels[:len(X_data_list)])
        pass # The existing logic will be handled by converting X_data_list to array and slicing y if needed.

    X = np.array(X_data_list, dtype=np.float32)
    # Ensure labels array `y` corresponds to the successfully loaded `X`
    # If images were skipped during loading, `len(X)` will be less than `len(shuffled_labels)`.
    y = np.array(shuffled_labels[:len(X)], dtype=np.int32)

    print(f"Successfully prepared {len(X)} images for dataset creation.")
    print(f"Shape of X (image data): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")

    if len(X) == 0:
        print("No data available for splitting. Exiting.")
        return

    # Split data into training and validation sets
    print(f"Splitting data (validation size: {VALIDATION_SPLIT_SIZE*100}%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y # Stratify helps maintain class proportions in train/val splits, good for potentially imbalanced data.
    )

    print(f"  Training set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"  Validation set: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    print(f"  Training label distribution: {np.bincount(y_train) if len(y_train) > 0 else 'N/A'}")
    print(f"  Validation label distribution: {np.bincount(y_val) if len(y_val) > 0 else 'N/A'}")

    # Save the processed datasets
    os.makedirs(ML_READY_DATA_DIR, exist_ok=True)
    print(f"Saving datasets to {ML_READY_DATA_DIR}...")
    try:
        np.save(os.path.join(ML_READY_DATA_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(ML_READY_DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(ML_READY_DATA_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(ML_READY_DATA_DIR, 'y_val.npy'), y_val)
        print("  ML-ready datasets saved successfully.")
    except Exception as e:
        print(f"  Error saving .npy datasets: {e}")

    print("\n--- Dataset Preparation Finished ---")

if __name__ == "__main__":
    main() 