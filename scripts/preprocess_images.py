"""
Preprocesses downloaded FITS images into NumPy arrays suitable for a Keras CNN.

This script iterates through categorized FITS images (AGN and non-AGN), loads them,
performs necessary transformations, and saves them as .npy files.

The key preprocessing steps for each image are:
1.  Loading the FITS image data (expected to be 3-channel: g, r, z).
2.  Transposing the image axes from (channels, height, width) typically found in
    astronomical FITS to (height, width, channels) expected by Keras.
3.  Normalizing pixel values: Global min-max scaling to the range [0, 1].
    The entire 3-channel image is scaled based on its overall minimum and maximum pixel values.
    This method was used in the previous version and is maintained for consistency.

The processed NumPy arrays are saved in a parallel directory structure under
`../data/processed_images/`.
"""
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt # For visualization later
from tqdm import tqdm # For a progress bar

# --- Configuration ---
INPUT_BASE_DIR = '../data/fits_images' # Adjusted path
OUTPUT_BASE_DIR = '../data/processed_images' # Adjusted path
CATEGORIES = ['agn', 'non_agn']

EXPECTED_SHAPE_BEFORE_TRANSPOSE = (3, 160, 160) # Channels, Height, Width
EXPECTED_SHAPE_AFTER_TRANSPOSE = (160, 160, 3)  # Height, Width, Channels (Keras default)

# --- Function to find the first FITS file in a directory (from inspect_fits_image.py) ---
def get_first_fits_file(directory):
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(".fits"):
                return os.path.join(directory, filename)
    except FileNotFoundError:
        print(f"Error: Directory not found - {directory}")
    return None

# --- Helper Function: Preprocess a single image ---
def preprocess_image(fits_path):
    """
    Loads a FITS image, reorders axes, and normalizes pixel values.

    Args:
        fits_path (str): Path to the input FITS file.

    Returns:
        numpy.ndarray or None: Processed image data as (height, width, channels) 
                                or None if an error occurs.
    """
    try:
        with fits.open(fits_path) as hdul:
            if not hdul or not hdul[0].data is not None:
                print(f"Warning: No data in primary HDU for {fits_path}. Skipping.")
                return None
            
            # Data is expected to be (bands, height, width) -> (3, 160, 160)
            data = hdul[0].data.astype(np.float32) # Ensure float32

            if data.shape != (3, 160, 160):
                print(f"Warning: Unexpected data shape {data.shape} for {fits_path}. Expected (3, 160, 160). Skipping.")
                return None

            # 1. Axis Reordering: (bands, height, width) -> (height, width, bands)
            processed_data = np.transpose(data, (1, 2, 0)) # from (0,1,2) to (1,2,0)

            # 2. Normalization: Per-image min-max scaling to [0, 1]
            # This scales the entire 3-channel image based on its global min/max
            min_val = processed_data.min()
            max_val = processed_data.max()

            if max_val == min_val: # Avoid division by zero if image is flat
                # If all pixels are the same, normalize to 0 or 0.5. Let's choose 0.
                processed_data = np.zeros_like(processed_data, dtype=np.float32)
                # print(f"Info: Image {fits_path} is flat (all pixels have same value). Normalized to zeros.")
            else:
                processed_data = (processed_data - min_val) / (max_val - min_val)
            
            return processed_data

    except FileNotFoundError:
        print(f"Error: FITS file not found at {fits_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing FITS file {fits_path}: {e}. Skipping.")
        return None

# --- Main Preprocessing Logic ---
def main():
    """Main script execution to preprocess all images in the input directories."""
    print("--- Starting Image Preprocessing ---")

    processed_counts = {cat: 0 for cat in CATEGORIES}
    skipped_counts = {cat: 0 for cat in CATEGORIES}

    for category in CATEGORIES:
        input_dir = os.path.join(INPUT_BASE_DIR, category)
        output_dir = os.path.join(OUTPUT_BASE_DIR, category)

        if not os.path.isdir(input_dir):
            print(f"Input directory {input_dir} not found. Skipping category: {category}")
            continue

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing category: {category}")
        print(f"Input from: {input_dir}")
        print(f"Output to: {output_dir}")

        fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.fits')]
        
        if not fits_files:
            print(f"No .fits files found in {input_dir}.")
            continue

        print(f"Found {len(fits_files)} FITS files to process for {category}.")

        for filename in tqdm(fits_files, desc=f"Processing {category}"):
            fits_path = os.path.join(input_dir, filename)
            
            # Preprocess the image
            processed_image_data = preprocess_image(fits_path)

            if processed_image_data is not None:
                # Construct output path for .npy file
                base_filename = os.path.splitext(filename)[0] # Get filename without .fits
                npy_output_path = os.path.join(output_dir, f"{base_filename}.npy")
                
                try:
                    np.save(npy_output_path, processed_image_data)
                    processed_counts[category] += 1
                except Exception as e:
                    print(f"Error saving .npy file {npy_output_path}: {e}")
                    skipped_counts[category] += 1
            else:
                skipped_counts[category] += 1
    
    print("\n--- Preprocessing Summary ---")
    for category in CATEGORIES:
        print(f"Category: {category}")
        print(f"  Successfully processed and saved: {processed_counts[category]} images")
        print(f"  Skipped or errors: {skipped_counts[category]} images")

    print("\n--- Image Preprocessing Finished ---")

    # Optional: Visualize one of the saved .npy files to verify
    # if total_processed_count > 0:
    #     # Find a sample .npy file to load and display
    #     sample_npy_category = ''
    #     if os.path.exists(os.path.join(OUTPUT_BASE_DIR, 'agn')) and len(os.listdir(os.path.join(OUTPUT_BASE_DIR, 'agn'))) > 0:
    #         sample_npy_category = 'agn'
    #     elif os.path.exists(os.path.join(OUTPUT_BASE_DIR, 'non_agn')) and len(os.listdir(os.path.join(OUTPUT_BASE_DIR, 'non_agn'))) > 0:
    #         sample_npy_category = 'non_agn'
        
    #     if sample_npy_category:
    #         output_cat_dir = os.path.join(OUTPUT_BASE_DIR, sample_npy_category)
    #         sample_npy_file = None
    #         for f_name in os.listdir(output_cat_dir):
    #             if f_name.endswith('.npy'):
    #                 sample_npy_file = os.path.join(output_cat_dir, f_name)
    #                 break
            
    #         if sample_npy_file:
    #             print(f"\nVisualizing a sample processed .npy file: {sample_npy_file}")
    #             loaded_image = np.load(sample_npy_file)
    #             display_channel = 0 # g-band
    #             plt.figure(figsize=(6,6))
    #             plt.imshow(loaded_image[:, :, display_channel], cmap='gray', origin='lower')
    #             plt.title(f"Processed Band {display_channel} from {os.path.basename(sample_npy_file)}")
    #             plt.colorbar(label="Normalized Pixel Value")
    #             plt.show() 

if __name__ == "__main__":
    main() 