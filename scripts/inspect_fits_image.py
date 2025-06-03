"""
Inspects and visualizes sample FITS image files.

This script is designed for manually checking the content and appearance of FITS images.
It loads specified FITS files, prints basic HDU (Header Data Unit) information,
and displays a selected band (e.g., g, r, or z) of the image using Matplotlib.
Pixel values are scaled using percentiles for better visualization of astronomical images.

Key functionalities:
- Defines a list of sample FITS files to inspect.
- Opens FITS files using Astropy.
- Prints HDU info (e.g., name, dimensions, format).
- Extracts image data from the primary HDU.
- Allows selection of a specific band (g, r, or z) for display.
- Plots the selected band with robust percentile-based intensity scaling.
"""
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration: Sample Images for Visualization ---
# These are examples; you might want to update these paths or SPECOBJIDs based on your downloads.
IMAGES_TO_VISUALIZE = [
    {"path": os.path.join("..", "data", "fits_images", "agn", "3.8850806398166016e+17.fits"), "type": "AGN", "id": "3.8850806398166016e+17"},
    {"path": os.path.join("..", "data", "fits_images", "agn", "3.884426430398075e+17.fits"), "type": "AGN", "id": "3.884426430398075e+17"},
    {"path": os.path.join("..", "data", "fits_images", "non_agn", "3.085491082499461e+17.fits"), "type": "Non-AGN", "id": "3.085491082499461e+17"},
    {"path": os.path.join("..", "data", "fits_images", "non_agn", "3.085491297365172e+17.fits"), "type": "Non-AGN", "id": "3.085491297365172e+17"}
]
BAND_NAMES = ['g', 'r', 'z'] # Corresponding to indices 0, 1, 2
DEFAULT_BAND_INDEX = 1 # Display r-band by default (it's often a good balance for galaxy features)

def plot_fits_band(fits_filepath, image_id, image_type, band_index=DEFAULT_BAND_INDEX):
    """
    Opens a FITS file, displays HDU info, and plots a specified image band.

    Args:
        fits_filepath (str): Path to the FITS file.
        image_id (str): Identifier for the image (e.g., SPECOBJID).
        image_type (str): Type of image (e.g., "AGN", "Non-AGN").
        band_index (int): Index of the band to display (0 for g, 1 for r, 2 for z).
                          Defaults to r-band (index 1).
    """
    if not os.path.exists(fits_filepath):
        print(f"Error: File not found at {fits_filepath}. Skipping.")
        return

    print(f"\n--- Inspecting: {os.path.basename(fits_filepath)} (Type: {image_type}, ID: {image_id}) ---")
    try:
        with fits.open(fits_filepath) as hdul:
            hdul.info() # Shows a summary of all HDUs in the file

            # We expect image data in the primary HDU for these cutouts.
            if not hdul or hdul[0].data is None:
                print(f"Warning: No data in primary HDU for {fits_filepath}. Skipping plot.")
                return
            
            image_data = hdul[0].data

            # Check if image_data has the expected 3-band structure
            if image_data.ndim == 3 and image_data.shape[0] == len(BAND_NAMES):
                if not (0 <= band_index < len(BAND_NAMES)):
                    print(f"Invalid band_index {band_index}. Defaulting to '{BAND_NAMES[DEFAULT_BAND_INDEX]}'-band (index {DEFAULT_BAND_INDEX}).")
                    band_index = DEFAULT_BAND_INDEX
                
                single_band_data = image_data[band_index, :, :]
                current_band_name = BAND_NAMES[band_index]
                
                print(f"  Image data shape: {image_data.shape}")
                print(f"  Displaying band: '{current_band_name}' (index {band_index})")

                # Percentile scaling helps to show features in astronomical images 
                # which can have a large dynamic range. Clipping the top/bottom 1% 
                # often gives a much better visual representation than simple min/max.
                vmin = np.percentile(single_band_data, 1)
                vmax = np.percentile(single_band_data, 99)

                plt.figure(figsize=(7, 7))
                # Using 'viridis' colormap as it's perceptually uniform and good for data visualization.
                # 'origin=\'lower\'' is standard for astronomical images.
                plt.imshow(single_band_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
                plt.title(f"{image_type}: {image_id}\nBand: {current_band_name}")
                plt.xlabel("Pixel X")
                plt.ylabel("Pixel Y")
                plt.colorbar(label="Pixel Value (Scaled)")
                print(f"  Plotting {image_id} ({current_band_name}-band). Close plot window to continue...")
                plt.show()
            else:
                print(f"  Image data in {fits_filepath} is not in the expected format ({len(BAND_NAMES)}, H, W). Shape is {image_data.shape}. Cannot plot band.")
    
    except Exception as e:
        print(f"An error occurred while processing {fits_filepath}: {e}")

def main():
    """Main script execution to visualize selected FITS images."""
    print("--- Visualizing Sample FITS Images ---")
    print(f"Displaying images using the '{BAND_NAMES[DEFAULT_BAND_INDEX]}' band by default.")

    if not IMAGES_TO_VISUALIZE:
        print("No images configured in IMAGES_TO_VISUALIZE list. Nothing to display.")
        return

    for image_info in IMAGES_TO_VISUALIZE:
        plot_fits_band(image_info["path"], image_info["id"], image_info["type"], band_index=DEFAULT_BAND_INDEX)
    
    print("\n--- Visualization Script Finished ---")

if __name__ == "__main__":
    main() 