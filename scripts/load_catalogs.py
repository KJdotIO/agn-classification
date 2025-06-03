"""
Loads astronomical catalog data from FITS files, performs necessary cleaning,
filtering, and merging, and saves a final target list as a CSV file.

This script handles two main FITS files:
1.  galSpecInfo: Contains basic spectral information, coordinates, and quality flags.
2.  galSpecExtra: Contains derived properties, including BPT classification for AGN.

The process involves:
- Loading FITS tables using Astropy.
- Converting to Pandas DataFrames, handling potential multidimensional columns.
- Cleaning string identifier columns.
- Filtering based on reliability, galaxy type, redshift warnings, and BPT class.
- Merging the filtered tables to create a unified dataset.
- Generating a binary 'label' (0 for Non-AGN, 1 for AGN).
- Saving the final target list (RA, Dec, SPECOBJID, label) to a CSV.
"""
from astropy.table import Table
import pandas as pd
import os

# Define file paths (relative to the script's location in scripts/)
SPEC_INFO_PATH = '../data/galSpecInfo-dr8.fits'
SPEC_EXTRA_PATH = '../data/galSpecExtra-dr8.fits'
OUTPUT_CSV_PATH = '../data/final_galaxy_targets.csv'

def load_and_filter_specinfo(file_path):
    """Loads and filters the galSpecInfo FITS table."""
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        astro_table = Table.read(file_path)
        print(f"  Loaded. Original rows: {len(astro_table)}")

        # Handle potential multidimensional columns before converting to Pandas
        oned_cols = [name for name in astro_table.colnames if len(astro_table[name].shape) <= 1]
        df = astro_table[oned_cols].to_pandas()
        
        # Clean key string columns
        if 'SPECOBJID' in df.columns and isinstance(df['SPECOBJID'].iloc[0], bytes):
            df['SPECOBJID'] = df['SPECOBJID'].str.decode('utf-8').str.strip()
        if 'SPECTROTYPE' in df.columns and isinstance(df['SPECTROTYPE'].iloc[0], bytes):
            df['SPECTROTYPE'] = df['SPECTROTYPE'].str.decode('utf-8').str.strip()

        # Apply filters
        reliable_mask = df['RELIABLE'] == 1
        spectrotype_mask = df['SPECTROTYPE'] == 'GALAXY'
        zwarning_mask = df['Z_WARNING'] == 0
        
        filtered_df = df[reliable_mask & spectrotype_mask & zwarning_mask]
        print(f"  Filtered. Rows remaining: {len(filtered_df)}")
        return filtered_df
    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

def load_and_filter_specextra(file_path):
    """Loads and filters the galSpecExtra FITS table."""
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        astro_table = Table.read(file_path)
        print(f"  Loaded. Original rows: {len(astro_table)}")

        # Check for multidimensional columns (though not expected here based on prior checks)
        multidim_cols = [name for name in astro_table.colnames if len(astro_table[name].shape) > 1]
        if multidim_cols:
            print(f"  Warning: {os.path.basename(file_path)} contains multidimensional columns: {multidim_cols}. Selecting 1D columns.")
            oned_cols = [name for name in astro_table.colnames if len(astro_table[name].shape) <= 1]
            df = astro_table[oned_cols].to_pandas()
        else:
            df = astro_table.to_pandas()

        if 'SPECOBJID' in df.columns and isinstance(df['SPECOBJID'].iloc[0], bytes):
            df['SPECOBJID'] = df['SPECOBJID'].str.decode('utf-8').str.strip()

        # BPTCLASS: 1 for Star Forming (label 0), 4 for AGN (label 1)
        bpt_mask = df['BPTCLASS'].isin([1, 4])
        filtered_df = df[bpt_mask]
        print(f"  Filtered. Rows remaining: {len(filtered_df)}")
        return filtered_df
    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

def main():
    """Main script execution."""
    print("--- Starting Catalog Loading and Preparation ---")

    info_df_filtered = load_and_filter_specinfo(SPEC_INFO_PATH)
    extra_df_filtered = load_and_filter_specextra(SPEC_EXTRA_PATH)

    if info_df_filtered.empty or extra_df_filtered.empty:
        print("One or both dataframes are empty after filtering. Cannot proceed with merge.")
        print("Script finished.")
        return

    print("Merging filtered dataframes...")
    try:
        # Select only necessary columns for the merge
        info_cols = ['SPECOBJID', 'RA', 'DEC']
        extra_cols = ['SPECOBJID', 'BPTCLASS']
        
        merged_df = pd.merge(
            info_df_filtered[info_cols],
            extra_df_filtered[extra_cols],
            on='SPECOBJID',
            how='inner'
        )
        print(f"  Successfully merged. Rows in merged_df: {len(merged_df)}")

        # Create the binary 'label': BPTCLASS 1 (Star Forming) -> 0, BPTCLASS 4 (AGN) -> 1
        merged_df['label'] = merged_df['BPTCLASS'].replace({1: 0, 4: 1})
        
        final_target_df = merged_df[['RA', 'DEC', 'SPECOBJID', 'label']]

        # Ensure output directory exists
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        final_target_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Final target list saved to {OUTPUT_CSV_PATH}")

        print("--- Final Target List Summary ---")
        print(f"Total galaxies: {len(final_target_df)}")
        print("Label distribution:")
        print(final_target_df['label'].value_counts(dropna=False))
        # print("First 5 entries:", final_target_df.head()) # Kept as optional for quick check

    except Exception as e:
        print(f"Error during merging or saving: {e}")

    print("--- Catalog preparation script finished. ---")

if __name__ == "__main__":
    main()