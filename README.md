# Classifying AGN Host Galaxies with Convolutional Neural Networks

This project documents the development of a Convolutional Neural Network (CNN) designed to classify galaxies based on whether they host an Active Galactic Nucleus (AGN). The goal is to apply deep learning techniques to astronomical image data. Similar to how MNIST serves as a foundational project for digit recognition, this work aims to explore galaxy classification using comparable machine learning principles, but adapted for the complexities of astrophysical analysis. We will detail the process of sourcing and preparing galaxy images, building a CNN, and training it to distinguish between AGN host galaxies and non-active galaxies.

## Motivation and Scientific Context

You might wonder, why focus on AGN? Well, the traditional ways astronomers find these energetic galactic cores involve something called spectroscopy. It's super detailed and accurate, but also pretty slow and expensive, especially when you think about how much data new telescopes are about to dump on us. Observatories like the Vera C. Rubin and the Nancy Grace Roman are going to be capturing images of *millions* of galaxies. We simply won't be able to do spectroscopy on all of them! This is where faster, automated methods for spotting potential AGN candidates for closer study become really important.

The work of [Guo et al. (2022)](https://arxiv.org/abs/2212.07881), "Identifying AGN host galaxies with convolutional neural networks," is a key reference. Their paper showed that it's actually possible to distinguish AGN hosts from normal, non-active galaxies with impressive accuracy (around 89%) just by looking at their shape and structure in images. That really sparked my curiosity – could I learn to do something similar?

## Project Goal

The primary goal of this project is to train a CNN that can classify galaxy images into two main groups:

-   **AGN Host Galaxies**: These are galaxies that have a supermassive black hole at their center actively gobbling up material and shining brightly.
-   **Non-Active Galaxies**: These are your more "normal" galaxies, going about their business without a bright AGN.

To do this, I'll be using optical imaging data – specifically, images taken in the g, r, and i light bands (think different colors) from the Sloan Digital Sky Survey (SDSS), which have been conveniently processed and made available through the Legacy Survey. For the "ground truth" – knowing which galaxies are which – I'll rely on labels derived from those detailed spectroscopic BPT diagrams I mentioned earlier.

## Project Setup

We'll be working with a carefully curated virtual environment to ensure all our cosmic tools work together harmoniously. Our project uses Python 3.12 and leverages the power of modern deep learning frameworks.

### File Structure

Our project directory is organized as follows:

```
agn-classification/
├── .git/                    # Git version control files
├── .vscode/                 # VS Code workspace settings (optional)
├── .cursor/                 # Cursor AI related files (optional)
├── agn-env/                 # Python virtual environment (ignored by Git)
├── data/
│   ├── fits_images/         # Raw downloaded FITS images (ignored by Git)
│   │   ├── agn/
│   │   └── non_agn/
│   ├── processed_images/    # Preprocessed images as .npy files (ignored by Git)
│   │   ├── agn/
│   │   └── non_agn/
│   ├── ml_ready_data/       # Train/validation .npy datasets (ignored by Git)
│   ├── galSpecInfo-dr8.fits # Original catalog file
│   └── galSpecExtra-dr8.fits# Original catalog file
├── docs/
│   ├── projectroadmap.md    # Detailed project roadmap and progress tracking
│   └── Identifying AGN host galaxies with convolutional neural networks.txt # Guiding paper notes
├── output/
│   ├── training_history_resnet18.png # Example training plot (ignored by Git)
│   └── agn_resnet18_model.keras      # Example saved model (ignored by Git)
├── scripts/
│   ├── load_catalogs.py
│   ├── download_images.py
│   ├── inspect_fits_image.py
│   ├── preprocess_images.py
│   ├── prepare_dataset.py
│   └── train_cnn.py
├── .gitignore               # Specifies intentionally untracked files for Git
├── README.md                # This file - our project story
└── requirements.txt         # Complete list of Python dependencies
```

## Required Libraries and Environment Setup

Setting up the right environment is crucial for this project. We're working with astronomical data, deep learning models, and complex visualizations, so we need a robust toolkit. Here's what we're using and why:

### Core Libraries

```python
import numpy as np           # For numerical operations on galaxy image arrays
import pandas as pd          # For handling astronomical catalogs and metadata  
import matplotlib.pyplot as plt  # For visualizing galaxies and training results
import seaborn as sns        # For enhanced statistical visualizations
import tensorflow as tf      # Our deep learning powerhouse (includes Keras)
import astropy              # For astronomical file formats (FITS) and coordinates
from sklearn import metrics, model_selection  # For ML utilities and evaluation
```

**NumPy (2.1.3)** - The foundation of our numerical work. Galaxy images are essentially arrays of pixel values (160×160×3 for our gri-band images), and NumPy gives us the speed and functionality to manipulate these efficiently.

**Pandas (2.2.3)** - Our data detective for handling astronomical catalogs. We'll be working with catalogs containing hundreds of thousands of galaxies, their coordinates, redshifts, colors, and AGN classifications. Pandas makes filtering and analyzing this data straightforward.

**Matplotlib & Seaborn** - Essential for visualizing our cosmic data. We'll plot galaxy images, training curves, confusion matrices, and astrophysical relationships between galaxy properties and AGN activity.

**TensorFlow (2.19.0) with Keras** - Our neural network engine. This will handle everything from building CNN architectures to training on GPU clusters and evaluating model performance.

**Astropy (7.1.0)** - The cosmic translator. Astronomical images come in FITS format with special headers containing coordinate systems, exposure times, and filter information. Astropy speaks this language fluently.

**Scikit-learn (1.6.1)** - Provides essential machine learning utilities like train/test splits, metrics calculation, and preprocessing tools that complement our deep learning workflow.

### Virtual Environment Setup

We use a virtual environment to create an isolated, reproducible workspace for our project. This prevents conflicts between different projects and ensures anyone can recreate our exact setup.

Here's how we set up :

```bash
# Create our dedicated AGN research environment
python3 -m venv agn-env

# Activate the environment (enter our isolated workspace)
source agn-env/bin/activate

# Install our complete scientific toolkit
pip install numpy pandas matplotlib seaborn astropy scikit-learn tensorflow jupyter

# Verify everything works correctly
python3 -c "import numpy, pandas, matplotlib, astropy, sklearn, tensorflow as tf; print('Works well')"

# Document our environment for reproducibility
pip freeze > requirements.txt
```

The beauty of this setup is that it's completely isolated from your system Python. When we activate `agn-env`, we're entering a specialized workspace where all our tools are precisely configured for astronomical deep learning.

## What's Next

With our environment ready, we're prepared to dive into the science! Our next steps will involve:

1. **Understanding the Astrophysics**: Learning about AGN, supermassive black holes, and why galaxy morphology matters
2. **Data Acquisition**: Downloading galaxy images and labels from SDSS/Legacy Survey
3. **Image Preprocessing**: Preparing astronomical data for neural network input
4. **Model Architecture**: Building our CNN from simple to sophisticated (ResNet-18 style)
5. **Training and Evaluation**: Teaching our model to recognize the subtle signs of AGN activity

This README will grow with our project, documenting each discovery and challenge along the way. The cosmos awaits our exploration! 🌌

---

*Last updated: Environment setup complete (Checkpoint 0.1) ✅*

## The Basics of AGN

Our universe is filled with billions of galaxies, each packed with an abundance of stars, hot gas, and dust. At the center of nearly every large galaxy, at least least astrophysicists believe, is a supermassive black hole (SMBH from now on). A SMBH, unlike its regular siblings, is a, well, supermassive object with millions, even billions of times the mass of our sun, compressed into an incredibly tiny space. 

Typically, these SMBHs remain dormant, just like the ones in our own Milky Way - Sagittarius A* (Sgr A*). However, sometimes, they wake up. When gas and dust from the galaxy fall towards the SMBH, it doesn't just go straight into it. Instead, it forms a swirling, superheated disk of material, heating up to millions of degrrees, creating a bright, energetic region called an accretion disk. This disk is so hot that it emits intense radiation across the electromagnetic spectrum, from visible light to X-rays. So bright, in fact, that it can outshine the entire galaxy around it. When this happens, we say that this galaxy hosts an Active Galactic Nucleus (AGN).

But why on earth do we care about these AGN? The reason is that the immene energy and radiation that is released can affect its host galaxy in a number of ways. For example, it can heat up the gas in the galaxy, causing it to emit light, or it can eject material from the galaxy, causing it to lose mass. It can also trigger star formation in the galaxy, or even cause the galaxy to merge with another galaxy. Understanding the properties of AGN host galaxies helps us understand how these galaxies evolve over time - how they form, how they grow, and how they die.

Okay, so, if AGN are so important, how do we find them? Traditionally, astronomers used spectroscopy. This involves taking the light from a galaxy, splitting it into its constituent colors, and looking for specific signatures of AGN activity. Certain patterns like the strength of certain emission lines can be used to tell us if the gas in the galaxys centre is being ionized (energyised) by the radiation from an AGN, or by the formation of new stars. In 1981, a tool known as the BPT diagram (developed and named after Baldwin, Phillips, and Terlevich) was developed to help distinguish between these two types of sources. It is a plot of the ratio of two emission lines against the ratio of two other emission lines, and this is used to distinguish between AGN and star-forming galaxies. The idea is that AGN produce much more intense ionising radiation than young stars, so the ratio of the two emission lines will be different for the two types of sources. This tool is incredibly accurate, and is still used today to identify AGN host galaxies, but it is _incredibly_ time consuming, which creates a bottleneck when we want to study large numbers of galaxies. In the paper by Guo et al. (2022), they mention that upcoming surveys done by the Vera C. Rubin Observatory and Nancy Grace Roman Space Telescope will capture more galaxies than all previous surveys combined, which also means that spectroscopic follow up will be practically impossible for every interesting object. We simply cant get the spectra for all of them, so we need a faster way to identify AGN host galaxies.

This brings us to the main hypothesis behind Guo et al. (2022), and this project: Could the overall appearance (morphology) of a galaxy be enough to tell us if it is hosting an AGN? There are a number of reasons to think why this might be true. Observational evidence put forward by Kauffmann et al. (2003) suggests that AGN are typically found in massive galaxies, which tend to have large central bulges and older stars, which look red. These characteristics (and maybe others) could be picked up by a smart enough algorithm to help us identify AGN host galaxies.

## Enter the Convolutional Neural Network

OK, so, we've got a hypothesis, but how can we actually, well, _do_ this? Enter the Convolutional Neural Network (CNN). CNNs are a type of neural network that are particularly good at image classification. They are made up of layers of neurons that are connected to each other, and each neuron is connected to a small number of neurons in the previous layer. The neurons in the first layer are connected to the input image, and the neurons in the last layer are connected to the output. The neurons in the middle layers are connected to the neurons in the previous and next layers, and they are used to extract features from the input image. CNNs are particularly good at image classification because they are able to learn the features of the input image by themselves, without the need for human intervention. Theyve been used for things like facial recognition, self-driving cars, and even to help doctors diagnose diseases from medical images. This is also great for relevant features might be subtle. We dont need to tell it to look for a big bulge or a red colour, it can learn this by itself through layers of filters called convolutions that detect edges, shapes, and textures. Theres a lot more that can be talked about, but this is the basic idea.

## Training (simplified)

To train our CNN to do these things, we'll need two things:

- Loads of galaxy images! We can get these from projects like the Sloan Digital Sky Survey (SDSS), who have millions of galaxy images.

- Labels! We need to know which galaxies are AGN host galaxies, and which are not. This'll be a bit slower, since we have to use traditional spectroscopy to get the labels, so we'll use the BPT diagram to help us.

### Heres our process:

1. Prepare the data: We'll resize them, normalize them, etc.

2. We need to augment the data: We'll flip them, rotate them, and maybe even add some noise to them. This will help the CNN learn to be more robust to different orientations and lighting conditions.

3. We'll use ResNet-18 as our base model, and then we'll train it on our data.

4. Training: The meat of the project. We feed the images and their labels to the CNN. The CNN will make a prediction, we compare its prediction to the true label. If its wrong, an optimisation algorithm will adjust the weights of the neurons in our CNN to make it more accurate. It'll repeat this process over many 'epochs' (think of it like a number of training sessions) until it gets it less wrong. I say less wrong because we'll never get it right 100% of the time, we need to reduce whats known as our cost function, which is a measure of how wrong the CNN is. its related to stuff like gradient descent, which is a way of finding the minimum of a function. You can read more about this in my neural network project for recognising handwritten digits [here](https://github.com/KJdotIO/handwriting-neural-network).

5. Validation: We'll keep a small portion of our data aside and use that to test our model. This is called the validation set. The CNN will never see this data, so we can use it to test how well it is doing. We want to avoid whats known as overfitting, which is when the CNN learns the training data too well, and it doesnt generalise to new data.

On the validation set, we can calculate the accuracy of the CNN (how many of the images were correctly classified). We can also calculate the precision and recall of the CNN (how many of the images that were classified as AGN were actually AGN, and how many of the images that were classified as non-AGN were actually non-AGN). And the F1 score, which is a weighted average of the precision and recall.

In the Guo et al. (2022) paper, they used a ResNet-18 model to classify galaxies into AGN and non-AGN categories. They found that the model was able to achieve an accuracy of 89% on the validation set. This is a great result, and it shows that the CNN is able to learn the features of the input image by itself, without the need for human intervention. More importantly, it shows that, yes, given the morphology of a galaxy, we can identify (to a reasonable degree of accuracy) if it is hosting an AGN.

## Whys this exciting?

This is exciting because we can sift through large numbers of galaxies and create a catalog of AGN _candidates_ far faster than spectroscopy can. It doesnt replace spectroscopy though, it acts more like a filter telling astronomers "hey, this galaxy looks like it might be hosting an AGN, point your telescope at these first!" We could also use this to find new types of AGN appearing in potentially unexpected environments.

## Data Sourcing: Catalogs and Images

### Galaxy Catalogs: MPA-JHU DR8

Prior to image acquisition, a target list is necessary. We need to know *which* galaxies to look at and what their status is – AGN or not. For this, we're turning to a specific dataset provided by the Sloan Digital Sky Survey (SDSS) community, known as the **MPA-JHU DR8 galaxy properties catalog**. This catalog is hosted on the [SDSS website](https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/) (though the page refers to DR17, it provides access to the DR8 version which aligns with our image data). This catalog is a valuable resource as it's based on the detailed spectroscopic analysis methods of Kauffmann et al. (2003) and others, which are foundational for the BPT diagram classifications used in our guiding paper by Guo et al. (2022).

From this source, the project utilizes two key FITS table files, `galSpecInfo-dr8.fits` and `galSpecExtra-dr8.fits`, located in the `data/` directory:

1.  **`galSpecInfo-dr8.fits`:** This table provides basic information for each galaxy spectrum analysed. For our purposes, the most important columns in this file are:
    *   `specObjID`: A unique identifier for each spectrum, which helps us link data across different tables.
    *   `ra` and `dec`: The Right Ascension and Declination – the celestial coordinates we need to tell the image cutout service where to point.
    *   `reliable`: A flag (usually 0 or 1). We'll only want to use data where this indicates the measurements are considered reliable.
    *   `spectrotype`: We'll filter for objects classified as 'GALAXY'.
    *   `z_warning`: A flag for redshift quality; we'll want cases where this is 0 (no warning).

2.  **`galSpecExtra-dr8.fits`:** This file contains derived physical parameters, including the BPT classification crucial for this project. The key column here is:
    *   `bptclass`: This integer tells us how the galaxy is classified based on its emission lines. The important values for us are:
        *   `1`: Star-forming (these will be our "non-AGN" examples)
        *   `4`: AGN (these will be our "AGN host" examples)
        *   Other values (like -1 for unclassifiable, 2 for low S/N star-forming, 3 for composite) will generally be excluded from our initial training set to keep our classes distinct, similar to the approach in Guo et al.

These two files are designed to be line-by-line matched using `specObjID`, so we can combine the coordinate information from `galSpecInfo` with the BPT classification from `galSpecExtra`. Our first coding task will be to load these files (likely with `Astropy` and then into `Pandas` DataFrames), filter them based on the `reliable`, `spectrotype`, `z_warning`, and `bptclass` flags, and then merge them to create our final target list of AGN and non-AGN galaxies with their precise sky coordinates.

### Image Acquisition: Legacy Survey DR8 Cutout Service

With the target list compiled from the MPA-JHU catalogs, the next step is image acquisition. For this, as mentioned earlier, we're using the **DESI Legacy Imaging Surveys Data Release 8 (DR8)**. The Legacy Surveys combine data from several telescopes, processed consistently through the NOIRLab Community Pipeline and "The Tractor" source detection algorithm.

We'll be using their "cutout service," which lets us request small, targeted FITS images centered on our specific galaxies. After some investigation on the [Legacy Survey DR8 Description page](https://www.legacysurvey.org/dr8/description/) and experimentation with the [Legacy Survey Viewer](https://www.legacysurvey.org/viewer/), the exact URL structure for this is:

`https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&layer=dr8&pixscale=0.262&size=160&bands=grz`

The URL parameters are as follows:
-   `ra={RA}` and `dec={DEC}`: These are the sky coordinates (in decimal degrees) that we'll pull from our filtered MPA-JHU catalog.
-   `layer=dr8`: Specifies Data Release 8.
-   `pixscale=0.262`: Sets the pixel scale to 0.262 arcseconds per pixel, matching the native DR8 resolution and the Guo et al. paper.
-   `size=160`: Requests a 160x160 pixel image cutout, again matching Guo et al.
-   `bands=grz`: This requests a FITS file with three image planes, for the **g** (green), **r** (red), and **z** (near-infrared) filters.

Note that the Guo et al. paper used `gri` bands. The `i`-band is a near-infrared filter slightly bluer than the `z`-band. Since the Legacy Survey cutout service most readily provides `grz` for these combined DR8 cutouts, we'll proceed with these bands. Our CNN will learn features specific to this `grz` combination. This choice is based on data availability and represents a slight deviation from the reference paper.

With the target list and URL structure defined, the next step is to use a Python script (like `scripts/download_images.py`) to systematically download these 160x160 pixel, `grz`-band FITS images for all selected AGN and non-AGN galaxies.

## Loading and Preparing the Data

With the sources for galaxy classifications (`galSpecInfo-dr8.fits` and `galSpecExtra-dr8.fits`) and image cutouts identified, the next step is to work with Python to bring this catalog data into a usable format. The primary tool for initially reading these FITS-formatted catalog files is the `astropy` library, specifically its `Table` class from the `astropy.table` module. This library is well-suited for such tasks as it's designed for astronomical data manipulation.

The script, `scripts/load_catalogs.py`, begins by loading both `galSpecInfo-dr8.fits` and `galSpecExtra-dr8.fits` into separate `astropy` Table objects. This allows for an immediate inspection of their structure, revealing column names and confirming the scale of the data (over 1.8 million entries in each). This initial check is critical for data validation.

Converting these Astropy Tables into Pandas `DataFrame` objects can present an interesting challenge. Pandas DataFrames are a common choice for general data wrangling, such as filtering and merging, due to their rich API. The direct conversion of the `galSpecInfo` table, however, would fail with a `ValueError` if attempted without care. The reason, as indicated by the error message, is the presence of 'multidimensional columns.' Some columns in astronomical FITS tables (e.g., `PLUG_MAG`, `SPECTRO_MAG`) contain arrays of values within each row, whereas standard Pandas DataFrames expect each cell to hold a single, scalar value.

To resolve this, let's follow `astropy`'s guidance. For the `galSpecInfo` table, we first programmatically identify all columns that are genuinely one-dimensional (i.e., not arrays). This is achieved with a list comprehension: `[name for name in info_table_astro.colnames if len(info_table_astro[name].shape) <= 1]`. Using this list of 1D column names, a new, "flattened" Astropy Table containing only these compatible columns is then created. This filtered Astropy Table can then be successfully converted into a Pandas DataFrame, which we can name `info_df`. A similar check on the `galSpecExtra` table reveals it contains no such multidimensional columns, allowing for a direct conversion to its corresponding Pandas DataFrame, `extra_df`. This process of adapting data structures is a common step in preparing diverse datasets for analysis.

This process results in `info_df` and `extra_df` – two Pandas DataFrames containing essential catalog information, cleaned of structural incompatibilities. These DataFrames are now primed for the next crucial phase: applying scientific and quality-based filters, and then merging them to create a final, unified target list. This list provides the precise sky coordinates for downloading galaxy images and the definitive labels (AGN or non-AGN) for training the neural network.

### Refining the Sample: Filtering and Merging for Clarity

After loading the primary catalogs (`galSpecInfo-dr8.fits` and `galSpecExtra-dr8.fits`) into Pandas DataFrames (`info_df` and `extra_df`), the subsequent step involves filtering these datasets to create a focused list of high-quality galaxies for the AGN classification task. This is achieved by filtering each DataFrame based on scientific and quality criteria, followed by merging them to consolidate the necessary information.

The `info_df` (derived from `galSpecInfo-dr8.fits`) is filtered to select galaxies with reliable spectral information. This involves applying a series of boolean masks:

1.  `RELIABLE == 1`: Ensuring the catalogued data for the galaxy is considered trustworthy.
2.  `SPECTROTYPE == 'GALAXY'`: Explicitly selecting objects classified as galaxies, excluding stars or quasars that might be in the broader spectroscopic sample.
3.  `Z_WARNING == 0`: Making sure there are no warnings associated with the redshift measurement, which indicates a more robust distance estimate and spectral analysis.

Text-based columns such as `SPECOBJID` (the merging key) and `SPECTROTYPE` are often read as byte strings (e.g., `b'GALAXY   '`) from FITS files. These are converted to standard Python strings, with leading/trailing whitespace stripped prior to filtering or merging to ensure accurate comparisons and joins. For example:

```python
# Example of cleaning and filtering info_df in scripts/load_catalogs.py
if 'SPECOBJID' in info_df.columns and isinstance(info_df['SPECOBJID'].iloc[0], bytes):
    info_df['SPECOBJID'] = info_df['SPECOBJID'].str.decode('utf-8').str.strip()
if 'SPECTROTYPE' in info_df.columns and isinstance(info_df['SPECTROTYPE'].iloc[0], bytes):
    info_df['SPECTROTYPE'] = info_df['SPECTROTYPE'].str.decode('utf-8').str.strip()

reliable_mask = info_df['RELIABLE'] == 1
spectrotype_mask = info_df['SPECTROTYPE'] == 'GALAXY'
zwarning_mask = info_df['Z_WARNING'] == 0
info_df_filtered = info_df[reliable_mask & spectrotype_mask & zwarning_mask]
```

Applying these filters to `info_df` (which initially has 1,843,200 entries) reduces it to 918,438 galaxies that meet our quality and type criteria.

Next up is `extra_df` (from `galSpecExtra-dr8.fits`). Here, the main thing is to pick out galaxies based on their BPT classification, which is how we'll get our AGN and non-AGN labels. We zero in on two specific BPT classes:

1.  `BPTCLASS == 1`: Galaxies classified as "Star-Forming." These become our non-AGN sample, getting a label of `0`.
2.  `BPTCLASS == 4`: Galaxies classified as "AGN" (specifically excluding LINERs, which can be a more ambiguous category). And these are our AGN sample, labeled `1`.

So, `extra_df` is filtered to keep only the rows with these `BPTCLASS` values:

```python
# Example of cleaning SPECOBJID and filtering extra_df in scripts/load_catalogs.py
if 'SPECOBJID' in extra_df.columns and isinstance(extra_df['SPECOBJID'].iloc[0], bytes):
    extra_df['SPECOBJID'] = extra_df['SPECOBJID'].str.decode('utf-8').str.strip()

bpt_mask = extra_df['BPTCLASS'].isin([1, 4])
extra_df_filtered = extra_df[bpt_mask]
```

This filtering trims `extra_df` down from 1,843,200 entries to 242,262 galaxies classified as either Star-Forming or AGN.

With both DataFrames filtered, the last step to get our target list ready is to merge them. We perform an 'inner' merge using the cleaned `SPECOBJID` column. An inner merge is good here because it means only galaxies that are in *both* our filtered DataFrames make it to the final list. This is super important because we need both the coordinates (from `info_df_filtered`) and the BPT class (from `extra_df_filtered`) for every target.

```python
# Example of merging in scripts/load_catalogs.py
merged_df = pd.merge(
    info_df_filtered[['SPECOBJID', 'RA', 'DEC']], # Selecting only needed columns
    extra_df_filtered[['SPECOBJID', 'BPTCLASS']],
    on='SPECOBJID',
    how='inner'
)

# Creating the final 'label' column
merged_df['label'] = merged_df['BPTCLASS'].replace({1: 0, 4: 1})

final_target_df = merged_df[['RA', 'DEC', 'SPECOBJID', 'label']]
```

The merge gives us `final_target_df` with 229,828 unique galaxies. For each one, we now have its Right Ascension (`RA`), Declination (`DEC`), unique `SPECOBJID`, and our important binary `label` (0 for Star-Forming, 1 for AGN). This dataset is then all set to be the input for downloading the image cutouts.

After the merge, we create that key `label` column. Based on the `BPTCLASS` (where 1 is Star-Forming and 4 is AGN), a `0` is assigned for non-AGN and a `1` for AGN. The `scripts/load_catalogs.py` script then grabs the essential columns – `RA`, `DEC`, `SPECOBJID`, and our new `label` – to make `final_target_df`. This DataFrame is then saved to `data/final_galaxy_targets.csv`, ready for the image downloading script.

```python
# In scripts/load_catalogs.py: Creating the label and saving the final list
# merged_df['label'] = merged_df['BPTCLASS'].replace({1: 0, 4: 1})
# final_target_df = merged_df[['RA', 'DEC', 'SPECOBJID', 'label']]
# output_csv_path = os.path.join('data', 'final_galaxy_targets.csv')
# final_target_df.to_csv(output_csv_path, index=False)
```

Running `scripts/load_catalogs.py` gives us our final target list, with 229,828 unique galaxies. Here's how they break down:
- Non-AGN (label 0): 208,057
- AGN (label 1): 21,771

This list is then ready to guide our image downloading.

### Downloading Our Sample Images

With our `final_galaxy_targets.csv` in hand, the next step is to actually download the images. For this, we can develop a script like `scripts/download_images.py`. This script can be designed to:

1.  Load the `final_galaxy_targets.csv` into a Pandas DataFrame.
2.  Allow us to specify how many images per class we want for an initial sample (e.g., `NUM_TO_DOWNLOAD_PER_CLASS = 5`).
3.  Iterate through a sample of AGN and non-AGN targets from our list.
4.  For each target, dynamically construct the download URL using its `RA` and `DEC` coordinates with the Legacy Survey FITS cutout service pattern we identified earlier:
    `https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&layer=dr8&pixscale=0.262&size=160&bands=grz`
5.  Use the `requests` library to fetch the FITS image.
6.  Save the downloaded FITS image into appropriately named subdirectories: `data/fits_images/agn/` or `data/fits_images/non_agn/`. The files are named using their unique `SPECOBJID` (e.g., `299501221847263232.fits`).

The script can also include error handling for download issues and a polite delay between requests to avoid overwhelming the server. For an initial run, this would successfully download a small set of 5 AGN and 5 non-AGN FITS images, giving us our first batch of raw image data.

```python
# Snippet from scripts/download_images.py illustrating the download logic:
# for index, row in download_sample_df.iterrows():
#     ra = row['RA']
#     dec = row['DEC']
#     specobjid = str(row['SPECOBJID'])
#     label = row['label']
#     output_dir = AGN_DIR if label == 1 else NON_AGN_DIR
#     output_filepath = os.path.join(output_dir, f"{specobjid}.fits")
#
#     if not os.path.exists(output_filepath): # Check if file already exists
#         download_url = DOWNLOAD_URL_PATTERN.format(RA=ra, DEC=dec)
#         response = requests.get(download_url, timeout=30)
#         response.raise_for_status() # Ensure we got a good response
#         with open(output_filepath, 'wb') as f:
#             f.write(response.content)
#         # print(f"Successfully downloaded and saved to {output_filepath}")
#         time.sleep(DOWNLOAD_DELAY) # Be polite
```

### Preparing Images for the Neural Network: Preprocessing

Raw astronomical FITS images, while scientifically rich, aren't directly suitable for input into a Convolutional Neural Network. They need to be transformed into a more standardized format. This crucial step can be handled by our `scripts/preprocess_images.py` script. The goal here is to convert our 3-channel (g, r, z bands) 160x160 pixel FITS images into NumPy arrays that our Keras model will expect.

The core of this script is the `load_and_preprocess_image` function, which performs several key operations on each FITS file:

1.  **Loading FITS Data**: We use `astropy.io.fits` to open the FITS file and access the image data. This data typically comes as a 3D NumPy array. We ensure the data type is `np.float32` for consistent numerical operations.

    ```python
    # In scripts/preprocess_images.py, loading data with astropy
    # with fits.open(fits_filepath) as hdul:
    #     image_data = hdul[0].data.astype(np.float32)
    ```

2.  **Correcting Image Dimensions (Axis Transposition)**: Astronomical FITS files often store multi-band image data in a `(channels, height, width)` format. For our `grz` images, this means the raw shape is `(3, 160, 160)`. However, Keras (and TensorFlow) typically expects image data in the `(height, width, channels)` format. To reconcile this, we use NumPy's `transpose` method:

    ```python
    # In scripts/preprocess_images.py: Transposing axes
    # Initial shape: (3, 160, 160) for (channels, height, width)
    # image_data_transposed = np.transpose(image_data, (1, 2, 0))
    # New shape: (160, 160, 3) for (height, width, channels)
    ```
    This operation rearranges the data axes to the `(160, 160, 3)` shape our CNN requires.

3.  **Normalizing Pixel Values**: Neural networks generally train more effectively when their input features (pixel values, in our case) are scaled to a small, consistent range. We can implement a *per-channel min-max normalization*. This means that for each of the g, r, and z bands independently, we scale its pixel values to lie strictly between 0 and 1. The formula applied to each pixel in a band is:
    `(pixel_value - min_band_pixel) / (max_band_pixel - min_band_pixel)`
    If a band has no variation (all pixel values are the same), we set all its normalized values to 0. This ensures that all input pixel data, regardless of the original brightness or offset in each band, is presented to the network in a uniform [0, 1] scale.

    ```python
    # In scripts/preprocess_images.py: Conceptual per-channel normalization
    # normalized_image = np.zeros_like(image_data_transposed, dtype=np.float32)
    # for i in range(image_data_transposed.shape[2]): # Loop through channels (g, r, z)
    #     band = image_data_transposed[:, :, i]
    #     min_val = np.min(band)
    #     max_val = np.max(band)
    #     if max_val - min_val > 1e-6: # Check for non-zero range (added epsilon for stability)
    #         normalized_image[:, :, i] = (band - min_val) / (max_val - min_val)
    #     else:
    #         normalized_image[:, :, i] = 0 # Default for zero-range bands
    ```

4.  **Saving Processed Data**: After an image is successfully preprocessed by `load_and_preprocess_image`, the result is a NumPy array of shape `(160, 160, 3)` containing these normalized `float32` pixel values. To efficiently store and later load these processed arrays for training, we save each one as a separate `.npy` file using `np.save()`. These files are organized into new directories: `data/processed_images/agn/` and `data/processed_images/non_agn/`, mirroring our raw FITS data structure but now containing ready-to-use arrays.

A batch execution of `scripts/preprocess_images.py` would process an initial sample of, say, 10 FITS images (5 AGN, 5 non-AGN) and save them as `.npy` files. We would then have a small, clean, and correctly formatted dataset, perfectly prepared for the next stages of structuring it for Keras and building our first CNN model.

### Structuring Data for Training and Validation

With our individual galaxy images preprocessed and saved as `.npy` files (each a NumPy array of shape `(160, 160, 3)` with normalized pixel values), the next crucial step, handled by our `scripts/prepare_dataset.py` script, is to organize this data into a format suitable for training a Keras model. This involves creating unified datasets for images and their corresponding labels, and then splitting them into training and validation sets.

1.  **Loading Processed Images and Assigning Labels**: The script begins by systematically loading all the `.npy` image arrays from our `data/processed_images/agn/` directory and assigning them a label of `1` (AGN). It then does the same for images in `data/processed_images/non_agn/`, assigning them a label of `0` (non-AGN).

2.  **Creating Master NumPy Arrays**: All loaded image arrays are then compiled into a single, large NumPy array, which we can call `X`. This array would have a shape like `(num_total_images, 160, 160, 3)`. Similarly, all labels are compiled into a single NumPy array `y` of shape `(num_total_images,)`.

3.  **Shuffling the Data**: Before splitting, it's essential to shuffle the `X` and `y` arrays (making sure to keep the image-label pairs intact). This randomization, performed using `sklearn.utils.shuffle` with a fixed `random_state` for reproducibility, helps ensure that our training and validation sets are representative samples of the overall dataset and not biased by the order in which files were loaded.

4.  **Splitting into Training and Validation Sets**: We then use `sklearn.model_selection.train_test_split` to divide our shuffled `X` and `y` arrays. For an initial small dataset of, say, 10 images, we might opt for an 80% training and 20% validation split. This would result in:
    *   `X_train`: `(8, 160, 160, 3)` - 8 images for training the model.
    *   `y_train`: `(8,)` - The corresponding 8 labels for training.
    *   `X_val`: `(2, 160, 160, 3)` - 2 images held back for validating the model's performance during training.
    *   `y_val`: `(2,)` - The corresponding 2 labels for validation.
    The script also includes logic to use `stratify=y` during the split if the dataset is large enough, which helps maintain the original class proportions in both the training and validation sets—a good practice especially for imbalanced datasets.

5.  **Saving the Final Datasets**: Finally, these four NumPy arrays (`X_train`, `y_train`, `X_val`, `y_val`) are saved as individual `.npy` files into a new directory, `data/ml_ready_data/`. This allows our subsequent model training scripts to easily load these ML-ready datasets.

```python
# Key steps in scripts/prepare_dataset.py
# (Conceptual - actual script uses glob to find files and loops through categories)

# all_image_arrays = [...] # List of loaded (160,160,3) image arrays
# all_labels = [...]      # List of corresponding labels (0 or 1)

# X = np.array(all_image_arrays)
# y = np.array(all_labels)

# from sklearn.utils import shuffle
# X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_shuffled, y_shuffled, test_size=0.2, random_state=42, stratify=y_shuffled # stratify if appropriate
# )

# np.save(os.path.join(ML_READY_DATA_DIR, 'X_train.npy'), X_train)
# np.save(os.path.join(ML_READY_DATA_DIR, 'y_train.npy'), y_train)
# ... and similarly for X_val, y_val
```

With these steps completed, we have neatly packaged our image data and labels into the precise structures our Keras CNN will expect for training and evaluation. Phase 1 of our data journey is now complete, and we're ready to start building our model!

### Defining Our Initial CNN Architecture

With our data prepared and structured, let's turn our attention to constructing the brain of our operation: the Convolutional Neural Network. Our script `scripts/train_cnn.py` is where we define and compile this model. For our first iteration, we're using the Keras `Sequential` API, which allows us to build a model layer by layer—a straightforward approach for a foundational CNN.

The input to our model will be the preprocessed galaxy images, each with a shape of `(160, 160, 3)`, representing 160x160 pixels in height and width, with 3 color channels (g, r, z bands).

Here's the Python code snippet from `scripts/train_cnn.py` that defines our initial model structure:

```python
# In scripts/train_cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

INPUT_SHAPE = (160, 160, 3) # Height, Width, Channels

def create_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape, name='input_layer'), # Explicit Input layer
        
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        
        # Flattening and Dense Layers
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense1'),
        Dense(1, activation='sigmoid', name='output_dense') # Sigmoid for binary classification
    ])
    return model
```

Let's walk through what each part of this architecture does:

1.  **`Input(shape=INPUT_SHAPE, name='input_layer')`**: This isn't a processing layer itself but rather defines the entry point and expected shape of the data our model will receive. In our case, it's set up for our `(160, 160, 3)` images.

2.  **`Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')`**: This is our first convolutional layer.
    *   `32`: Specifies that this layer will learn 32 different feature detectors (also known as filters or kernels).
    *   `(3, 3)`: Each filter is 3x3 pixels in size. These filters slide across the input image, detecting low-level features like edges, corners, or simple textures.
    *   `activation='relu'`: The Rectified Linear Unit (ReLU) activation function is applied to the output of the convolution. ReLU introduces non-linearity, allowing the model to learn more complex patterns. It's a common and effective choice, outputting the input directly if it's positive and zero otherwise.
    *   `padding='same'`: This ensures that the output feature map has the same height and width as the input feature map (by adding padding around the input if necessary).

3.  **`MaxPooling2D((2, 2), name='pool1')`**: After the first convolution, we apply a max-pooling layer.
    *   This layer reduces the spatial dimensions (height and width) of the feature maps by taking the maximum value within each 2x2 window.
    *   Pooling helps make the detected features more robust to their exact location in the image and reduces the number of parameters for subsequent layers, which can help control computational cost and overfitting.

4.  **`Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')`**: Our second convolutional layer. This time, it uses `64` filters. Having more filters in deeper layers allows the network to learn more complex and a greater variety of features based on the simpler features detected by the previous layer.

5.  **`MaxPooling2D((2, 2), name='pool2')`**: Another max-pooling layer, further reducing the spatial dimensions.

6.  **`Flatten(name='flatten')`**: This layer takes the multi-dimensional output from the convolutional and pooling layers (which are like grids of numbers) and flattens it into a single, long one-dimensional vector. This is necessary to transition to the standard fully-connected layers that follow.

7.  **`Dense(128, activation='relu', name='dense1')`**: This is a fully connected (or "Dense") layer with `128` neurons. Each neuron in this layer receives input from all neurons in the previous (flattened) layer. These layers are capable of learning complex relationships between the high-level features extracted by the convolutional part of the network. Again, ReLU activation is used.

8.  **`Dense(1, activation='sigmoid', name='output_dense')`**: This is our final output layer.
    *   It has `1` neuron because we are performing a binary classification task: is the galaxy hosting an AGN (class 1) or not (class 0)?
    *   `activation='sigmoid'`: The sigmoid activation function is crucial here. It squashes the output of the neuron to a value between 0 and 1. This output can be interpreted as the probability that the input image belongs to the positive class (AGN). For example, an output of 0.85 would suggest an 85% probability of the galaxy being an AGN.

After defining this architecture, calling `model.summary()` in our script provides a table listing all layers, their output shapes, and the number of trainable parameters. Our initial model has just over 2.5 million parameters to learn!

### Compiling the Model: Preparing for Training

Once the model architecture is defined, the next step is to "compile" it. This configures the learning process. In `scripts/train_cnn.py`, we compile our model like this:

```python
# In scripts/train_cnn.py
# agn_cnn_model = create_cnn_model(INPUT_SHAPE) # Model is created
agn_cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Here's what these compilation settings mean for our model:

*   **`optimizer='adam'`**: The optimizer is the algorithm that dictates how the model's internal parameters (weights and biases) are updated based on the data it sees and the loss function. `Adam` (Adaptive Moment Estimation) is a widely used and generally effective optimizer that adapts the learning rate during training, often leading to good performance without extensive tuning.

*   **`loss='binary_crossentropy'`**: The loss function measures how far the model's predictions are from the true labels during training. Since our task is to classify galaxies into two categories (AGN or non-AGN), `binary_crossentropy` is the appropriate choice. It's designed for binary (0 or 1) target variables and works well with the sigmoid activation in our output layer. The goal of training is to minimize this loss.

*   **`metrics=['accuracy']`**: Metrics are used to monitor the model's performance during training and evaluation, but they are not used by the optimizer to update the model's weights. Here, Keras is asked to track `accuracy`, which is simply the proportion of images that are correctly classified.

With our CNN model defined and compiled, it's now ready to start learning from our galaxy images. The next step in our journey is to feed it the training data and begin the training process.

### Training the Simple CNN: First Attempt (Without Augmentation)

With the model compiled, let's proceed to train it using our prepared training data (`X_train.npy`, `y_train.npy`) and validate its performance against our hold-out validation data (`X_val.npy`, `y_val.npy`). Our dataset for this initial proper training run consists of 1,598 images for training and 400 images for validation.

The training can be configured in `scripts/train_cnn.py` as follows:

```python
# In scripts/train_cnn.py (First major training run on the simple CNN)
# Assuming X_train, y_train, X_val, y_val are loaded NumPy arrays
BATCH_SIZE = 2 
EPOCHS = 15    

history = agn_cnn_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val), 
    verbose=1 
)
```

During training, Keras displays the progress for each of the 15 epochs. We can capture the training history (loss and accuracy for both training and validation sets) and plot it.

<!-- 
    Placeholder for Figure X: Learning Curves (Simple CNN, No Augmentation)
    This plot would show two subplots:
    1. Model Accuracy vs. Epoch (Training and Validation)
    2. Model Loss vs. Epoch (Training and Validation)
-->
*Caption: Learning curves for the initial simple CNN model trained for 15 epochs without data augmentation. The left plot shows Model Accuracy, and the right plot shows Model Loss. The blue lines represent training metrics, and orange lines represent validation metrics.*

The learning curves from this first run were quite revealing. The **training accuracy** (blue line in the accuracy plot) started around 72-73% and climbed steadily, reaching over 95% by the 15th epoch. Correspondingly, the **training loss** (blue line in the loss plot) started around 0.55 and decreased consistently to below 0.2. This indicated that the model was effectively learning the training data.

However, the **validation accuracy** (orange line) painted a different picture. It started around 76%, peaked at about 80% early on (epoch 2), and then fluctuated, generally not improving much and ending around 78%. More critically, the **validation loss** (orange line) started around 0.5, briefly decreased, but then began to rise significantly after epoch 4-5, eventually exceeding 1.75. This divergence—training loss decreasing while validation loss increases—is a classic sign of **overfitting**. The model was memorizing the training examples, including their noise, rather than learning generalizable features.

### Evaluating Our First Simple CNN (Without Augmentation)

After training, let's perform a detailed evaluation on the validation set (400 images) using `scripts/train_cnn.py` which includes evaluation logic.

1.  **Overall Validation Performance**:
    *   Validation Loss: ~1.9136
    *   Validation Accuracy: ~0.7775 (77.75%)

2.  **Classification Report**:
    ```
                      precision    recall  f1-score   support

         Non-AGN (0)       0.82      0.70      0.76       200
             AGN (1)       0.74      0.85      0.79       200

            accuracy                           0.78       400
           macro avg       0.78      0.78      0.78       400
        weighted avg       0.78      0.78      0.78       400
    ```

3.  **Confusion Matrix**:
    ```
    Actual vs. Predicted
                      Predicted Non-AGN  Predicted AGN
    Actual Non-AGN         141               59              
    Actual AGN              30              170              
    ```
    This translates to:
    *   True Negatives (Non-AGN correctly identified): 141
    *   False Positives (Non-AGN misclassified as AGN): 59
    *   False Negatives (AGN misclassified as Non-AGN): 30
    *   True Positives (AGN correctly identified): 170

While an accuracy of ~78% was significantly better than random guessing, the overfitting was a concern. The model was better at recalling actual AGN (85%) than Non-AGN (70%), but it also incorrectly flagged 59 Non-AGN galaxies as AGN.

### Enhancing Robustness: Adding Data Augmentation to the Simple CNN

The clear overfitting observed in our first training run prompted us to introduce **data augmentation**. The goal is to artificially expand our training dataset by creating modified versions of existing images, encouraging the model to learn more general features and become less sensitive to minor variations.

We modify our `create_cnn_model` function in `scripts/train_cnn.py` to include Keras preprocessing layers for random flipping and rotation. These layers are inserted at the beginning of the model and are only active during the training phase.

```python
# In scripts/train_cnn.py, modifying the simple CNN to include data augmentation layers
from tensorflow.keras.layers import RandomFlip, RandomRotation 

# Data Augmentation Block defined (can be a separate Sequential model or inline)
data_augmentation_layers = Sequential([
    RandomFlip("horizontal_and_vertical", name="random_flip"),
    RandomRotation(0.2, name="random_rotation"), # Rotate by up to 20% (72 degrees)
], name="data_augmentation")

# Updated create_cnn_model function
def create_cnn_model(input_shape): 
    model = Sequential([
        Input(shape=input_shape, name='input_layer'),
        data_augmentation_layers, # Apply data augmentation here
        # First Convolutional Block (remains the same as before)
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        # Second Convolutional Block (remains the same as before)
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        # Flattening and Dense Layers (remain the same as before)
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense1'),
        Dense(1, activation='sigmoid', name='output_dense') 
    ])
    return model
```
We also adjust the training parameters for this augmented run, increasing the batch size to 32 and the number of epochs to 25, to give the augmented data more opportunity to be seen by the model.

```python
# In scripts/train_cnn.py (Training run with data augmentation for the simple CNN)
BATCH_SIZE = 32 
EPOCHS = 25    
# ... model.fit() call with these new parameters ...
```

The impact of data augmentation was immediately visible in the new learning curves.

<!-- 
    Placeholder for Figure Y: Learning Curves (Simple CNN, With Augmentation)
    This plot would show two subplots:
    1. Model Accuracy vs. Epoch (Training and Validation)
    2. Model Loss vs. Epoch (Training and Validation)
-->
*Caption: Learning curves for the simple CNN model trained for 25 epochs with data augmentation. The left plot shows Model Accuracy, and the right plot shows Model Loss. Training metrics are in blue, validation metrics in orange.*

The new learning curves showed a marked improvement in training behavior. The **training accuracy** (blue) now hovered around 86%, while the **validation accuracy** (orange) tracked it much more closely, generally staying in the 78-84% range without a significant sustained drop. Most importantly, the **validation loss** (orange) no longer "exploded." It decreased and then stabilized around 0.35-0.45, staying much closer to the **training loss** (blue), which settled around 0.3-0.35. This indicated that overfitting had been significantly mitigated.

### Evaluating the Augmented Simple CNN

Let's evaluate this augmented simple CNN (using the evaluation logic in `scripts/train_cnn.py`) on the same validation set. The evaluation on the validation set yields the following:

1.  **Overall Validation Performance**:
    *   Validation Loss: ~0.4210 (a significant improvement from ~1.91)
    *   Validation Accuracy: ~0.7950 (79.5%, a modest improvement from ~77.8%)

2.  **Classification Report (Augmented Model)**:
    ```
                      precision    recall  f1-score   support

         Non-AGN (0)       0.75      0.89      0.81       200
             AGN (1)       0.86      0.70      0.77       200

            accuracy                           0.80       400
           macro avg       0.81      0.79      0.79       400
        weighted avg       0.81      0.80      0.79       400
    ```

3.  **Confusion Matrix (Augmented Model)**:
    ```
    Actual vs. Predicted
                      Predicted Non-AGN  Predicted AGN
    Actual Non-AGN         178               22              
    Actual AGN              60              140              
    ```
    This translates to:
    *   True Negatives: 178 (up from 141)
    *   False Positives: 22 (down from 59)
    *   False Negatives: 60 (up from 30)
    *   True Positives: 140 (down from 170)

Data augmentation led to a healthier training process and a slight increase in overall accuracy. Interestingly, it made the model much better at correctly identifying Non-AGN galaxies (recall for Non-AGN jumped from 70% to 89%, and False Positives dropped from 59 to 22). However, this came at the cost of correctly identifying fewer AGN (recall for AGN dropped from 85% to 70%, and False Negatives doubled). This illustrates a common trade-off: reducing overfitting and improving one aspect of classification can sometimes shift how errors are made for different classes. The choice of the "best" model here might depend on whether it's more critical to find all AGN or to avoid misclassifying non-AGN.

With these experiments on our simple CNN, we've established a baseline and seen the positive effects of data augmentation in combating overfitting. Now we're ready to explore more advanced architectures.

### Stepping Up the Architecture: Understanding ResNets

While our initial CNN and the addition of data augmentation are good starting points, the Guo et al. (2022) paper, which inspires our project, uses a more sophisticated architecture called ResNet-18. To understand why, we need to touch upon a challenge in deep learning: making networks deeper doesn't always make them better.

**The Trouble with "Plain" Deep Networks**

Intuitively, we might think that stacking more and more layers in a neural network should allow it to learn increasingly complex patterns and achieve better performance. However, in practice, simply making networks very deep (e.g., by just adding more `Conv2D` and `Dense` layers like in our first model) can lead to a couple of significant problems:

1.  **Vanishing or Exploding Gradients**: During the training process (specifically, backpropagation, where the model learns from its errors), the error signals (gradients) have to travel backward through all the layers. In very deep networks, these signals can either shrink exponentially with each step, becoming so tiny by the time they reach the early layers that these layers learn extremely slowly or not at all (the "vanishing gradient" problem). Conversely, the signals can grow exponentially and become too large, leading to unstable training (the "exploding gradient" problem).

2.  **The Degradation Problem**: Even if the gradient issues are managed, researchers found that very deep "plain" networks could perform *worse* on the training data itself compared to shallower networks. This isn't just a matter of overfitting (where a model does well on training data but poorly on new data); the deep model struggles even to learn the training examples effectively. It suggests that it's difficult for a deep stack of layers to learn even an "identity mapping" – where a layer or set of layers should ideally just pass its input through unchanged if that's the best course of action.

**ResNets and the Magic of Skip Connections**

Residual Networks, or ResNets, introduced a clever architectural element called a **residual block** to overcome these challenges. The core idea is the **skip connection** (also known as a shortcut connection).

Imagine a small block of layers within our network (say, a couple of convolutional layers followed by their activations). Let's say the input to this block is `x`. These layers will perform some transformation on `x` to produce an output, which we can call `F(x)`.

*   In a standard, "plain" network, the output of this block would simply be `F(x)`.
*   In a ResNet's residual block, we do something different. We take the original input `x` and *add* it to the output of the layers `F(x)`. So, the final output of the residual block, let's call it `H(x)`, becomes `H(x) = F(x) + x`.

The connection that allows `x` to "skip" over the main layers `F(x)` and be added back is the skip connection.

```
<!-- 
  Placeholder for a conceptual diagram of a Residual Block:

  Input (x) ---+----------------------+
               |                      |
               |  [Conv Layer -> ReLU] |
               |        |             | F(x)
               |  [Conv Layer -> ReLU] |
               |        |             |
               +---->(+)---------------> Output H(x) = F(x) + x
                     ^
                     |
                     (Skip Connection from x)
-->
```
*Caption: A conceptual diagram of a ResNet residual block. The input `x` passes through a series of transformations `F(x)` (e.g., convolutional layers) and is also added directly to `F(x)` via a "skip connection" to produce the block's output `H(x)`.*


**Why is this `F(x) + x` structure so effective?**

1.  **Learning an Identity Becomes Trivial**: If, for a particular part of the network, the optimal transformation is simply to pass the input `x` through unchanged (an identity mapping), the layers within `F(x)` can easily learn to output zeros (by setting their weights close to zero). If `F(x)` is zero, then `H(x) = 0 + x = x`. The input just flows through. For plain networks, forcing layers to learn an exact identity mapping is surprisingly difficult. ResNets make this effortless.

2.  **Better Gradient Flow**: The skip connections create direct paths for the gradient to flow backward during training. This "expressway" for gradients helps mitigate the vanishing gradient problem, allowing error signals to propagate more effectively to earlier layers, even in very deep networks.

3.  **Tackling Degradation**: Because it's easy for blocks to learn identity mappings, adding more ResNet blocks to a network is less likely to hurt performance. If an extra block isn't useful, it can effectively learn to be "skipped" by outputting a near-zero `F(x)`, and the network behaves like the shallower version.

Essentially, instead of trying to make layers learn the entire desired output from scratch, ResNets ask them to learn the *residual* – the difference or correction (`F(x)`) that needs to be applied to the input `x` to get the final desired output `H(x)`. If the input is already pretty good, the layers only need to learn a small adjustment.

This innovation allowed for the successful training of much deeper networks (like ResNet-18, ResNet-34, ResNet-50, and even deeper versions with over 100 layers) than was previously feasible, leading to significant breakthroughs in image classification and other deep learning tasks. Our next step will be to explore how we can implement a ResNet-style architecture for our AGN classification problem.

### Implementing ResNet-18 for Galaxy Classification

Armed with an understanding of residual blocks, we proceed to implement a ResNet-18 model in our `scripts/train_cnn.py` script. This architecture, while being one of the shallower variants of ResNet, is significantly more complex than our initial CNN and mirrors the approach used in the Guo et al. (2022) paper.

Building a ResNet requires a more flexible way to connect layers than the simple `Sequential` model offers. For this, we use the **Keras Functional API**, which allows for creating complex graphs of layers, including those with multiple inputs/outputs or shared layers, and importantly for us, skip connections.

**1. The Residual Block Function (`residual_block`)**

The heart of our ResNet-18 is the `residual_block` function. This Python function defines the structure of a single block:

```python
# In scripts/train_cnn.py
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add

def residual_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), use_projection=False, block_name='res_block'):
    # Main path
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', name=f'{block_name}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{block_name}_bn1')(x)
    x = Activation('relu', name=f'{block_name}_relu1')(x)

    x = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'{block_name}_conv2')(x)
    x = BatchNormalization(name=f'{block_name}_bn2')(x)

    # Shortcut path
    shortcut = input_tensor
    if use_projection: # If strides > 1 or filter counts change
        shortcut = Conv2D(filters, (1, 1), strides=strides, kernel_initializer='he_normal', name=f'{block_name}_shortcut_conv')(input_tensor)
        shortcut = BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)

    # Add shortcut to main path
    x = Add(name=f'{block_name}_add')([x, shortcut])
    x = Activation('relu', name=f'{block_name}_relu2')(x) # Final activation after addition
    return x
```

Let's break this down:
-   `input_tensor`: This is the output from the previous layer.
-   `filters`: The number of filters for the convolutional layers in this block (e.g., 64, 128, 256, 512 in ResNet-18).
-   `kernel_size=(3,3)`: We use 3x3 convolutions, standard for ResNets.
-   `strides=(1,1)`: By default, the convolutions don't reduce the image size. However, at the start of some stages, `strides` will be `(2,2)` to downsample the feature maps.
-   `use_projection=False`: This boolean flag tells the block whether the "skip connection" (the `shortcut`) needs to be modified.
    -   The `shortcut` directly takes the `input_tensor`.
    -   If `use_projection` is `True` (which happens when the main path changes the dimensions or number of filters, typically due to `strides=(2,2)` or an increase in `filters`), the `shortcut` path also gets a `Conv2D` layer (with a 1x1 kernel) and `BatchNormalization`. This ensures that the `shortcut` tensor has the same shape as the output of the main path, so they can be added together.
-   **Main Path**: Consists of two `Conv2D` layers. Each is followed by `BatchNormalization` (which helps stabilize and speed up training) and a `ReLU` activation function (though the ReLU for the second convolution effectively happens *after* the addition with the shortcut).
-   **Addition**: The `Add` layer performs the element-wise sum of the main path's output and the (possibly projected) shortcut.
-   **Final Activation**: A `ReLU` activation is applied after the addition.

**2. Assembling ResNet-18 (`create_resnet18_model`)**

With the `residual_block` function defined, we create `create_resnet18_model` in `scripts/train_cnn.py` to construct the full network:

```python
# In scripts/train_cnn.py (simplified structure)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense # ... other layers

def create_resnet18_model(input_shape, num_classes=1):
    img_input = Input(shape=input_shape, name='input_layer')
    
    # Apply data augmentation
    x = data_augmentation_layers(img_input) # Our pre-defined augmentation Sequential model

    # Initial Convolution (conv1)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv1_conv')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    # Residual Stages (conv2_x to conv5_x)
    # Stage 1 (conv2_x): 2 blocks, 64 filters
    x = residual_block(x, filters=64, block_name='conv2_block1') # strides=(1,1), use_projection=False (default)
    x = residual_block(x, filters=64, block_name='conv2_block2')

    # Stage 2 (conv3_x): 2 blocks, 128 filters
    x = residual_block(x, filters=128, strides=(2,2), use_projection=True, block_name='conv3_block1') # Downsample
    x = residual_block(x, filters=128, block_name='conv3_block2')

    # Stage 3 (conv4_x): 2 blocks, 256 filters
    x = residual_block(x, filters=256, strides=(2,2), use_projection=True, block_name='conv4_block1') # Downsample
    x = residual_block(x, filters=256, block_name='conv4_block2')

    # Stage 4 (conv5_x): 2 blocks, 512 filters
    x = residual_block(x, filters=512, strides=(2,2), use_projection=True, block_name='conv5_block1') # Downsample
    x = residual_block(x, filters=512, block_name='conv5_block2')

    # Final Layers
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', name='fc_output')(x)

    model = Model(inputs=img_input, outputs=outputs, name='resnet18')
    return model
```

Key parts of the full ResNet-18 structure:
-   An `Input` layer defines the entry point for our `(160, 160, 3)` images.
-   Our existing `data_augmentation_layers` are applied first.
-   **Initial Convolution (`conv1`)**: A larger 7x7 `Conv2D` layer with 64 filters and a stride of 2, followed by `BatchNormalization`, `ReLU`, and a `MaxPooling2D` layer. This initial block aggressively reduces the feature map size and extracts basic features.
-   **Residual Stages (`conv2_x` through `conv5_x`)**: The ResNet-18 architecture consists of four stages, each containing two residual blocks.
    -   `conv2_x`: Uses 64 filters per block.
    -   `conv3_x`: Uses 128 filters. The first block in this stage uses `strides=(2,2)` and `use_projection=True` to halve the feature map dimensions and double the filter count.
    -   `conv4_x`: Uses 256 filters, again with downsampling and projection in the first block.
    -   `conv5_x`: Uses 512 filters, with downsampling and projection in the first block.
-   **Final Layers**:
    -   `GlobalAveragePooling2D`: Instead of a `Flatten` layer, ResNets often use Global Average Pooling. This layer computes the average of each entire feature map from the last convolutional stage, resulting in a single value per feature map. This drastically reduces the number of parameters before the final Dense layer and can help prevent overfitting.
    -   `Dense`: A final `Dense` layer with a `sigmoid` activation produces our binary classification (AGN or non-AGN).

**Initial Training Run**

We compile this ResNet-18 model with the same `adam` optimizer and `binary_crossentropy` loss function as our simpler CNN (this is all handled in `scripts/train_cnn.py`). Upon running it with our very small dataset (8 training, 2 validation images), the script successfully executes, building the model (with over 11 million parameters!) and completes 15 epochs of training.

As expected with such a tiny dataset, the training and validation metrics were not particularly stable or indicative of true performance (validation accuracy ended at 0.50). However, the key outcome was confirming that our ResNet-18 implementation is structurally sound and integrates correctly into our training pipeline. This more advanced architecture was now ready for a proper test on our full dataset.

**Training and Evaluating the ResNet-18 Model**

With the ResNet-18 architecture correctly implemented in `scripts/train_cnn.py` (incorporating the `residual_block` helper and using the Keras Functional API), we proceed to train it on our dataset of 1,598 training images and 400 validation images. The same data augmentation layers (RandomFlip and RandomRotation) used with our improved simple CNN are also applied here to enhance model robustness.

The training can be configured in `scripts/train_cnn.py` for 25 epochs with a batch size of 32:

```python
# In scripts/train_cnn.py (Training the full ResNet-18 model)
BATCH_SIZE = 32 
EPOCHS = 25    

# agn_cnn_model = create_resnet18_model(input_shape=INPUT_SHAPE, num_classes=1)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = agn_cnn_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val), 
    verbose=1 
)
```

The ResNet-18 model, with its approximately 11.2 million trainable parameters, then undergoes training. After 25 epochs, we can plot the training history (saved as `output/training_history_resnet18.png`) and evaluate its performance on the unseen validation set using the evaluation functions within `scripts/train_cnn.py`.

<!-- 
    Placeholder for Figure ResNet-18: Learning Curves (ResNet-18, 25 Epochs, With Augmentation)
    This plot shows two subplots:
    1. Model Accuracy vs. Epoch (Training and Validation)
    2. Model Loss vs. Epoch (Training and Validation)
    (This would be the training_history.png from the ResNet-18 run on 2025-06-02)
-->
*Caption: Learning curves for the ResNet-18 model trained for 25 epochs with data augmentation. The left plot (Model Accuracy) shows training accuracy (blue) steadily climbing from ~70% to ~85%. Validation accuracy (orange) is more erratic, starting near 50%, showing peaks up to ~83% and ending around 83.25%. The right plot (Model Loss) shows training loss (blue) decreasing from ~0.6 to ~0.33. Validation loss (orange) starts very high (~1.9), drops sharply to ~0.6, then fluctuates between ~0.4 and ~0.8, generally staying near or below 0.4 for the latter half, ending at 0.3863.*

The learning curves for our ResNet-18 model were quite informative. The training accuracy (blue line, left plot) showed a consistent climb from around 70% to approximately 85% over the 25 epochs, indicating that the model was effectively learning from the training data. The training loss (blue line, right plot) mirrored this by steadily decreasing from about 0.6 down to around 0.33.

The validation accuracy (orange line, left plot), while more variable, demonstrated an overall upward trend. It started lower, around 50%, but showed several peaks, reaching up to 83% and ultimately finishing at 83.25%. The validation loss (orange line, right plot) began very high (near 1.9) but dropped dramatically within the first few epochs to the 0.6-0.7 range. It then continued to fluctuate, mostly staying between 0.4 and 0.8, and importantly, did not show a sustained increase that would indicate severe overfitting, ending at a respectable 0.3863. This behavior suggests that the combination of data augmentation and the inherent regularization properties of the ResNet architecture (like Batch Normalization) helped to control overfitting, even with its greater complexity.

**Evaluation Results for ResNet-18**

Upon completion of training, the ResNet-18 model achieved the following on our 400-image validation set:

1.  **Overall Validation Performance**:
    *   Validation Loss: 0.3863
    *   Validation Accuracy: 0.8325 (83.25%)

2.  **Classification Report**:
    ```
                      precision    recall  f1-score   support

         Non-AGN (0)       0.89      0.76      0.82       200
             AGN (1)       0.79      0.91      0.84       200

            accuracy                           0.83       400
           macro avg       0.84      0.83      0.83       400
        weighted avg       0.84      0.83      0.83       400
    ```

3.  **Confusion Matrix**:
    ```
    Actual vs. Predicted
                      Predicted Non-AGN  Predicted AGN
    Actual Non-AGN         152               48              
    Actual AGN              19              181              
    ```
    This translates to:
    *   True Negatives (Non-AGN correctly identified as Non-AGN): 152
    *   False Positives (Non-AGN misclassified as AGN): 48
    *   False Negatives (AGN misclassified as Non-AGN): 19
    *   True Positives (AGN correctly identified as AGN): 181

These results from the ResNet-18 model mark a significant step up from our simpler CNN. An overall validation accuracy of 83.25% is a solid achievement. The standout result is the model's **recall for AGN, which reached 91%**. This means that out of all the true AGN in our validation set, the model correctly identified 91% of them—a very effective outcome for our primary goal of finding AGN. This high recall for the AGN class did come with a precision of 79% for AGN, indicating that some Non-AGN galaxies were misclassified as AGN (48 False Positives). For the Non-AGN class, the model achieved a recall of 76% and a precision of 89%.

The F1-scores, which balance precision and recall, were 0.82 for Non-AGN and 0.84 for AGN, further underscoring the model's strong performance, particularly in identifying the target AGN class. This successful training of a ResNet-18 architecture, even on our moderately sized dataset, demonstrates the power of more sophisticated neural network designs for tackling complex astrophysical classification tasks and brings our project to a successful milestone.

### Project Reflections and Conclusion

This project has been an exciting and illuminating journey into the intersection of astrophysics and machine learning. Our primary goal was to build a Convolutional Neural Network (CNN) capable of identifying galaxies hosting Active Galactic Nuclei (AGN) based on their optical images, drawing inspiration from the work of Guo et al. (2022). We can now conclude this phase of the project with a ResNet-18 model that achieves a promising 83.25% validation accuracy and, crucially, a 91% recall for identifying AGN.

**Key Learnings and Achievements:**

Throughout this endeavor, I've navigated several key stages of a typical machine learning workflow, tailored to an astronomical context:

*   **Data Understanding and Acquisition**: The project involved delving into astronomical catalogs (MPA-JHU DR8), understanding FITS file structures and the importance of data quality flags (`RELIABLE`, `SPECTROTYPE`, `Z_WARNING`). It also covered identifying and utilizing the Legacy Survey DR8 cutout service to fetch galaxy images based on precise coordinates and BPT classifications (`bptclass`).
*   **Data Preprocessing**: I learned to handle multi-dimensional columns in Astropy tables when converting to Pandas DataFrames. For the images themselves, implementing crucial preprocessing steps was key: transposing image axes to the Keras-expected `(height, width, channels)` format and applying per-channel min-max normalization to scale pixel values to a [0, 1] range.
*   **Data Augmentation**: The importance of data augmentation (`RandomFlip`, `RandomRotation`) was clearly demonstrated. It significantly helped in mitigating overfitting, especially with the initial simpler CNN, leading to more stable training and better generalization.
*   **CNN Architecture Exploration**: The project started with a basic sequential CNN, which served as a valuable baseline. It then progressed to implementing a much more complex ResNet-18 architecture, which involved learning about residual blocks, skip connections, and the Keras Functional API. This progression clearly showed the performance benefits of more advanced architectures designed to handle deeper networks.
*   **Model Training and Evaluation**: I gained hands-on experience with the `model.fit()` process, understanding epochs and batch sizes. Learning to interpret learning curves (accuracy and loss plots) was vital to diagnose training behavior like overfitting. Furthermore, utilizing detailed evaluation metrics, including precision, recall, F1-score, and the confusion matrix, helped to understand the nuances of the model's performance beyond simple accuracy.

**Challenges Encountered:**

Like any research endeavor, this project had its share of challenges. Early on, I encountered `ValueError`s due to multidimensional columns in the FITS catalogs when converting to Pandas, which required careful filtering. Debugging model architectures, especially the more complex ResNet-18, also required attention to detail. Ensuring the data pipeline from raw FITS images to correctly shaped and normalized NumPy arrays for Keras was another critical step that involved multiple iterations.

**Personal Reflection:**

This project has been a rewarding experience, offering me a hands-on opportunity to apply deep learning techniques to a genuine scientific question. It provided me a glimpse into how these powerful tools can aid in astrophysical discovery. The process of taking raw data, shaping it, building models, and seeing them learn to classify complex objects like galaxies has been particularly satisfying for me.

In conclusion, while the achieved accuracy of 83.25% may not be the absolute state-of-the-art compared to studies using vastly larger datasets, our ResNet-18 model demonstrates a strong capability, especially in its high recall for AGN. This project successfully establishes a complete pipeline for AGN classification and serves as a testament to the learning I accomplished.

### Future Work and Potential Improvements

While this phase of the project is concluding, there are many interesting avenues for future exploration and improvement:

1.  **Massively Expanded Dataset**: The most impactful next step would undoubtedly be to train the ResNet-18 (or other advanced models) on a significantly larger dataset, aiming for tens or hundreds of thousands of images, similar to the scale used in the Guo et al. (2022) paper. This would likely lead to substantial improvements in both accuracy and generalization.
2.  **Extended Training and Advanced Hyperparameter Tuning**:
    *   **More Epochs**: Our ResNet-18 was trained for 25 epochs. Given the learning curve, further training might yield benefits, especially if combined with learning rate adjustments.
    *   **Learning Rate Scheduling**: Implementing techniques like `ReduceLROnPlateau` in Keras, where the learning rate is automatically reduced if validation loss plateaus, could help the model converge to a better solution.
    *   **Optimizer Exploration**: Experimenting with different optimizers (e.g., SGD with Nesterov momentum, AdamW) or fine-tuning the parameters of the Adam optimizer could also be beneficial.
3.  **Transfer Learning**: A very promising direction would be to leverage transfer learning. Using a model like ResNet50 (or even more advanced ones like EfficientNet) pre-trained on a massive dataset like ImageNet, and then fine-tuning it on our galaxy dataset, could potentially boost performance significantly with less data than training from scratch.
4.  **Deeper Error Analysis**: A qualitative analysis of the misclassified images (the False Positives and False Negatives) could provide valuable insights into the model's weaknesses. Are there specific types of galaxies, image artifacts, or observational conditions that confuse the model? This understanding can guide further preprocessing improvements or model adjustments.
5.  **More Sophisticated Data Augmentation**: Exploring a wider range or more fine-tuned data augmentation techniques could further enhance model robustness.
6.  **Ensemble Methods**: Combining predictions from several different models (e.g., our simple CNN and the ResNet-18, or different ResNet variants) can sometimes lead to better and more stable overall performance.
7.  **Integration with More Astrophysical Data**: For a richer scientific analysis, one could correlate the model's predictions (and prediction probabilities) with other available astrophysical parameters for the galaxies, such as their colors, redshifts, stellar masses, or environment. This could help uncover biases or reveal interesting trends in how the model perceives AGN hosts.

This project lays a solid foundation, and these potential future steps highlight the ongoing and iterative nature of machine learning research in astrophysics.
