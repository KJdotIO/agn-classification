"""
Trains a Convolutional Neural Network (CNN), specifically a ResNet-18 variant,
for classifying galaxies as AGN hosts or non-AGN based on their images.

This script performs the following main operations:
1.  Loads pre-prepared training and validation datasets (X_train, y_train, X_val, y_val)
    from .npy files located in `../data/ml_ready_data/`.
2.  Defines the ResNet-18 model architecture, including:
    - Data augmentation layers (RandomFlip, RandomRotation).
    - A helper function for creating residual blocks.
    - The main ResNet-18 model assembled using the Keras Functional API.
3.  Compiles the model with an optimizer (Adam), loss function (binary_crossentropy),
    and metrics (accuracy).
4.  Prints a summary of the model architecture.
5.  Trains the model using `model.fit()` on the training data, validating on the
    validation data.
6.  Plots the training history (accuracy and loss vs. epochs) and saves the plot
    to `../output/`.
7.  Evaluates the trained model on the validation set, printing loss, accuracy,
    a classification report, and a confusion matrix.
8.  Saves the trained Keras model to `../output/`.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, 
    RandomFlip, RandomRotation, BatchNormalization, 
    Activation, Add, GlobalAveragePooling2D
)
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
INPUT_SHAPE = (160, 160, 3)  # Expected image shape: (Height, Width, Channels)
ML_READY_DATA_DIR = "../data/ml_ready_data/"
BATCH_SIZE = 32
EPOCHS = 25
MODEL_SAVE_PATH = "../output/agn_resnet18_model.keras"
PLOT_SAVE_PATH = "../output/training_history_resnet18.png"

# --- Data Augmentation Layers ---
# A small Sequential model for applying on-the-fly data augmentation during training.
# These layers are only active during training, not during inference.
data_augmentation_layers = Sequential([
    RandomFlip("horizontal_and_vertical", name="random_flip"),
    RandomRotation(0.2, name="random_rotation"), # Rotates by up to 20% of 360 degrees (72 degrees)
], name="data_augmentation")

# --- ResNet-18 Model Definition ---

def residual_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), use_projection=False, block_name='res_block'):
    """
    Creates a single residual block as used in ResNet architectures.

    A residual block consists of a main path with convolutions and batch normalization,
    and a shortcut path (skip connection). The outputs of these two paths are added.

    Args:
        input_tensor: The input tensor to the block.
        filters (int): The number of filters for the convolutional layers.
        kernel_size (tuple): Size of the convolutional kernels (default: (3,3)).
        strides (tuple): Strides for the first convolution in the main path 
                         (default: (1,1)). If (2,2), downsampling occurs.
        use_projection (bool): If True, a 1x1 convolution is applied to the shortcut
                               path to match dimensions if `strides` > 1 or if the 
                               number of filters changes. (Default: False)
        block_name (str): A string prefix for naming the layers within this block.

    Returns:
        Tensor: The output tensor of the residual block.
    """
    # Main path
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', name=f'{block_name}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{block_name}_bn1')(x)
    x = Activation('relu', name=f'{block_name}_relu1')(x)

    x = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', name=f'{block_name}_conv2')(x)
    x = BatchNormalization(name=f'{block_name}_bn2')(x)

    # Shortcut path (the "skip connection")
    shortcut = input_tensor
    if use_projection:
        # This 1x1 conv is the "projection shortcut" needed when dimensions or filter counts change.
        shortcut = Conv2D(filters, (1, 1), strides=strides, kernel_initializer='he_normal', name=f'{block_name}_shortcut_conv')(input_tensor)
        shortcut = BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)

    x = Add(name=f'{block_name}_add')([x, shortcut]) # Element-wise addition
    x = Activation('relu', name=f'{block_name}_relu2')(x) # Final activation after addition
    return x

def create_resnet18_model(input_shape, num_classes=1):
    """
    Constructs a ResNet-18 model architecture using the Keras Functional API.

    The model includes an initial convolutional block, followed by four stages
    of residual blocks, and finally global average pooling and a dense output layer.
    Data augmentation layers are applied at the beginning.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes (1 for binary classification).

    Returns:
        keras.Model: The constructed ResNet-18 model.
    """
    img_input = Input(shape=input_shape, name='input_layer')
    
    # Apply data augmentation first
    x = data_augmentation_layers(img_input)

    # Initial Convolutional Block (conv1)
    # This block aggressively reduces feature map size and extracts low-level features.
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv1_conv')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    # Residual Stages: ResNet-18 typically has [2, 2, 2, 2] blocks in its stages.
    # Stage 1 (conv2_x): 2 blocks, 64 filters
    # No projection needed for the first block if input filters match (which they do from conv1).
    x = residual_block(x, filters=64, use_projection=False, block_name='conv2_block1') 
    x = residual_block(x, filters=64, block_name='conv2_block2')

    # Stage 2 (conv3_x): 2 blocks, 128 filters
    # Downsampling occurs at the start of this stage, so projection is needed.
    x = residual_block(x, filters=128, strides=(2,2), use_projection=True, block_name='conv3_block1') 
    x = residual_block(x, filters=128, block_name='conv3_block2')

    # Stage 3 (conv4_x): 2 blocks, 256 filters
    x = residual_block(x, filters=256, strides=(2,2), use_projection=True, block_name='conv4_block1')
    x = residual_block(x, filters=256, block_name='conv4_block2')

    # Stage 4 (conv5_x): 2 blocks, 512 filters
    x = residual_block(x, filters=512, strides=(2,2), use_projection=True, block_name='conv5_block1')
    x = residual_block(x, filters=512, block_name='conv5_block2')

    # Final Layers
    x = GlobalAveragePooling2D(name='avg_pool')(x) # Reduces each feature map to a single value
    # Output layer: 1 neuron with sigmoid for binary AGN/non-AGN classification.
    outputs = Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal', name='fc_output')(x)

    model = Model(inputs=img_input, outputs=outputs, name='resnet18_agn_classifier')
    return model

def plot_training_history(history, save_path):
    """Plots and saves the training and validation accuracy and loss."""
    print("\n--- Plotting Training History ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Training & Validation Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot Training & Validation Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Training history plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    # plt.show() # Usually commented out for automated runs

def main():
    """Main script execution for training and evaluating the ResNet-18 model."""
    print("--- Starting AGN Classification Model Training (ResNet-18) ---")

    # 1. Load Data
    print(f"\nLoading data from: {ML_READY_DATA_DIR}")
    try:
        X_train = np.load(os.path.join(ML_READY_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(ML_READY_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(ML_READY_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(ML_READY_DATA_DIR, 'y_val.npy'))
        print("  Data loaded successfully:")
        print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"    X_val: {X_val.shape}, y_val: {y_val.shape}")
    except FileNotFoundError:
        print(f"Error: Not all .npy data files found in {ML_READY_DATA_DIR}. Please run prepare_dataset.py.")
        return
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        return

    # 2. Create and Compile Model
    print("\n--- Creating and Compiling ResNet-18 Model ---")
    model = create_resnet18_model(input_shape=INPUT_SHAPE, num_classes=1)
    model.compile(
        optimizer='adam', # Adam is a good general-purpose optimizer
        loss='binary_crossentropy', # Standard for binary classification
        metrics=['accuracy'] # To monitor performance
    )
    model.summary() # Display model architecture

    # 3. Train the Model
    print("\n--- Training the Model ---")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1 # Show progress bar during training
    )
    print("--- Model Training Complete ---")

    # 4. Plot Training History
    plot_training_history(history, PLOT_SAVE_PATH)

    # 5. Evaluate the Model
    print("\n--- Evaluating Model on Validation Set ---")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")

    y_pred_proba = model.predict(X_val)
    y_pred_classes = (y_pred_proba > 0.5).astype("int32") # Convert probabilities to 0 or 1

    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=['Non-AGN (0)', 'AGN (1)']))
    
    print("  Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))

    # 6. Save the Trained Model
    print(f"\n--- Saving Trained Model to {MODEL_SAVE_PATH} ---")
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"  Model saved successfully to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"  Error saving model: {e}")

    print("\n--- AGN Classification Training Script Finished ---")

if __name__ == "__main__":
    main()