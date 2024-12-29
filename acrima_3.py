import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import logging
from sklearn.linear_model import LogisticRegression

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
N_SPLITS = 5

def create_image_data_flow(image_folder):
    """Load and preprocess images from the given folder."""
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png')):
            try:
                label = 1 if 'g' in filename else 0
                image_path = os.path.join(image_folder, filename)
                img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                logging.error(f"Error processing image {filename}: {str(e)}")
    
    return np.array(images), np.array(labels)

ddef create_model(model_type='vgg16', fine_tune=True):
    """Create a model based on the specified type with regularization."""
    if model_type == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    elif model_type == 'densenet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    else:
        raise ValueError("Invalid model type. Choose 'vgg16', 'resnet50', or 'densenet121'")

    if not fine_tune:
        base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

def train_model_with_kfold(model_creator, X, y, n_splits=N_SPLITS, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """Train the model using K-fold cross-validation with enhanced regularization and data augmentation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logging.info(f"Training on fold {fold+1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_creator()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc', mode='max'),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, monitor='val_auc', mode='max'),
            ModelCheckpoint(f'best_model_fold_{fold+1}.h5', save_best_only=True, monitor='val_auc', mode='max')
        ]
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,  # Usually not flipped vertically for medical images
            zoom_range=0.15,
            shear_range=0.15,
            fill_mode='nearest'
        )
        
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
        
        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        histories.append(history)
    
    return histories

def plot_kfold_history(histories, model_name):
    """Plot the training history for K-fold cross-validation."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_accuracy'], label=f'Fold {i+1} Val')
    plt.title(f'{model_name} Accuracy (K-fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_loss'], label=f'Fold {i+1} Val')
    plt.title(f'{model_name} Loss (K-fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_kfold_history.png')
    plt.close()

def lime_explanation(model, X, y, num_images=10, output_dir=None):
    """Generate LIME explanations for the specified number of images."""
    explainer = lime_image.LimeImageExplainer()

    for idx in range(num_images):
        explanation = explainer.explain_instance(X[idx], model.predict, top_labels=1, hide_color=0, num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(X[idx])
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f"LIME Explanation (Predicted: {model.predict(X[idx].reshape(1, IMG_HEIGHT, IMG_WIDTH, 3))[0][0]:.2f}, Actual: {y[idx]})")
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"LIME_explanation_image_{idx+1}.png"))
        plt.close()

def grad_cam(model, img_array, layer_name='block5_conv3'):
    """Generate Grad-CAM heatmap for the given image."""
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_grad_cam(model, X, y, num_images=10, output_dir=None, layer_name='block5_conv3'):
    """Generate Grad-CAM visualizations for the specified number of images."""
    for idx in range(num_images):
        img_array = np.expand_dims(X[idx], axis=0)
        heatmap = grad_cam(model, img_array, layer_name)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(X[idx])
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(X[idx])
        plt.imshow(heatmap, alpha=0.6, cmap='jet')
        plt.title(f"Grad-CAM (Predicted: {model.predict(img_array)[0][0]:.2f}, Actual: {y[idx]})")
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"GradCAM_image_{idx+1}.png"))
        plt.close()

def main():
    # Load and preprocess data
    image_folder_path = '/content/drive/MyDrive/21BDS0137/ACRIMA/Images'
    all_images, all_labels = create_image_data_flow(image_folder_path)
    
    # Train models with different learning rates
    learning_rates = [1e-4, 5e-5, 1e-5]
    for lr in learning_rates:
        for model_name, model_creator in models.items():
            logging.info(f"Training {model_name} with learning rate {lr}")
            model_creator_with_lr = lambda: create_model(model_name)
            model_creator_with_lr().optimizer.learning_rate = lr
            histories = train_model_with_kfold(model_creator_with_lr, X_train, y_train)
            plot_kfold_history(histories, f"{model_name}_lr_{lr}")
            
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'VGG16': lambda: create_model('vgg16'),
        'ResNet50': lambda: create_model('resnet50'),
        'DenseNet121': lambda: create_model('densenet121')
    }
    
    for model_name, model_creator in models.items():
        logging.info(f"Training {model_name}")
        histories = train_model_with_kfold(model_creator, X_train, y_train)
        plot_kfold_history(histories, model_name)
    
    # Train baseline model (Logistic Regression)
    logging.info("Training Logistic Regression baseline")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_flat, y_train)
    lr_preds = lr_model.predict(X_test_flat)
    lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_flat)[:, 1])
    logging.info(f"Logistic Regression AUC: {lr_auc:.4f}")
    logging.info("\n" + classification_report(y_test, lr_preds))
    
    # Generate explanations
    best_model = create_model('vgg16')  # Assuming VGG16 performed best
    best_model.load_weights('best_model_fold_1.h5')  # Load the best model weights
    
    output_dir = '/content/drive/MyDrive/ACRIMA/Explanations'
    os.makedirs(output_dir, exist_ok=True)
    
    lime_explanation(best_model, X_test, y_test, num_images=10, output_dir=output_dir)
    generate_grad_cam(best_model, X_test, y_test, num_images=10, output_dir=output_dir, layer_name='block5_conv3')

if __name__ == "__main__":
    main()