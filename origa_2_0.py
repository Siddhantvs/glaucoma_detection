import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import classification_report
from lime import lime_image
import cv2
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from google.colab.patches import cv2_imshow
from skimage.segmentation import mark_boundaries

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to images
image_folder_path = '/content/drive/MyDrive/21BDS0137/ACRIMA/Images'

# Set image dimensions
img_height, img_width = 224, 224

# Load and preprocess images
def create_image_data_flow(image_folder):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Determine label based on filename
            if 'g' in filename:
                label = 1  # Glaucoma positive
            else:
                label = 0  # Glaucoma negative
            
            # Load image and preprocess
            image_path = os.path.join(image_folder, filename)
            img = load_img(image_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Define file path and load data
image_folder_path = '/content/drive/MyDrive/21BDS0137/ACRIMA/Images'
all_images, all_labels = create_image_data_flow(image_folder_path)

# Check if data is correctly loaded
print("Images shape:", all_images.shape)
print("Labels shape:", all_labels.shape)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Check if split is correct
print("Training images shape:", train_images.shape)
print("Validation images shape:", val_images.shape)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model definitions
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_densenet121_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Create models
vgg16_model = create_vgg16_model()
resnet50_model = create_resnet50_model()
densenet121_model = create_densenet121_model()


# Train models
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history

# Train each model
vgg16_history = train_model(vgg16_model, X_train, y_train, X_val, y_val)
resnet50_history = train_model(resnet50_model, X_train, y_train, X_val, y_val)
densenet121_history = train_model(densenet121_model, X_train, y_train, X_val, y_val)

# Plot accuracy and loss for each model
def plot_history(history, model_name):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_history(vgg16_history, 'VGG16')
plot_history(resnet50_history, 'ResNet50')
plot_history(densenet121_history, 'DenseNet121')

# Logistic Regression
train_images_flatten = train_images.reshape(len(train_images), -1)
val_images_flatten = val_images.reshape(len(val_images), -1)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(train_images_flatten, train_labels)
logistic_preds = logistic_model.predict(val_images_flatten)
print(classification_report(val_labels, logistic_preds))

# Simple neural network
nn_model = Sequential()
nn_model.add(Dense(128, activation='relu', input_shape=(224*224*3,)))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nn_model.fit(train_images_flatten, train_labels, validation_data=(val_images_flatten, val_labels), epochs=40)

# LIME Explanation
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(train_images[0], vgg_model.predict, top_labels=1)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp, mask))
plt.show()

# Grad-CAM Function
def grad_cam(base_model, img_array, layer_name, class_idx=None):
    grad_model = tf.keras.models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    conv_outputs *= pooled_grads

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Example Grad-CAM Application
img_array = np.expand_dims(val_images[0], axis=0)
heatmap = grad_cam(VGG16(weights='imagenet', include_top=True), img_array, 'block5_conv3')  # Adjust layer name as needed

def display_superimposed_image(img_array, heatmap, alpha=0.4):
    img = img_array[0]
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2_imshow(superimposed_img)

display_superimposed_image(val_images, heatmap)

# SHAP Explanation
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))]

# Predict using ViT
for image_path in image_paths[:10]:  # Limit to first 10 images
    inputs = feature_extractor(images=Image.open(image_path), return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")

# Integrated Gradients
def integrated_gradients(model, img_input, target_class_idx, baseline=None, num_steps=50, batch_size=1):
    if baseline is None:
        baseline = np.zeros_like(img_input)

    img_input = tf.convert_to_tensor(img_input, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    scaled_inputs = [(baseline + (float(i) / num_steps) * (img_input - baseline)) for i in range(num_steps + 1)]

    grads = np.zeros_like(img_input)

    for i in range(0, len(scaled_inputs), batch_size):
        batch_inputs = tf.convert_to_tensor(np.array(scaled_inputs[i:i + batch_size]).squeeze(), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(batch_inputs)
            preds = model(batch_inputs, training=False)
            one_hot_labels = tf.one_hot([target_class_idx] * batch_inputs.shape[0], preds.shape[-1])
            loss = tf.keras.losses.categorical_crossentropy(one_hot_labels, preds, from_logits=True)

        grads_batch = tape.gradient(loss, batch_inputs)
        grads += np.sum(grads_batch.numpy(), axis=0) / num_steps

    integrated_gradients = (img_input - baseline) * grads
    return integrated_gradients.numpy()[0]

img_input = np.expand_dims(val_images[0], axis=0)
target_class_idx = np.argmax(VGG16(weights='imagenet', include_top=True).predict(img_input))
integrated_grad = integrated_gradients(VGG16(weights='imagenet', include_top=True), img_input, target_class_idx)

plt.imshow(np.squeeze(integrated_grad[0]), cmap='viridis')
plt.colorbar()
plt.show()
