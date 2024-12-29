import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from skimage.segmentation import mark_boundaries
from lime import lime_image

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
    vertical_flip=False,  # Usually not flipped vertically for medical images
    fill_mode='nearest'
)

# model definitions with regularization and dropout
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_densenet121_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Updated training function with K-fold cross-validation
def train_model_with_kfold(model_creator, X, y, n_splits=5, batch_size=32, epochs=50):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training on fold {fold+1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_creator()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=5),
            ModelCheckpoint(f'best_model_fold_{fold+1}.h5', save_best_only=True)
        ]
        
        # Use data augmentation for training data
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

# Train models with K-fold cross-validation
vgg16_histories = train_model_with_kfold(create_vgg16_model, all_images, all_labels)
resnet50_histories = train_model_with_kfold(create_resnet50_model, all_images, all_labels)
densenet121_histories = train_model_with_kfold(create_densenet121_model, all_images, all_labels)

# Updated plotting function for K-fold results
def plot_kfold_history(histories, model_name):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_accuracy'], label=f'Fold {i+1} Val')
    plt.title(f'{model_name} Accuracy (K-fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Fold {i+1} Train')
        plt.plot(history.history['val_loss'], label=f'Fold {i+1} Val')
    plt.title(f'{model_name} Loss (K-fold)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_kfold_history(vgg16_histories, 'VGG16')
plot_kfold_history(resnet50_histories, 'ResNet50')
plot_kfold_history(densenet121_histories, 'DenseNet121')

# Function to calculate average performance across folds
def average_kfold_performance(histories):
    final_val_accuracies = [h.history['val_accuracy'][-1] for h in histories]
    final_val_losses = [h.history['val_loss'][-1] for h in histories]
    
    avg_val_accuracy = np.mean(final_val_accuracies)
    avg_val_loss = np.mean(final_val_losses)
    
    return avg_val_accuracy, avg_val_loss

print("VGG16 Average Performance:", average_kfold_performance(vgg16_histories))
print("ResNet50 Average Performance:", average_kfold_performance(resnet50_histories))
print("DenseNet121 Average Performance:", average_kfold_performance(densenet121_histories))

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

# LIME Explanation for 100 Images
def lime_explanation(model, validation_generator, num_images=100, output_dir=None):
    explainer = lime_image.LimeImageExplainer()

    for idx in range(num_images):
        img_path = validation_generator.filenames[idx]  # Get image path from the validation generator
        img = load_img(os.path.join('/content/drive/MyDrive/ACRIMA', img_path), target_size=(224, 224))  # Load and resize image
        img_array = img_to_array(img).astype('double')  # Convert image to array

        explanation = explainer.explain_instance(img_array, model.predict, top_labels=1, hide_color=0, num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

        # Display the LIME explanation for each image
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f"LIME Explanation for Image {idx+1}")
        plt.show()

        # Optionally, save each figure
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"LIME_explanation_image_{idx+1}.png"))
        plt.close()

# Grad-CAM Function
def grad_cam(base_model, img_array, layer_name='block5_conv3', class_idx=None):
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

# Grad-CAM for 100 Images
def generate_grad_cam(model, validation_generator, num_images=100, output_dir=None, layer_name='block5_conv3'):
    for idx in range(num_images):
        img_path = validation_generator.filenames[idx]
        img = load_img(os.path.join('/content/drive/MyDrive/ACRIMA', img_path), target_size=(224, 224))
        img_array = np.expand_dims(img_to_array(img), axis=0) / 255.0  # Normalize image

        # Generate Grad-CAM for the image
        grad_cam_image = grad_cam(model, img_array, layer_name)

        # Display Grad-CAM heatmap
        plt.imshow(grad_cam_image)
        plt.title(f"Grad-CAM for Image {idx+1}")
        plt.show()

        # Optionally, save each figure
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"GradCAM_image_{idx+1}.png"))
        plt.close()

# Directory to save the images
output_dir = '/content/drive/MyDrive/ACRIMA/Explanations'  # Adjust the path where you want to save the results

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Call LIME explanation and Grad-CAM generation for 100 images
lime_explanation(model, validation_generator, num_images=100, output_dir=output_dir)
generate_grad_cam(model, validation_generator, num_images=100, output_dir=output_dir, layer_name='block5_conv3')

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
