import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import tensorflow_addons as tfa

# Define functions
def create_image_data_flow(image_folder):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png')):  # Add other extensions if needed
            img_path = os.path.join(image_folder, filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            # Placeholder for labels, assuming you have a way to assign them
            labels.append(0)  # Replace with actual label extraction logic

    return np.array(images), np.array(labels)

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def grad_cam(model, img_array, layer_name, class_idx=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
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

def display_superimposed_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imshow('Superimposed Image', superimposed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load data
image_folder_path = '/content/drive/MyDrive/21BDS0137/ACRIMA/Images'
all_images, all_labels = create_image_data_flow(image_folder_path)

# Split data
split_idx = int(0.8 * len(all_images))
train_images, val_images = all_images[:split_idx], all_images[split_idx:]
train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

# Define VGG16 model with custom classification head
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
vgg_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = vgg_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=40, callbacks=[early_stopping])

# Plot training history
plot_history(history)

# Evaluate model
def evaluate_model(model, val_images, val_labels):
    val_preds = model.predict(val_images)
    val_preds = (val_preds > 0.5).astype(int)  # Convert probabilities to binary labels
    accuracy = accuracy_score(val_labels, val_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

# Evaluate the model
accuracy = evaluate_model(vgg_model, val_images, val_labels)

# Grad-CAM example
layer_name = 'block5_conv3'  # Change if necessary
img_path = '/content/drive/MyDrive/21BDS0137/ACRIMA/Images/' + 'example_image.jpg'  # Replace with an actual image filename
img_array = np.expand_dims(val_images[0], axis=0)  # Use a validation image
heatmap = grad_cam(vgg_model, img_array, layer_name)

# Display superimposed image
display_superimposed_image(img_path, heatmap)
