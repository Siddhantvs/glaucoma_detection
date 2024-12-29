import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime import lime_image
from lime.wrappers.scikit_image import mark_boundaries
import shap
import cv2
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Heatmap and Grad-CAM functions
def generate_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    last_conv_layer_output *= pooled_grads

    heatmap = tf.reduce_mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def display_superimposed_image(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imshow('Superimposed Image', superimposed_img)
    cv2.waitKey(0)

# Data preparation
def create_image_data_flow(df, image_folder):
    images = []
    labels = []
    cdr_values = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['Filename'])
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row['Glaucoma'])
        cdr_values.append(row['ExpCDR'])
    
    return np.array(images), np.array(labels), np.array(cdr_values)

# Path to images
image_folder_path = '/content/drive/MyDrive/21BDS0137/ORIGA/ORIGA/Images'
df = pd.read_csv('/path/to/glaucoma.csv')  # Update path
all_images, all_labels, all_cdr = create_image_data_flow(df, image_folder_path)

# Check if data is correctly loaded
print("Images shape:", all_images.shape)
print("Labels shape:", all_labels.shape)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Check if split is correct
print("Training images shape:", train_images.shape)
print("Validation images shape:", val_images.shape)

# Build and train transfer learning models
def build_transfer_model(base_model):
    base_model.trainable = False  # Freeze base model
    model = Sequential([base_model, Flatten(), Dense(128, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# VGG16
vgg_model = build_transfer_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
vgg_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=40)

# ResNet50
resnet_model = build_transfer_model(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
resnet_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=40)

# DenseNet121
densenet_model = build_transfer_model(DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
densenet_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=40)

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

# SHAP Explanation
shap.initjs()
explainer = shap.DeepExplainer(vgg_model, train_images[:50])  # Reduced the number of samples
shap_values = explainer.shap_values(val_images[:10])  # Use fewer validation images
shap.image_plot(shap_values, val_images[:10])

# Transformers (ViT)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=[Image.open(image_folder_path)], return_tensors="pt")
outputs = model(**inputs)
print(outputs)

# Grad-CAM and XAI (Explainable AI)
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

img_array = np.expand_dims(train_images[0], axis=0)
heatmap = grad_cam(vgg_model, img_array, 'block5_conv3')  # Example: Replace with actual conv layer name
display_superimposed_image('/content/drive/MyDrive/21BDS0137/ORIGA/ORIGA/Images/' + df.iloc[0]['Filename'], heatmap)  # Use the path of one image

# Integrated Gradients
def integrated_gradients(model, img_input, target_class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros(img_input.shape)

    interpolated_images = [
        baseline + (step / steps) * (img_input - baseline)
        for step in range(steps + 1)
    ]

    interpolated_images = np.array(interpolated_images)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        logits = model(interpolated_images)
        probs = tf.nn.softmax(logits, axis=-1)
        target_class_probs = probs[:, target_class_idx]
    
    grads = tape.gradient(target_class_probs, interpolated_images)
    avg_grads = np.average(grads, axis=0)
    integrated_grad = (img_input - baseline) * avg_grads
    return integrated_grad

img_input = np.expand_dims(train_images[0], axis=0)
target_class_idx = np.argmax(vgg_model.predict(img_input))
integrated_grad = integrated_gradients(vgg_model, img_input, target_class_idx)
plt.imshow(integrated_grad[0])
plt.show()

# DeepLIFT
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import compile_func
from deeplift.models.keras_model import KerasModel

def deep_lift(model, img_input):
    deeplift_model = KerasModel(model, nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT)
    
    find_scores_layer_idx = 0  # Usually the input layer
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx)

    baseline = np.zeros(img_input.shape)
    deeplift_scores = deeplift_contribs_func(task_idx=0, input_data_list=[img_input], input_references_list=[baseline], batch_size=1)
    return deeplift_scores[0]

img_input = np.expand_dims(train_images[0], axis=0)
deeplift_scores = deep_lift(vgg_model, img_input)
plt.imshow(deeplift_scores[0])
plt.show()
