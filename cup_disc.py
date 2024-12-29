import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from lime import lime_image
from skimage import filters, morphology
from skimage.color import rgb2gray
from skimage.feature import canny
import openpyxl  # For Excel operations

import sys
import io

# Set default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load dataset
dataset_path = r'C:\S\Glaucoma\1\Fundus_Train_Val_Data\Fundus_Scanes_Sorted'
csv_file = r'C:\S\Glaucoma\1\glaucoma.csv'
df = pd.read_csv(csv_file)

# Preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be opened: {image_path}")
        image = cv2.resize(image, target_size)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Optic Disc and Cup Segmentation
def segment_optic_disc_cup(image):
    gray_image = rgb2gray(image)
    threshold = filters.threshold_otsu(gray_image)
    binary_image = gray_image > threshold
    clean_image = morphology.remove_small_objects(binary_image, min_size=500)
    contours, _ = cv2.findContours(clean_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Calculate C/D Ratio
def calculate_cd_ratio(image):
    contours = segment_optic_disc_cup(image)
    if len(contours) < 2:
        return None  # Unable to determine both cup and disc
    # Sort contours by area (largest should be the disc)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    disc_contour = contours[0]
    cup_contour = contours[1]
    
    # Calculate diameters
    _, _, disc_w, disc_h = cv2.boundingRect(disc_contour)
    _, _, cup_w, cup_h = cv2.boundingRect(cup_contour)
    disc_diameter = (disc_w + disc_h) / 2
    cup_diameter = (cup_w + cup_h) / 2
    
    cd_ratio = cup_diameter / disc_diameter
    return cd_ratio

# Evaluate C/D Ratio
def evaluate_cd_ratio(cd_ratio):
    if cd_ratio is None:
        return "Unable to determine C/D ratio"
    elif cd_ratio < 0.4:
        return "Non-glaucomatous nerve (unless optic disc is abnormally small)"
    elif 0.4 <= cd_ratio < 0.8:
        return "Normal optic disc, glaucoma suspect, or early to moderate glaucoma"
    elif cd_ratio >= 0.8:
        return "Considered glaucomatous unless proven otherwise"

# Vessel Detection
def detect_vessels(image):
    gray_image = rgb2gray(image)
    edges = canny(gray_image)
    return edges

# Extract Retinal Image Features
def extract_features(image):
    features = {}
    gray_image = rgb2gray(image)
    features['mean_intensity'] = np.mean(gray_image)
    return features

# Create and compile a complex CNN model
def create_complex_cnn_model(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create an LSTM model
def create_lstm_model(input_shape=(10, 224, 224, 3)):
    cnn_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    cnn_base.trainable = False
    inputs = Input(shape=input_shape)
    x = TimeDistributed(cnn_base)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load images and labels
def load_data(image_dir):
    images = []
    labels = []
    for class_label in ['Glaucoma_Negative', 'Glaucoma_Positive']:
        class_path = os.path.join(image_dir, class_label)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image = preprocess_image(img_path)
            if image is not None:
                images.append(image)
                labels.append(1 if class_label == 'Glaucoma_Positive' else 0)
    return np.array(images), np.array(labels)

# Prepare data
images, labels = load_data(r'C:\S\Glaucoma\1\Fundus_Train_Val_Data\Fundus_Scanes_Sorted\Train')
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Train complex CNN model
cnn_model = create_complex_cnn_model()
train_generator = datagen.flow(X_train, y_train, batch_size=64)
cnn_history = cnn_model.fit(train_generator, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Prepare data for LSTM
# Assume we have a series of images; this part requires sequences (e.g., video frames)
# For simplicity, let's simulate it by repeating the single image 10 times in a sequence
X_train_seq = np.repeat(X_train[:, np.newaxis, :, :, :], 10, axis=1)
X_val_seq = np.repeat(X_val[:, np.newaxis, :, :, :], 10, axis=1)

# Train LSTM model
lstm_model = create_lstm_model(input_shape=(10, 224, 224, 3))
lstm_history = lstm_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=20, batch_size=16)

# Model Explainability and Interpretability
def explain_model(image, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, model.predict, top_labels=1, hide_color=0, num_samples=1000)
    return explanation

def plot_explanation(image, explanation):
    temp, mask = explanation.get_image_and_mask(label=1, positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(temp)
    plt.title('LIME Explanation')
    plt.show()

# Predict and save results to Excel
results = []
for index, row in df.iterrows():
    image_path = os.path.join(dataset_path, row['Filename'])
    image = preprocess_image(image_path)
    if image is not None:
        image_for_prediction = np.expand_dims(image, axis=0)
        prediction = cnn_model.predict(image_for_prediction)
        is_glaucoma = prediction[0][0] > 0.5
        cd_ratio = calculate_cd_ratio(image)
        cd_evaluation = evaluate_cd_ratio(cd_ratio)
        if is_glaucoma:
            results.append({
                'Filename': row['Filename'],
                'ExpCDR': row['ExpCDR'],
                'Eye': row['Eye'],
                'Set': row['Set'],
                'Glaucoma': row['Glaucoma'],
                'C/D Ratio': cd_ratio,
                'C/D Evaluation': cd_evaluation
            })

results_df = pd.DataFrame(results)
results_df.to_excel('output.xlsx', index=False)

print("Excel file 'output.xlsx' has been created with images predicted as having glaucoma.")

# Example usage
positive_folder = r'C:\S\Glaucoma\1\Fundus_Train_Val_Data\Fundus_Scanes_Sorted\Train\Glaucoma_Positive'
# List all image files in the directory
image_files = os.listdir(positive_folder)

# Process each image
for img_file in image_files:
    test_image_path = os.path.join(positive_folder, img_file)
    test_image = preprocess_image(test_image_path)
    
    if test_image is not None:
        # Perform the same processing for each image
        contours = segment_optic_disc_cup(test_image)
        vessels = detect_vessels(test_image)
        features = extract_features(test_image)
        cd_ratio = calculate_cd_ratio(test_image)
        cd_evaluation = evaluate_cd_ratio(cd_ratio)
        explanation = explain_model(test_image, cnn_model)
        plot_explanation(test_image, explanation)

        # Print extracted features and contours
        print(f"Extracted Features for {img_file}:", features)
        print(f"Optic Disc and Cup Contours for {img_file}:", contours)
        print(f"C/D Ratio for {img_file}: {cd_ratio}, Evaluation: {cd_evaluation}")

        # Display images with segmentation and vessel detection
        plt.imshow(test_image)
        plt.title(f'Optic Disc and Cup Segmentation for {img_file}')
        plt.show()
        
        plt.imshow(vessels, cmap='gray')
        plt.title(f'Vessel Detection for {img_file}')
        plt.show()
    else:
        print(f"Image not found or could not be opened: {test_image_path}")

# Plot training history
def plot_training_history(history, model_name='Model'):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot history for CNN
plot_training_history(cnn_history, model_name='Complex CNN')

# Plot history for LSTM
plot_training_history(lstm_history, model_name='LSTM')
