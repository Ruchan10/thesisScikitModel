import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import pandas as pd


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Define paths for datasets
train_dir = 'train_dataset'
validation_dir = 'val_dataset'

# Get all images in the dataset
all_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]

# Seperate the labels
all_labels = [img.split('_')[-1].split('.')[0] for img in all_images]

# Check if the lengths of 'all_images' and 'all_labels' are the same
if len(all_images) != len(all_labels):
    raise ValueError("Length mismatch between 'all_images' and 'all_labels'")

# Create a DataFrame with the paths and labels
data = pd.DataFrame({'path': all_images, 'label': all_labels})

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


# Define parameters for images
img_size = (224, 224)
batch_size = 4  # number of samples

# Create data generators with data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,  # normalization step to scale pixel value
    rotation_range=20,  # rotate image to make the modal more robust
    width_shift_range=0.2, # shift the image horizontally
    height_shift_range=0.2, # shift the image vertically
    shear_range=0.2, # transform the image size
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' # fill the image with nearest pixel value
)

# Using 'flow_from_dataframe'
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # or 'binary' if two classes
    subset='training'
)

# Create data generator for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    val_data,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build a simple Convolutional Neural Network (CNN)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 1 output neuron for binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Create a DataFrame from the training history
history_df = pd.DataFrame(history.history)

# Display the DataFrame
print("Training History:")
print(history_df)

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Save the model
model.save('xray_classification_model.h5')
