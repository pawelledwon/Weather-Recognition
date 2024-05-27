from input_Data import *
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

print(tf.config.list_physical_devices())

image_size_x = 180
image_size_y = 180

weather_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
labels, images = read_data_set(weather_names, 'dataset', (image_size_x, image_size_y))

# Normalize the images
images = images / 255.0

# Split the data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2, shuffle=True)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the datagen to the training data
datagen.fit(x_train)

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_valid.shape}")

# Model definition
model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(image_size_x, image_size_y, 3), kernel_regularizer=l2(0.0005)))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=75, validation_data=(x_valid, y_valid))

loss, accuracy = model.evaluate(x_valid, y_valid)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.h5')
