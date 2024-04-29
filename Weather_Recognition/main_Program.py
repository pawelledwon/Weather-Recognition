from input_Data import *
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow

print(tensorflow.config.list_physical_devices())

image_size_x = 180
image_size_y = 180

weather_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
training_labels, training_images = read_data_set(weather_names, 'dataset', (image_size_x, image_size_y))
testing_labels, testing_images = read_data_set(weather_names, 'testset', (image_size_x, image_size_y))

training_images = training_images/255

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(weather_names[training_labels[i]])

plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(image_size_x,image_size_y,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(training_images, training_labels, epochs=10,validation_data=(testing_images, testing_labels), batch_size=32)

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.h5')




