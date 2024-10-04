import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt



source_folder = 'C://Users//Moe//Desktop//School//W2024//Machine Learning//Final Project//_PROPER//img_full_face'

dim = 32 #Store dimension of images 
random.seed(314) #Set Seed
tf.random.set_seed(1618) #Set Seed

# Function to remove numbers from a string
def remove_numbers(s):
    return ''.join([i for i in s if not i.isdigit()])

data = []
for filename in os.listdir(source_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')) and ( ("Butler" in filename) or ("Radcliffe" in filename) or ("Vartan" in filename) or ("Bracco" in filename) or ("Gilpin" in filename) or ("Harmon" in filename)):  # Check for image files
        # Load the image
        img = Image.open(os.path.join(source_folder, filename)).convert('L') #grayscale
        img = img.resize((dim, dim))

        # Convert the image to a numpy array and append to the training set with the label
        img_array = np.array(img)
        last_name = remove_numbers(filename.split('_')[1]).lower()
        if  last_name == 'bracco' or last_name == 'gilpin' or last_name == 'harmon':
            label = 1
        elif last_name == 'butler' or last_name == 'radcliffe' or last_name == 'vartan':
            label = 0
        data.append((img_array, label))
print(data)

# Group images by label
images_by_label = {}
for image_array, label in data:
    if label not in images_by_label:
        images_by_label[label] = []
    images_by_label[label].append(image_array)

train_data = []
test_data = []

for label, images in images_by_label.items():
    # Split the images for each label into 80% training and 20% testing
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=271)
    
    # Add images and labels to the respective datasets
    train_data.extend([(img, label) for img in train_images])
    test_data.extend([(img, label) for img in test_images])


# Define input shape
input_shape = (dim, dim, 1)

# Prepare data
X_train = np.array([img for img, label in train_data])
X_test = np.array([img for img, label in test_data])

# Reshape for CNN input
X_train = X_train.reshape(-1, dim, dim, 1)
X_test = X_test.reshape(-1, dim, dim, 1)

# Normalize image data
X_train, X_test = X_train / 255.0, X_test / 255.0

y_train = np.array([label for img, label in train_data])
y_test = np.array([label for img, label in test_data])

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(3, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(6, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', #perhaps try sgd
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Fit the model on the training data
history = model.fit(X_train, y_train, epochs=64)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print("Test accuracy: ", round(test_acc*100,2), "%")

# Code for Feature Map Visualization. Adapted from <https://www.kaggle.com/code/arpitjain007/guide-to-visualize-filters-and-feature-maps-in-cnn>
layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

img_to_visualize = np.expand_dims(X_test[3], axis=0)  # Choose an image to pass through the model
plt.imshow(X_test[3], cmap = "grey")

features = activation_model.predict(img_to_visualize)

for conv_layer in range(0,3):
    first_layer_feature_maps = features[conv_layer]  #Select Convolutional Layer

    fig = pyplot.figure(figsize=(32, 32))
    n_features = first_layer_feature_maps.shape[-1]  # Number of feature maps
    for i in range(1, n_features + 1):
        pyplot.subplot(3, 3, i) 
        pyplot.imshow(first_layer_feature_maps[0, :, :, i - 1], cmap='coolwarm')
        pyplot.axis('off')  

    pyplot.show()

from IPython.display import Image
plot_model(model, to_file='gender_model_architecture.png', show_shapes=True, show_layer_names=True)

pred = model.predict(img_to_visualize)
print('Probability Female: ', (round(pred[0][0] *100, 2)), "%" )
print('Probability Male: ', 100 - (round(pred[0][0] *100, 2)), "%" )

if y_test[3] ==1:
    true_label = 'Female'
else:
    true_label = "Male"
print('True Label: ', true_label)
Image('gender_model_architecture.png')