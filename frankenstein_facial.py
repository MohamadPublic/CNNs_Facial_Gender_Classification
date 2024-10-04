import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

source_folder = 'C://Users//Moe//Desktop//School//W2024//Machine Learning//Final Project//_PROPER//img_full_face'

dim = 32 #Store dimension of images 
random.seed(314) #Set Seed
tf.random.set_seed(1618) #Set Seed

# Function to remove numbers from a string
def remove_numbers(s):
    return ''.join([i for i in s if not i.isdigit()])

data = []
for filename in os.listdir(source_folder):
    label = remove_numbers(filename.split('_')[1]).lower()
    if label == 'butler' or label == 'radcliffe' or label == 'vartan' or label == 'bracco' or label == 'gilpin' or label == 'harmon':
        img = Image.open(os.path.join(source_folder, filename)).convert('L')
        img = img.resize((dim, dim))

        # Convert the image to a numpy array and append to the training set with the label
        img_array = np.array(img)
        data.append((img_array, label))

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

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform([label for img, label in train_data])
y_test = le.transform([label for img, label in test_data])

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Classes:' , le.classes_)

for i in range (58,60):
    plt.imshow(X_test[i].reshape(dim, dim), cmap="gray")
    print(y_test[i])
    plt.title(f"Label: {le.inverse_transform([np.argmax(y_test[i])])[0]}")
    #plt.show()
    

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(3, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(6, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', #perhaps try sgd
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Fit the model on the training data
history = model.fit(X_train, y_train, epochs=128)

# Extract the feature vectors
feature_model = models.Model(inputs=model.input, outputs=model.layers[-5].output)  # Access the Flatten layer
X_train_features = feature_model.predict(X_train)
X_test_features = feature_model.predict(X_test)
from sklearn.model_selection import GridSearchCV

# Define a KNN model
knn = KNeighborsClassifier()

# Set up GridSearchCV to find the best K
param_grid = {'n_neighbors': range(1, 50)}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_features, np.argmax(y_train, axis=1))

# Best K value found
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_
print(f"Best K: {best_k} with Cross-Validated Accuracy: {best_score}")

# Plotting the result
test_scores = [grid_search.cv_results_['mean_test_score'][k-1] for k in range(1, 50)]
plt.figure(figsize=(10, 5))
plt.plot(range(1, 50), test_scores, marker='o', linestyle='-', color='b')
plt.title('KNN Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.grid(True)
plt.xticks(range(1, 50))  # Ensure all k values are marked
plt.show()

# Evaluate the model on the test set using the best K
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_features, np.argmax(y_train, axis=1))
final_accuracy = final_knn.score(X_test_features, np.argmax(y_test, axis=1))
print(f"Final Test Set Accuracy with K={best_k}: {final_accuracy}")
