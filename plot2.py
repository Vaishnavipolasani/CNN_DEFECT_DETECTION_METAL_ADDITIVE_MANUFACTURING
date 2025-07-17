import glob
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(label_type):
    path = f'./{label_type.lower()}/*.tif'
    labels = []
    images = []
    for file in glob.glob(path):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            if label_type.lower() == "lof":
                labels.append(0)
            elif label_type.lower() == "gas":
                labels.append(1)
    return np.array(images), np.array(labels)


lof_images, lof_labels = load_data('lof')
gas_images, gas_labels = load_data('gas')

if len(lof_images) == 0 or len(gas_images) == 0:
    print("No images were loaded. Please check the file paths and ensure there are images in the specified directories.")
    exit()

images = np.concatenate([lof_images, gas_images])
labels = np.concatenate([lof_labels, gas_labels])

# Normalize the images
images = images.astype('float32') / 255.0

# Convert the images back to 8-bit unsigned integers
images = (images * 255).astype('uint8')

# Preprocessing: Apply a filter to enhance the edges
images = np.array([cv2.Laplacian(img, cv2.CV_8U) for img in images])

# Resize images to a fixed size (e.g., 128x128) if necessary
images_resized = np.array([cv2.resize(img, (128, 128)) for img in images])

# Reshape images to include the channel dimension
images_resized = images_resized.reshape((images_resized.shape[0], 128, 128, 1))


# Convert labels to one-hot encoding
labels_categorical = to_categorical(labels, 2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images_resized, labels_categorical, test_size=0.2, random_state=42)

# Split the training data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with data augmentation
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=10,
          validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy}')

print(y_test_classes)
print(y_pred_classes)
