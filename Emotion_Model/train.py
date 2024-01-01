import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

# Initialize image data generators with rescaling and data augmentation
train_data_gen = ImageDataGenerator(rescale=1./255)

validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess training and validation data
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change color_mode to grayscale
    class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change color_mode to grayscale
    class_mode='categorical')

# Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(48,48,3))

# Add a global spatial average pooling layer to the model
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with L2 regularization and ReLU activation
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.25)(x)

# Add another fully-connected layer with L2 regularization and softmax activation for the output
predictions = Dense(7, activation='softmax', kernel_regularizer=l2(0.001))(x)

# Define the model for training
emotion_model = Model(inputs=base_model.input, outputs=predictions)

# Display the summary, including total and trainable parameters
emotion_model.summary()

# Compile the model with a lower learning rate
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=7e-7,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
    metrics=['accuracy'])

# Implement Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
model_checkpoint = ModelCheckpoint('model/best_model.h5',monitor="val_accuracy", save_best_only=True, verbose=1)

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=80,  # Increase the number of epochs
    validation_data=validation_generator,
    validation_steps=7188 // 64,
    callbacks=[early_stopping, model_checkpoint])

# Save the model structure in a JSON file
emotion_model.save('model/my_model.h5')

# Assuming you have 'val_loss' and 'val_accuracy' in your history
plt.figure(figsize=[12, 6])

# Plotting Loss
plt.subplot(1, 2, 1)
plt.plot(emotion_model_info.history['loss'], 'blue', linewidth=2.0, label='Training Loss')
plt.plot(emotion_model_info.history['val_loss'], 'orange', linewidth=2.0, label='Validation Loss')
plt.legend(fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Training and Validation Loss Curves', fontsize=10)

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(emotion_model_info.history['accuracy'], 'blue', linewidth=2.0, label='Training Accuracy')
plt.plot(emotion_model_info.history['val_accuracy'], 'orange', linewidth=2.0, label='Validation Accuracy')
plt.legend(fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Training and Validation Accuracy Curves', fontsize=10)

plt.tight_layout()
plt.show()

