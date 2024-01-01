import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Function to plot a normalized confusion matrix
def plot_normalized_confusion_matrix(conf_matrix, class_names, save_path):
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.show()

# Load the trained model
loaded_model = load_model('model/my_model.h5')

# Initialize image data generator for the test set
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess test data
test_generator = test_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change color_mode to grayscale
    class_mode='categorical',
    shuffle=False  # Do not shuffle for confusion matrix calculation
)

# Generate predictions
predictions = loaded_model.predict(test_generator)

# Get true labels
true_labels = test_generator.classes

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get class names from the data generator
class_names = list(test_generator.class_indices.keys())

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot normalized confusion matrix using seaborn
plot_normalized_confusion_matrix(conf_matrix, class_names, 'confusion_matrix_normalized.png')

# Print classification report
classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names)

# Save classification report as text file
with open('classification_report.txt', 'w') as f:
    f.write(classification_rep)

print("Classification Report:\n", classification_rep)

