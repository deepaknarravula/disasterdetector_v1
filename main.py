import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub
deepaknarravula_prj_3563_path = kagglehub.dataset_download('deepaknarravula/prj-3563')

print('Data source import complete.')


# Define the path to the original dataset (single folder with images)
input_path = '/kaggle/input/prj-3563/PRJ-3563_archive/PRJ-3563/images'

# Define new directory paths where images will be copied
output_path = '/kaggle/working/restructured_data'


# Create the necessary subdirectories for post_disaster and pre_disaster
train_dir = os.path.join(output_path, 'train')
post_disaster_dir = os.path.join(train_dir, 'post_disaster')
pre_disaster_dir = os.path.join(train_dir, 'pre_disaster')

# Create directories if they don't exist
os.makedirs(post_disaster_dir, exist_ok=True)
os.makedirs(pre_disaster_dir, exist_ok=True)

# Get all image files from the input directory
image_files = os.listdir(input_path)

# Copy images based on filenames
for image_file in image_files:
    if 'pre_disaster' in image_file:
        shutil.copy(os.path.join(input_path, image_file), os.path.join(pre_disaster_dir, image_file))
    elif 'post_disaster' in image_file:
        shutil.copy(os.path.join(input_path, image_file), os.path.join(post_disaster_dir, image_file))

print("Dataset has been restructured and saved in the new directory.")
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Set the path to the original dataset (single folder with images)
input_path = '/kaggle/input/prj-3563/PRJ-3563_archive/PRJ-3563/images'  # Adjust the path if necessary

# Get all image files from the input directory
image_files = os.listdir(input_path)

# Filter images for pre-disaster and post-disaster based on filename
pre_disaster_files = [file for file in image_files if 'pre_disaster' in file]
post_disaster_files = [file for file in image_files if 'post_disaster' in file]

# Sort the files to ensure pairs are correctly aligned
pre_disaster_files.sort()
post_disaster_files.sort()

# Take the first 4 pairs (adjust if you need more or fewer pairs)
selected_pre_disaster = pre_disaster_files[:4]
selected_post_disaster = post_disaster_files[:4]

# Create a subplot to display the comparison of images (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for i, ax in enumerate(axes.flatten()):
    # For odd columns, show post-disaster images, for even columns, show pre-disaster images
    if i % 2 == 0:  # Even index for pre-disaster
        img_path = os.path.join(input_path, selected_pre_disaster[i // 2])  # i // 2 to pair with post-disaster
        img = image.load_img(img_path, target_size=(224, 224))  # Resize images to fit in the collage
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        ax.imshow(img_array)
        ax.set_title(f"Pre Disaster {i // 2 + 1}")
    else:  # Odd index for post-disaster
        img_path = os.path.join(input_path, selected_post_disaster[i // 2])
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        ax.imshow(img_array)
        ax.set_title(f"Post Disaster {i // 2 + 1}")

    ax.axis('off')  # Turn off axes

plt.tight_layout()
plt.show()
# Initialize ImageDataGenerator with validation_split for splitting training and validation data
train_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess,  # Preprocessing for ResNet50
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% of the data for validation
)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the newly structured data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary classification: post_disaster vs. pre_disaster
    subset='training'  # Use the training subset
)

# Load the validation data
validation_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the newly structured data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation'  # Use the validation subset
)

# Check the number of classes and images
print(f"Classes: {train_generator.class_indices}")
print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {validation_generator.samples}")
# Initialize ImageDataGenerator with validation_split for splitting training and validation data
train_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess,  # Preprocessing for ResNet50
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% of the data for validation
)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the newly structured data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary classification: post_disaster vs. pre_disaster
    subset='training'  # Use the training subset
)

# Load the validation data
validation_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the newly structured data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation'  # Use the validation subset
)

# Check the number of classes and images
print(f"Classes: {train_generator.class_indices}")
print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {validation_generator.samples}")
# Plot the accuracy comparison
plt.figure(figsize=(12, 6))
plt.plot(history_resnet.history['accuracy'], label='ResNet50 Train Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='ResNet50 Validation Accuracy')
plt.plot(history_efficientnet.history['accuracy'], label='EfficientNetB0 Train Accuracy')
plt.plot(history_efficientnet.history['val_accuracy'], label='EfficientNetB0 Validation Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the loss comparison
plt.figure(figsize=(12, 6))
plt.plot(history_resnet.history['loss'], label='ResNet50 Train Loss')
plt.plot(history_resnet.history['val_loss'], label='ResNet50 Validation Loss')
plt.plot(history_efficientnet.history['loss'], label='EfficientNetB0 Train Loss')
plt.plot(history_efficientnet.history['val_loss'], label='EfficientNetB0 Validation Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate ResNet50 model
test_loss_resnet, test_accuracy_resnet = model_resnet.evaluate(validation_generator)
print(f"ResNet50 Test Accuracy: {test_accuracy_resnet * 100:.2f}%")

# Evaluate EfficientNetB0 model
test_loss_efficientnet, test_accuracy_efficientnet = model_efficientnet.evaluate(validation_generator)
print(f"EfficientNetB0 Test Accuracy: {test_accuracy_efficientnet * 100:.2f}%")

# Generate predictions for ResNet50
predictions_resnet = model_resnet.predict(validation_generator)
predicted_classes_resnet = (predictions_resnet > 0.5).astype(int)

# Generate predictions for EfficientNetB0
predictions_efficientnet = model_efficientnet.predict(validation_generator)
predicted_classes_efficientnet = (predictions_efficientnet > 0.5).astype(int)

# Confusion Matrix for ResNet50
cm_resnet = confusion_matrix(validation_generator.classes, predicted_classes_resnet)
sns.heatmap(cm_resnet, annot=True, fmt="d", xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.title('Confusion Matrix - ResNet50')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Confusion Matrix for EfficientNetB0
cm_efficientnet = confusion_matrix(validation_generator.classes, predicted_classes_efficientnet)
sns.heatmap(cm_efficientnet, annot=True, fmt="d", xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.title('Confusion Matrix - EfficientNetB0')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Classification Report for ResNet50
report_resnet = classification_report(validation_generator.classes, predicted_classes_resnet, target_names=validation_generator.class_indices.keys())
print("Classification Report - ResNet50:")
print(report_resnet)

# Classification Report for EfficientNetB0
report_efficientnet = classification_report(validation_generator.classes, predicted_classes_efficientnet, target_names=validation_generator.class_indices.keys())
print("Classification Report - EfficientNetB0:")
print(report_efficientnet)
