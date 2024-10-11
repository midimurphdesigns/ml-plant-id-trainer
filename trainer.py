import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# Set parameters
IMAGE_SIZE = (224, 224)  # Resize all images to this size
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = len(
    os.listdir("dataset/train")
)  # Number of subfolders in train directory

# Define the directories
TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
MODEL_SAVE_PATH = "model/plant_species_classifier.h5"


# Load and preprocess data
def load_data():
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",  # Integer encoding for labels
        shuffle=True,
        seed=42,
    )

    validation_dataset = image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    return train_dataset, validation_dataset


# Build the model (Simple CNN)
def build_model():
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(224, 224, 3)),
            layers.Rescaling(1.0 / 255),  # Normalize pixel values between 0 and 1
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),  # Output layer
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# Train and save the model
def train_model():
    train_dataset, validation_dataset = load_data()

    model = build_model()

    history = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=EPOCHS
    )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
