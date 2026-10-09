import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_baseline_cnn_improved(num_classes=6, input_shape=(150, 150, 3), learning_rate=1e-3):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1 (Unregularized - early edge features)
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d"),
            layers.MaxPooling2D((2, 2), name="max_pooling2d"),
            
            # Block 2 (Unregularized - mid-level patterns)
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_1"),
            layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
            
            # Block 3 (High-level spatial features)
            layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_2"),
            layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
            # NEW: Drop 20% of spatial channels to prevent deep-block overfitting to weather
            layers.SpatialDropout2D(0.2, name="spatial_dropout2d"),
            
            # Classification Head
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense"),
            # NEW: Drop 40% of dense neurons to prevent memorization
            layers.Dropout(0.4, name="dropout"),
            layers.Dense(num_classes, activation="softmax", dtype="float32", name="dense_1"),
        ],
        name="baseline_cnn_improved"
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
