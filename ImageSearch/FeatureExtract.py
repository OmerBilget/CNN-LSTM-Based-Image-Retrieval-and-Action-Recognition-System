import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feature_extractor():
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Feature vector (embedding)
    features = layers.Dense(128, activation='relu', name="encoder")(x)

    # Classification head (only used for training)
    outputs = layers.Dense(10, activation='softmax')(features)

    model = keras.Model(inputs, outputs)
    return model

def build_feature_extractor_v2():
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Better embedding
    features = layers.Dense(256, activation='relu', name="encoder")(x)

    outputs = layers.Dense(10, activation='softmax')(features)

    return models.Model(inputs, outputs)
def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_feature_extractor_v2()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1
    )

    return model

model=train_model()
model.save("features.keras")
