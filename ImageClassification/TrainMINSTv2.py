import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mnist_cnn():
    model = keras.Sequential([
        #Convolution
        layers.Input(shape=(28, 28, 1)),


        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),


        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def train_mnist():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize 0-1 range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Data augmentation for improving accuracy
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    model = build_mnist_cnn()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]

    # Train
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1 #logging output
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, history


model,history=train_mnist()
model.save("MNISTcnnv2.keras")