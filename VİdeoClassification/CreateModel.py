import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import GlobalAveragePooling3D, TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import keras


NFRAMES = 20
NUM_CLASSES = 12
IMG_SIZE = 112
EPOCH=50
LEARNİNG_RATE=1e-4
Dataset="Dataset1"
#check gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')

#configure gpu
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("set gpu memory growth")



class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=8, frames=NFRAMES, size=(IMG_SIZE , IMG_SIZE ), shuffle=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.frames = frames
        self.size = size
        self.shuffle = shuffle
        self.num_classes=NUM_CLASSES
        self.class_names = sorted(os.listdir(root_dir))
        self.class_map = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        for cls in self.class_names:
            cls_path = os.path.join(root_dir, cls)
            for vid in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, vid), self.class_map[cls]))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("Cannot open:", path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        step = max(1, total_frames // self.frames)
        count = 0

        while len(frames) < self.frames:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                frame = cv2.resize(frame, self.size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0

                frames.append(frame)

            count += 1

        cap.release()

        if len(frames) == 0:
            print("Empty video:", path)
            return None

        while len(frames) < self.frames:
            frames.append(frames[-1])

        return np.array(frames)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []

        for path, label in batch:
            video = self.load_video(path)

            if video is None:
                video = np.zeros((self.frames, self.size[0], self.size[1], 3), dtype=np.float32)


            if video.shape != (self.frames, self.size[0], self.size[1], 3):
                video = np.zeros((self.frames, self.size[0], self.size[1], 3), dtype=np.float32)

            X.append(video)
            y.append(label)


        X = np.stack(X, axis=0).astype(np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        return X, y

train_data = VideoDataset("../"+ Dataset+"/train", batch_size=8)
val_data = VideoDataset("../"+ Dataset+"/val", batch_size=8, shuffle=False)
print(len(train_data)*8)
print(len(val_data)*8)

print(train_data)

def build_3d_cnn(input_shape=(NFRAMES,IMG_SIZE,IMG_SIZE,3), num_classes=NUM_CLASSES):
    model = keras.models.Sequential()

    # 3D convolutional layer with 64 filters, kernel size of (3, 3, 3), and ReLU activation
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    # 3D max pooling layer with pool size of (2, 2, 2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Batch normalization layer
    model.add(BatchNormalization())

    # Another 3D convolutional layer with 128 filters, kernel size of (3, 3, 3), and ReLU activation
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    # Another 3D max pooling layer with pool size of (2, 2, 2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Another batch normalization layer
    model.add(BatchNormalization())

    # Another 3D convolutional layer with 256 filters, kernel size of (3, 3, 3), and ReLU activation
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    # Another 3D max pooling layer with pool size of (2, 2, 2)
    model.add(MaxPooling3D((2, 2, 2)))
    # Another batch normalization layer
    model.add(BatchNormalization())

    # Flatten layer to flatten the output of the convolutional layers
    model.add(GlobalAveragePooling3D())#raises accuracy
    model.add(Dense(128, activation='relu')) #raises accuracy
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNİNG_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
def build_3d_cnnV2(input_shape=(NFRAMES, IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    x = inputs

    # Block 1 (NO temporal pooling)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1,2,2))(x)  # only spatial

    # Block 2
    x = Conv3D(128, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1,2,2))(x)

    # Block 3
    x = Conv3D(256, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2,2,2))(x)  # NOW reduce time

    # Block 4 (new - deeper)
    x = Conv3D(256, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2,2,2))(x)

    # Head
    x = GlobalAveragePooling3D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNİNG_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def build_3d_cnn_transfer_learning(input_shape=(NFRAMES,IMG_SIZE,IMG_SIZE,3), num_classes=NUM_CLASSES):
    """
    input_shape: (num_frames, height, width, channels), e.g., (10,112,112,3)
    num_classes: total number of action classes
    """
    num_frames, h, w, c = input_shape

    # Pretrained MobileNetV2 backbone (feature extractor)
    base_cnn = MobileNetV2(
        weights='imagenet', include_top=False,
        input_shape=(h, w, c)
    )
    base_cnn.trainable = False  # freeze CNN

    # Input layer: sequence of frames
    video_input = layers.Input(shape=input_shape)

    # TimeDistributed: apply CNN to each frame
    x = layers.TimeDistributed(base_cnn)(video_input)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # frame-level feature vector

    # LSTM for temporal modeling
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=video_input, outputs=output)

    # Compile model
    model.compile(
        #1e-4
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNİNG_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
def build_lstm_model(input_shape=(NFRAMES, IMG_SIZE,IMG_SIZE, 3), num_classes=NUM_CLASSES):

    inputs = keras.Input(shape=input_shape)

    # --- Frame feature extractor ---
    x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)

    x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)

    x = TimeDistributed(Conv2D(128, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)

    x = TimeDistributed(Conv2D(256, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # --- Temporal learning ---
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    x = LSTM(128)(x)
    x = Dropout(0.5)(x)

    # --- Classifier ---
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def build_lstmV2(input_shape=(16,IMG_SIZE,IMG_SIZE, 3), num_classes=NUM_CLASSES):

    inputs = keras.Input(shape=input_shape)

    # -------- Frame feature extractor --------
    x = TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(2,2))(x)

    x = TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(2,2))(x)

    x = TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(2,2))(x)

    x = TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # -------- Temporal modeling --------
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.5)(x)

    # -------- Classifier --------
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def build_lstmV3(input_shape=(16,IMG_SIZE,IMG_SIZE, 3), num_classes=NUM_CLASSES):

    inputs = keras.Input(shape=input_shape)

    x = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D(2))(x)

    x = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)

    x = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same'))(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def build_3dcnn_lstm_v2(input_shape=(16,112,112,3), num_classes=NUM_CLASSES):

    inputs = keras.Input(shape=input_shape)

    x = inputs

    # -------- 3D CNN (feature extractor) --------
    # Block 1 (keep time)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1,2,2))(x)

    # Block 2
    x = Conv3D(128, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1,2,2))(x)

    # Block 3 (NOW reduce time)
    x = Conv3D(256, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2,2,2))(x)

    # -------- Convert to sequence --------
    # (batch, time, H, W, C) → (batch, time, features)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # -------- LSTM --------
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    # -------- Classifier --------
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def build_transfer_lstm(input_shape=(20, 112, 112, 3), num_classes=12):

    inputs = keras.Input(shape=input_shape)

    # -------- Pretrained CNN --------
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(112, 112, 3)
    )

    base_model.trainable = False  # freeze initially

    # Apply CNN to each frame
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # -------- Temporal --------
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = LSTM(128)(x)
    x = Dropout(0.3)(x)

    # -------- Classifier --------
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
print(len(train_data.samples))
model = build_3d_cnn()
# X, y = train_data[0]
#
# print("X:", X.shape, X.min(), X.max())
# print("y:", y.shape)
# print("label distribution:", y.sum(axis=0))

# print(train_data.class_names)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1
    )
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCH,
    callbacks=callbacks,

)

model.save("transfer12class.keras")