import os
import random
import cv2
import numpy as np
import tensorflow as tf
import keras
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QSpinBox, QLabel
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QThread, QObject, pyqtSignal
import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

NUM_CLASSES = 10
NFRAMES=10 # transfer 10 cnn 10 ltsm 16 ltsmv2 16

def getClassNames(DatasetName):
    folder_path = DatasetName

    subfolders = [name for name in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, name))]

    return subfolders

DATASET="Dataset1"
CLASS_NAMES=getClassNames("../Dataset1/train/")
CLASS_NAMES.sort()
print(CLASS_NAMES)

class Worker(QObject):
    update_text = pyqtSignal(str)
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.model = keras.models.load_model("transfer.keras")
        self.videoPaths=glob.glob("../"+DATASET+"/**/*.avi",recursive=True)



    def stop(self):
        self.finished.emit()
        self.running = False

    def run(self):
        correct, false = 0, 0
        random.seed(27788)
        random.shuffle(self.videoPaths)
        y_true = []
        y_pred = []
        for i in range(self.videoPaths.__len__()):
            if not self.running:
                break
            videoPath = self.videoPaths[i]
            classname =self.videoPaths[i].split("/")[3]
            frames = frames_from_video_file(self.videoPaths[i], NFRAMES)
            frames = np.expand_dims(frames, 0)
            prediction = self.model(frames)

            label = np.argmax(prediction[0])
            y_pred.append(label)
            y_true.append(CLASS_NAMES.index(classname))
            if classname == CLASS_NAMES[label]:
                correct += 1
            else:
                false += 1
            print(self.videoPaths[i],i,label)
            accuracy = correct * 100 / (i + 1)
            percent=i/self.videoPaths.__len__()*100
            percentstr=f"{percent:.1f}"
            self.update_text.emit("%"+percentstr  +"    "+str(i)+ "    " +classname+"    "+CLASS_NAMES[label] +"    accuracy: "+str(accuracy))
            #print(i, classname, CLASS_NAMES[label], "correct: ", correct, "false: ", false, "accuracy: ", accuracy)

        accuracy = correct * 100 / self.videoPaths.__len__()
        self.compute_all_metrics(y_true, y_pred)
        #self.update_text.emit("Metrics")
        self.finished.emit()

    def compute_all_metrics(self,y_true, y_pred):
        # Convert to numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(classification_report(y_true, y_pred))


class VideoApp(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 1024, 768)

        self.videoPaths=glob.glob("../"+DATASET+"/**/*.avi",recursive=True)
        random.shuffle(self.videoPaths)
        print(self.videoPaths)
        self.media_player = QMediaPlayer()
        self.media_player.setSource(QUrl.fromLocalFile(self.videoPaths[0]))

        self.model = keras.models.load_model("transfer.keras")
        self.model.summary()

        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)

        self.numberInput = QSpinBox()
        self.numberInput.setMinimum(0)
        self.numberInput.setMaximum(self.videoPaths.__len__())
        self.numberInput.setSingleStep(1)  # Or e.g. 0.5 for QDoubleSpinBox

        self.label_text = QLabel("result")
        self.label_text.setMaximumHeight(20)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)

        self.metrics_button = QPushButton("Metrics")
        self.metrics_button.clicked.connect(self.toggle_thread)
        self.metrics_button.setStyleSheet("text-align: left; padding-left: 10px;")

        self.thread = None
        self.worker = None
        self.running = False

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_model)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.metrics_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.numberInput)
        layout.addWidget(self.label_text)
        layout.addWidget(self.slider)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_video(self):
        self.media_player.setSource(QUrl.fromLocalFile(self.videoPaths[self.numberInput.value()]))
        self.media_player.play()

    def compute_all_metrics(self,y_true, y_pred, num_classes=NUM_CLASSES):
        # Convert to numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(classification_report(y_true, y_pred))


    def stop_thread(self):
        if self.worker:
            self.worker.stop()  # signal loop to exit

        if self.thread:
            self.thread.quit()
            self.thread.wait()

        self.running = False
        self.metrics_button.setText("Start")

    def toggle_thread(self):
        if not self.running:
            self.calculate_metrics_thread()
        else:
            self.stop_thread()

    def calculate_metrics_thread(self):


        self.thread = QThread()
        self.worker = Worker()

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)

        # cleanup (VERY IMPORTANT in PyQt6)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.update_text.connect(self.update_label)

        self.thread.start()
        self.running = True
        self.metrics_button.setText("Run")

    def update_label(self, text):
        self.metrics_button.setText(text)
    def calculate_metrics(self):
        correct, false = 0, 0
        random.seed(27788)
        random.shuffle(self.videoPaths)
        y_true = []
        y_pred = []
        for i in range(self.videoPaths.__len__()):
            classname = self.videoPaths[i].split("/")[3]
            print(classname)
            frames = frames_from_video_file(self.videoPaths[i], NFRAMES)
            frames = np.expand_dims(frames, 0)
            prediction = self.model(frames)

            label = np.argmax(prediction[0])
            y_pred.append(label)
            y_true.append(CLASS_NAMES.index(classname))
            if classname == CLASS_NAMES[label]:
                correct += 1
            else:
                false += 1
            # print(self.videoPaths[i],i,label)
            accuracy = correct * 100 / (i + 1)
            print(i, classname, CLASS_NAMES[label], "correct: ", correct, "false: ", false, "accuracy: ", accuracy)

        accuracy = correct * 100 / self.videoPaths.__len__()
        self.compute_all_metrics(y_true, y_pred)

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()

    def run_model(self):
        self.media_player.pause()
        print("run model")
        print(self.videoPaths[self.numberInput.value()])
        frames=frames_from_video_file(self.videoPaths[self.numberInput.value()],NFRAMES)
        frames=np.expand_dims(frames,0)
        prediction=self.model(frames)

        label=np.argmax(prediction[0])
        self.label_text.setText(str(label)+"  "+CLASS_NAMES[label])
        print(CLASS_NAMES[label])

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (112,112), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = int(video_length - need_length)
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_player = VideoPlayer()
    video_player.show()
    sys.exit(app.exec())

