import math
import pickle
import random
from typing import Any

import numpy as np
import tensorflow as tf
import keras
from certifi.__main__ import args
from keras import layers
from keras.src.ops import dtype
from tensorflow.keras.models import Model
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, \
    QGridLayout, QLabel, QSpinBox
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt




class Data:
	def __init__(self,indexes,features):
		self.indexes = indexes
		self.features = features

def euclidean_distance(v1, v2):
	return np.linalg.norm(v1 - v2)

def cosine_distance(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Handle zero vectors to avoid division by zero errors

    cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2)
    # Cosine distance is 1 - cosine similarity
    cosine_dist = 1 - cosine_similarity
    return cosine_dist


class ImageRetrieval:
	def __init__(self, image_data:Data, images, query_data:Data, query_images):
		self.imageData = image_data
		self.images = images
		self.queryData = query_data
		self.queryImages = query_images

	def searchEuclidian(self,index,max_query_length):
		result = []
		for i in range(0, self.imageData.features.shape[0]):
			dist=euclidean_distance(self.imageData.features[i], self.queryData.features[index])
			result.append((dist,i))
		result = sorted(result)[:max_query_length]
		return result


	def searchCosine(self, index, max_query_length):
		result = []
		for i in range(0, self.imageData.features.shape[0]):
			dist = cosine_distance(self.imageData.features[i], self.queryData.features[index])
			result.append((dist, i))
		result = sorted(result)[:max_query_length]
		return result

class App(QDialog):

    imagesArray: list[QLabel]

    def __init__(self):

        super().__init__()
        self.title = 'Image Retrieval'
        self.left = 40
        self.top = 40
        self.width = 400
        self.height = 900



        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 3)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.imagesArray = []
        self.imagesLabels = []

        self.labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        self.initUI()

        self.model= keras.models.load_model("features.keras")
        self.model.summary()
        self.encoder = Model(inputs=self.model.inputs, outputs=self.model.get_layer("encoder").output)
        self.encoder.summary()
        self.saveFile=False
        self.loadFile=False
        if self.saveFile:
            self.saveToFile()
            print("saved")
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createGridLayout()
        self.isEuclidian=True
        windowLayout = QVBoxLayout()
        button_run=QPushButton("Run")
        button_run.clicked.connect(self.the_button_run)
        windowLayout.addWidget(button_run)
        button_random = QPushButton("Random")
        button_random.clicked.connect(self.the_button_random)
        windowLayout.addWidget(button_random)

        self.button_E = QPushButton("Euclidian")
        self.button_E.clicked.connect(self.the_button_Euclidian)
        windowLayout.addWidget(self.button_E)
        self.button_Accuracy = QPushButton("Accuracy")
        self.button_Accuracy.clicked.connect(self.calculateAccuracy)
        windowLayout.addWidget(self.button_Accuracy)

        self.numberInput = QSpinBox()
        # Or: widget = QDoubleSpinBox()

        self.numberInput.setMinimum(0)
        self.numberInput.setMaximum(9999)
        self.numberInput.setSingleStep(1)  # Or e.g. 0.5 for QDoubleSpinBox

        windowLayout.addWidget(self.numberInput)


        self.input_image=QLabel()
        self.input_image.setStyleSheet("background-color: rgb(205, 205, 205);")
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter)
        self.input_image.setFixedSize(100,100)
        self.input_image.setScaledContents(True)
        windowLayout.addWidget(self.input_image)

        self.similarity = QLabel()
        self.similarity.setStyleSheet("background-color: rgb(50, 50, 50);")
        self.similarity.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.similarity.setMaximumHeight(30)
        self.similarity.setText("Similarity:")
        self.similarity.setScaledContents(True)

        windowLayout.addWidget(self.similarity)

        windowLayout.addWidget(self.horizontalGroupBox)

        self.setLayout(windowLayout)
        self.show()

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("Grid")
        layout = QGridLayout()
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(2, 4)
        for i in range(1, 6):
            for j in range(1, 6):
                label=QLabel(str(i) + "x" + str(j))
                label.setFixedSize(100,100)
                label.setScaledContents(True)
                label.setAlignment( Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                label.setStyleSheet("background-color: rgb(20, 20, 20);")
                self.imagesArray.append(label)
                layout.addWidget(label, i, j)
                lb=QLabel(str(i) + "x" + str(j))
                lb.setFixedSize(100,40)
                lb.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
                self.imagesLabels.append(lb)
                layout.addWidget(lb, i, j)
        self.horizontalGroupBox.setLayout(layout)

    def the_button_run(self):
        print("clicked")

        # flatten 3 dim vector
        if self.loadFile:
            data = self.loadFromFile()
            features_train=data.features
        else:
            features_train = self.encoder.predict(self.x_train)



        features_test = self.encoder.predict(self.x_test)




        indexes = list(range(0, self.x_train.shape[0]))
        data = Data(indexes, features_train)
        latent_vectors = data

        search_index = self.numberInput.value()
        query_length = len(self.imagesArray)
        query_indexes = self.x_test.shape[0]
        query_features = features_test
        queryData = Data(query_indexes, query_features)
        imgR = ImageRetrieval(latent_vectors, self.x_train, queryData, self.x_test)
        if self.isEuclidian:
            result = imgR.searchEuclidian(search_index, query_length)
        else:
            result = imgR.searchCosine(search_index, query_length)
        accuracy=0
        query_type = self.y_test[search_index]
        for i in range(5):
            result_type = self.y_train[result[i][1]]
            if query_type == result_type:
                accuracy+=1


        self.similarity.setText("min distance vector:" + str(result[0][0]) +  "    accuracy:" + str(accuracy) + "/5")

        for i in range(0,query_length):
            self.imagesArray[i].setPixmap(self.getPixmap(self.x_train[result[i][1]].reshape(32,32,3)))
            self.imagesLabels[i].setText(str(result[i][0]) + "\n" + self.labels[self.y_train[result[i][1]][0]])

        self.input_image.setPixmap(self.getPixmap(self.x_test[search_index].reshape(32,32,3)))


    def the_button_random(self):
        self.numberInput.setValue(random.randint(0, 9999))
        self.numberInput.update()
    def the_button_Euclidian(self):
        if self.isEuclidian:
            self.isEuclidian = False
            self.button_E.setText("Cosine")
        else:
            self.isEuclidian = True
            self.button_E.setText("Euclidian")


    def calculateAccuracy(self):
        if self.loadFile:
            data = self.loadFromFile()
            features_train = data.features
        else:
            features_train = self.encoder.predict(self.x_train)

        features_test = self.encoder.predict(self.x_test)


        indexes = list(range(0, self.x_train.shape[0]))
        data = Data(indexes, features_train)
        latent_vectors = data

        len=5000#self.y_train.shape[0]
        accuracy_in1=0.0
        accuracy_in5=0.0
        accuracy_in5_total=0.0
        query_indexes = self.x_test.shape[0]
        query_features = features_test
        queryData = Data(query_indexes, query_features)
        img_r = ImageRetrieval(latent_vectors, self.x_train, queryData, self.x_test)
        for i in range(len):
            search_index = i
            print(str(i))
            query_length = 5
            if self.isEuclidian:
                result = img_r.searchEuclidian(search_index, query_length)
            else:
                result = img_r.searchCosine(search_index, query_length)
            accuracy = 0
            query_type = self.y_test[search_index]
            for i in range(5):
                result_type = self.y_train[result[i][1]]
                if query_type == result_type:
                    accuracy += 1
            accuracy_in5+=accuracy/5
            if self.y_test[search_index][0] == self.y_train[result[0][1]][0]:
                accuracy_in1 += 1
            if accuracy > 0:
                accuracy_in5_total += 1
        accuracy_in1/=len
        accuracy_in5/=len
        accuracy_in5_total/=len

        print("accuracy_1:" + str(accuracy_in1)+"  "+ "accuracy_5:" + str(accuracy_in5) + "precision:" + str(accuracy_in5_total))



    def getPixmap(self,im):
        im = im * 255
        img_data = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                for k in range(3):
                    img_data[i][j][k] = im[i][j][k]
        return QPixmap(QImage(img_data, 32, 32, 32 * 3, QImage.Format.Format_RGB888))
    def saveToFile(self):
        indexes = list(range(0, self.x_train.shape[0]))
        vectors = self.encoder.predict(self.x_train)
        data = Data(indexes,vectors)
        # write the data dictionary to disk
        f = open('latent_vector_list', "wb")
        f.write(pickle.dumps(data))
        f.close()
    def loadFromFile(self):
        with open('latent_vector_list', 'rb') as file:
            data = pickle.load(file)
        return data

app = QApplication(sys.argv)
window = App()
window.show()
app.exec()
#9359
#1634
#2394
#1108
#6585 color red
#9828
#5595
#7450 cars
#7034 planes
#9963 ships
#2148

#4723
#5163
#5243 car
#2351
#2636
#7610 birds
#2544
#2679