import sys
import numpy as np
import keras
from PIL import Image
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap, QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


#main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(400, 300))
        self.setWindowTitle("CNN MNIST")
        ##########################################################
        #LOAD different models here
        self.model = keras.models.load_model("MNISTcnnv2.keras")
        ###########################################################
        layout = QVBoxLayout()


        button = QPushButton("Run")
        button.setCheckable(False)
        button.setFixedSize(100,40)
        button.clicked.connect(self.the_button_was_clicked)
        layout.addWidget(button)


        button_reset = QPushButton("Reset")
        button_reset.setCheckable(False)
        button_reset.setFixedSize(100, 40)
        button_reset.clicked.connect(self.the_button_reset)
        layout.addWidget(button_reset)


        self.label_number = QLabel("draw a number")

        layout.addWidget(self.label_number)




        #layout.addWidget(self.labelPixmap)
        self.canvas=Canvas()
        layout.addWidget(self.canvas)
        widget = QWidget()
        widget.setLayout(layout)
        # Set the central widget of the Window.
        self.setCentralWidget(widget)

    def the_button_was_clicked(self):
        print("clicked")
        image= self.canvas.myPixmap.toImage()
        b = image.bits()
        # sip.voidptr must know size to support python buffer interface
        b.setsize(100*100*4)
        arr = np.frombuffer(b, np.uint8).reshape((100, 100, 4))
        image = Image.fromarray(arr)
        image = image.convert("L")
        image = image.resize((28, 28))

        image = image.convert("L")
        image_array = np.array(image)
        print(image_array.shape)
        model =self.model
        input_for_predict = np.expand_dims(image_array, axis=0)
        prediction = model.predict(input_for_predict)
        print(f"Shape of input passed to predict: {input_for_predict.shape}")
        print(f"Prediction output: {prediction}")
        # get the maximum probability
        max_index = np.argmax(prediction[0])
        print(max_index)
        self.label_number.setText("Number is "+str(max_index))

    def the_button_reset(self):
        self.canvas.reset()
        self.label_number.setText("draw a number")


class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("black"))
        self.setAutoFillBackground(True)
        self.setPalette(p)
        self.myPixmap = QPixmap(100,100)
        self.setFixedSize(100,100)
        self.painter = QPainter(self.myPixmap)
        self.pen = QPen(QColor("white"))
        self.pen.setWidth(5)
        self.painter.setPen(self.pen)
        self.painter.fillRect(0,0,100,100, QColor("black"))
        self.setPixmap(self.myPixmap)
        self.last = None
    def reset(self):
        self.painter.fillRect(0, 0, 100, 100, QColor("black"))
        self.setPixmap(self.myPixmap)
    def mouseMoveEvent(self, event):
        if self.last:
            self.painter.drawLine(self.last, event.pos())

            self.last = event.pos()
            self.setPixmap(self.myPixmap)
            self.update()

    def mousePressEvent(self, event):
        self.last = event.pos()

    def mouseReleaseEvent(self, event):
        self.last = None

    def updateSize(self, width, height):
        pm = QPixmap(width, height)
        pm.fill(QColor("black"))
        old = self.myPixmap
        self.myPixmap = pm
        self.pen = QPen(QColor("white"))
        self.painter = QPainter(pm)
        self.painter.drawPixmap(0,0,old)
        self.setPixmap(pm)

    def resizeEvent(self, event):
        if event.oldSize().width() > 0:
            self.updateSize(event.size().width(), event.size().height())
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()



