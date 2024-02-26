import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF
import torchvision
import torch
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F


class DrawingScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(Qt.black)  # set background color
        self.setSceneRect(0, 0, 300, 300)  # set fixed size
        self.pen = QPen(Qt.white, 5, Qt.SolidLine)  # set pen color and width
        self.drawing = False
        self.last_point = QPointF()  # the last point of the mouse

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.scenePos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            self.addLine(self.last_point.x(), self.last_point.y(), event.scenePos().x(), event.scenePos().y(), self.pen)
            self.last_point = event.scenePos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(300, 300, 1000, 600)  # set the position and size of the window
        self.setWindowTitle("CV DL HW2")

        # initialize parameters
        self.params()

        # set the layout
        main_layout = QHBoxLayout(self)

        # Load Image
        load_image_button = QPushButton("Load Image")
        load_image_button.clicked.connect(self.load_image)
        main_layout.addWidget(load_image_button)

        # Block 1: VBoxLayout to contain 3 widgets
        block1 = QVBoxLayout()
        main_layout.addLayout(block1)

        # Hough Circle Transform
        HoughCircleTransform = QGroupBox("1. Hough Circle Transform")
        HoughCircleTransform.setLayout(QVBoxLayout())
        draw_contour_button = QPushButton("1.1 Draw Contour")
        draw_contour_button.clicked.connect(self.draw_contour)
        HoughCircleTransform.layout().addWidget(draw_contour_button)
        count_coins_button = QPushButton("1.2 Count Coins")
        count_coins_button.clicked.connect(self.count_coins)
        HoughCircleTransform.layout().addWidget(count_coins_button)
        self.coins_text = QLabel("There are _ coins in the image. ")
        HoughCircleTransform.layout().addWidget(self.coins_text)
        block1.addWidget(HoughCircleTransform)

        # Histogram Equalization
        HistogramEqualization = QGroupBox("2. Histogram Equalization")
        HistogramEqualization.setLayout(QVBoxLayout())
        histogram_equalization_button = QPushButton("2. Histogram Equalization")
        histogram_equalization_button.clicked.connect(self.histogram_equalization)
        HistogramEqualization.layout().addWidget(histogram_equalization_button)
        block1.addWidget(HistogramEqualization)

        # Morphology Operation
        MorphologyOperation = QGroupBox("3. Morphology Operation")
        MorphologyOperation.setLayout(QVBoxLayout())
        closing_button = QPushButton("3.1 Closing")
        closing_button.clicked.connect(self.closing)
        MorphologyOperation.layout().addWidget(closing_button)
        opening_button = QPushButton("3.2 Opening")
        opening_button.clicked.connect(self.opening)
        MorphologyOperation.layout().addWidget(opening_button)
        block1.addWidget(MorphologyOperation)

        # Block 2: VBoxLayout to contain 2 widgets
        block2 = QVBoxLayout()
        main_layout.addLayout(block2)

        # MNIST Classifier Using VGG19
        MNISTClassifierUsingVGG19 = QGroupBox("MNIST Classifier Using VGG19")
        MNISTClassifierUsingVGG19.setLayout(QHBoxLayout())
        # left layout
        MNISTClassifierUsingVGG19_layout1 = QVBoxLayout()
        show_model_structure_vgg19_button = QPushButton("1. Show Model Structure")
        show_model_structure_vgg19_button.clicked.connect(self.show_model_structure_vgg19)
        MNISTClassifierUsingVGG19_layout1.addWidget(show_model_structure_vgg19_button)
        show_accuracy_and_loss_button = QPushButton("2. Show Accuracy and Loss")
        show_accuracy_and_loss_button.clicked.connect(self.show_accuracy_and_loss)
        MNISTClassifierUsingVGG19_layout1.addWidget(show_accuracy_and_loss_button)
        predict_button = QPushButton("3. Predict")
        predict_button.clicked.connect(self.predict)
        MNISTClassifierUsingVGG19_layout1.addWidget(predict_button)
        reset_button = QPushButton("4. Reset")
        reset_button.clicked.connect(self.reset)
        MNISTClassifierUsingVGG19_layout1.addWidget(reset_button)
        self.mnist_predicition_text = QLabel("Prediction:")
        MNISTClassifierUsingVGG19_layout1.addWidget(self.mnist_predicition_text)
        # right layout
        MNISTClassifierUsingVGG19_layout2 = QVBoxLayout()
        ## Graffiti board (drawing area)
        self.mnist_graphic_view = QGraphicsView()
        self.mnist_graphic_view.setFixedSize(320, 320)
        self.mnist_graphic_view.setScene(DrawingScene())
        MNISTClassifierUsingVGG19_layout2.addWidget(self.mnist_graphic_view)
        ## Prediction result
        # set the layout of MNISTClassifierUsingVGG19
        MNISTClassifierUsingVGG19.layout().addLayout(MNISTClassifierUsingVGG19_layout1)
        MNISTClassifierUsingVGG19.layout().addLayout(MNISTClassifierUsingVGG19_layout2)
        block2.addWidget(MNISTClassifierUsingVGG19)

        # ResNet 50
        ResNet50 = QGroupBox("ResNet 50")
        ResNet50.setLayout(QHBoxLayout())
        # left layout
        ResNet50_layout1 = QVBoxLayout()
        load_images_button = QPushButton("Load Image")
        load_images_button.clicked.connect(self.load_images)
        ResNet50_layout1.addWidget(load_images_button)
        show_images_button = QPushButton("5.1. Show Images")
        show_images_button.clicked.connect(self.show_images)
        ResNet50_layout1.addWidget(show_images_button)
        show_model_structure_resnet50_button = QPushButton("5.2. Show Model Structure")
        show_model_structure_resnet50_button.clicked.connect(self.show_model_structure_resnet50)
        ResNet50_layout1.addWidget(show_model_structure_resnet50_button)
        show_comparison_button = QPushButton("5.3. Show Comparison")
        show_comparison_button.clicked.connect(self.show_comparison)
        ResNet50_layout1.addWidget(show_comparison_button)
        inference_button = QPushButton("5.4. Inference")
        inference_button.clicked.connect(self.inference)
        ResNet50_layout1.addWidget(inference_button)
        # right layout
        ResNet50_layout2 = QVBoxLayout()
        self.cat_dog_image_view = QGraphicsView()
        self.cat_dog_image_view.setScene(QGraphicsScene())
        self.cat_dog_image_view.setFixedSize(224, 224)
        self.cat_dog_predicition_text = QLabel("Prediction:")
        ResNet50_layout2.layout().addWidget(self.cat_dog_image_view)
        ResNet50_layout2.layout().addWidget(self.cat_dog_predicition_text)
        # set the layout of ResNet50
        ResNet50.layout().addLayout(ResNet50_layout1)
        ResNet50.layout().addLayout(ResNet50_layout2)
        block2.addWidget(ResNet50)

    def params(self):
        self.image = None
        # Q1
        self.coins = None
        # Q2
        # no parameters
        # Q3
        # no parameters
        # Q4
        self.mnist_model = torchvision.models.vgg19_bn(num_classes=10)
        self.mnist_model.load_state_dict(torch.load("./model/mnist.pth", map_location=torch.device("cpu")))
        # Q5
        self.cat_dog_model = torchvision.models.resnet50(num_classes=2)
        self.cat_dog_model.load_state_dict(torch.load("./model/cat_dog_resnet50_re.pth", map_location=torch.device("cpu")))

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", os.getcwd(), "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.image = image_path
        else:
            QMessageBox.warning(self, "Warning", "Please select an image file.")

    def draw_contour(self):
        if self.image is None:
            return

        # read the image
        original_image = cv2.imread(self.image)
        # convert to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # apply Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=20, maxRadius=40)
        print(circles.shape)
        # draw the detected circles as contours on the original image
        if circles is not None:
            self.coins = circles.shape[1]
            circles = np.uint16(np.around(circles))

            # draw the detected circles as contours
            process_image = original_image.copy()
            for x, y, r in circles[0, :]:  # x, y, r are the coordinates of the center and the radius of the circle
                cv2.circle(process_image, (x, y), r, (0, 255, 0), 2)

            # draw the center of the circle
            mask_image = np.zeros(process_image.shape, dtype=np.uint8)  # create a mask image with the same shape as the original image
            for x, y, r in circles[0, :]:
                cv2.circle(mask_image, (x, y), 1, (255, 255, 255), 2)

            # use matplot to display the original image, processed image and the mask image
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("img_src")
            plt.subplot(1, 3, 2)
            plt.imshow(process_image)
            plt.title("img_process")
            plt.subplot(1, 3, 3)
            plt.imshow(mask_image)
            plt.title("Circle_center")
            plt.show()

    def count_coins(self):
        if self.image is not None:
            self.coins_text.setText("There are {} coins in the image. ".format(self.coins))
        else:
            return

    def histogram_equalization(self):
        if self.image is None:
            return

        # the given image is default in grayscale, but we still convert it to grayscale to make sure
        original_image = cv2.imread(self.image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # apply histogram equalization
        ## part 1: use cv2.equalizeHist() to equalize the histogram of the grayscale image
        equalized_image_cv = cv2.equalizeHist(original_image)

        ## part 2: use PDF and CDF
        ### calculate the PDF of the original image
        pdf = np.zeros(256)
        for i in range(original_image.shape[0]):
            for j in range(original_image.shape[1]):
                pdf[original_image[i, j]] += 1
        pdf /= original_image.shape[0] * original_image.shape[1]

        ### calculate the CDF of the original image
        cdf = np.zeros(256)
        cdf[0] = pdf[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + pdf[i]

        ### calculate the mapping function
        mapping = np.zeros(256)
        for i in range(256):
            mapping[i] = np.round(cdf[i] * 255)

        ### apply the mapping function to the original image
        equalized_image_manual = np.zeros(original_image.shape, dtype=np.uint8)
        for i in range(original_image.shape[0]):
            for j in range(original_image.shape[1]):
                equalized_image_manual[i, j] = mapping[original_image[i, j]]

        # calculate the histogram of the images
        ## original image
        hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256]).ravel()
        ## part 1
        hist_cv = cv2.calcHist([equalized_image_cv], [0], None, [256], [0, 256]).ravel()
        ## part 2
        hist_manual = cv2.calcHist([equalized_image_manual], [0], None, [256], [0, 256]).ravel()

        # use matplot to display the image and histogram
        ## image
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.axis("off")
        plt.imshow(original_image)
        plt.subplot(2, 3, 2)
        plt.title("Equalized with OpenCV")
        plt.axis("off")
        plt.imshow(equalized_image_cv)
        plt.subplot(2, 3, 3)
        plt.title("Equalized Manually")
        plt.axis("off")
        plt.imshow(equalized_image_manual)
        ## histogram
        plt.subplot(2, 3, 4)
        plt.title("Histogram of Original")
        plt.bar(np.arange(256), hist_original)
        plt.subplot(2, 3, 5)
        plt.title("Histogram of Equalized (OpenCV)")
        plt.bar(np.arange(256), hist_cv)
        plt.subplot(2, 3, 6)
        plt.title("Histogram of Equalized (Manually)")
        plt.bar(np.arange(256), hist_manual)
        plt.show()

    def closing(self):
        if self.image is None:
            return

        # read the image
        original_image = cv2.imread(self.image)

        # convert to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Binarize the grayscale image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Pad the image with zeros based on the kernel size (K=3)
        kernel_size = 3
        padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode="constant", constant_values=0)

        # Define a 3x3 all-ones structuring element
        structuring_element = np.ones((3, 3), np.uint8)

        # Perform the dilation operation
        dilated_image = np.zeros_like(padded_image)
        for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
                dilation_result = np.max(
                    padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1]
                    * structuring_element
                )
                dilated_image[i, j] = dilation_result

        # Perform the erosion operation
        eroded_image = np.zeros_like(dilated_image)
        for i in range(kernel_size // 2, dilated_image.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, dilated_image.shape[1] - kernel_size // 2):
                erosion_result = np.min(
                    dilated_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1]
                    * structuring_element
                )
                eroded_image[i, j] = erosion_result

        # Show the resulting image in a popup window
        plt.imshow(eroded_image, cmap="gray")
        plt.title("Closing Operation Result")
        plt.show()

    def opening(self):
        if self.image is None:
            return

        # read the image
        original_image = cv2.imread(self.image)

        # convert to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Binarize the grayscale image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Pad the image with zeros based on the kernel size (K=3)
        kernel_size = 3
        padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode="constant", constant_values=0)

        # Define a 3x3 all-ones structuring element
        structuring_element = np.ones((3, 3), np.uint8)

        # Perform the erosion operation
        eroded_image = np.zeros_like(padded_image)
        for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
                erosion_result = np.min(
                    padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1]
                    * structuring_element
                )
                eroded_image[i, j] = erosion_result

        # Perform the dilation operation
        dilated_image = np.zeros_like(eroded_image)
        for i in range(kernel_size // 2, eroded_image.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, eroded_image.shape[1] - kernel_size // 2):
                dilation_result = np.max(
                    eroded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1]
                    * structuring_element
                )
                dilated_image[i, j] = dilation_result

        # Show the resulting image in a popup window
        plt.imshow(dilated_image, cmap="gray")
        plt.title("Opening Operation Result")
        plt.show()

    def show_model_structure_vgg19(self):
        # initialize the model
        model = torchvision.models.vgg19_bn(num_classes=10)
        # print the model structure
        summary(model, (3, 32, 32))

    def show_accuracy_and_loss(self):
        cwd = os.getcwd()
        mnist_acc_loss = os.path.join(cwd, "mnist.png")
        self.mnist_graphic_view.scene().clear()
        self.mnist_graphic_view.fitInView(QGraphicsPixmapItem(QPixmap(mnist_acc_loss)))
        self.mnist_graphic_view.scene().addItem(QGraphicsPixmapItem(QPixmap(mnist_acc_loss)))

    def predict(self):
        # initialize the model
        if self.mnist_model is None:
            model = torchvision.models.vgg19_bn(num_classes=10)
            model.load_state_dict(torch.load("./model/mnist.pth", map_location=torch.device("cpu")))
        else:
            model = self.mnist_model

        # get the image drawn on the drawing board
        image = self.mnist_graphic_view.grab().toImage()
        print(image.save("drawn_img.png"))
        image = Image.open("drawn_img.png")

        # preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),  # Convert to three channels
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        image = transform(image).unsqueeze(0)

        # inference
        model.eval()
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1)
            self.mnist_predicition_text.setText("Prediction: {}".format(prediction.item()))

        # show the probability of each class
        # labels
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(output.data)
        probabilities = F.softmax(output.data, dim=1).squeeze()
        print(probabilities)
        # show the probability distribution of model prediction in new window.
        plt.figure(figsize=(10, 5))
        plt.bar(labels, probabilities)
        plt.xticks(labels)
        plt.title("Probability of each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.show()

    def reset(self):
        self.mnist_graphic_view.scene().clear()
        self.mnist_graphic_view.setScene(DrawingScene())

    def load_images(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", os.getcwd(), "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.image = image_path
        else:
            QMessageBox.warning(self, "Warning", "Please select an image file.")
            return

        # show the image
        self.cat_dog_image_view.scene().clear()
        # fit the image to the image view
        self.cat_dog_image_view.fitInView(QGraphicsPixmapItem(QPixmap(self.image)))
        # add the image to the scene
        self.cat_dog_image_view.scene().addItem(QGraphicsPixmapItem(QPixmap(self.image)))

    def show_images(self):
        # read the current working directory
        cwd = os.getcwd()
        inference_dataset_path = os.path.join(cwd, "inference_dataset")
        # access the 'Cat' and 'Dog' folders
        cat_folder_path = os.path.join(inference_dataset_path, "Cat")
        dog_folder_path = os.path.join(inference_dataset_path, "Dog")
        # randomly select 1 image from each folder
        cat_image_path = os.path.join(cat_folder_path, np.random.choice(os.listdir(cat_folder_path)))
        dog_image_path = os.path.join(dog_folder_path, np.random.choice(os.listdir(dog_folder_path)))
        # plot the images
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(cat_image_path))
        plt.title("Cat")
        plt.subplot(1, 2, 2)
        plt.imshow(Image.open(dog_image_path))
        plt.title("Dog")
        plt.show()

    def show_model_structure_resnet50(self):
        # initialize the model
        model = torchvision.models.resnet50(num_classes=2)
        # print the model structure
        summary(model, (3, 224, 224))

    def show_comparison(self):
        cwd = os.getcwd()
        comparison_image_path = os.path.join(cwd, "model_comparison.png")
        plt.imshow(Image.open(comparison_image_path))
        plt.axis("off")
        plt.show()

    def inference(self):
        if self.image is None:
            return
        else:
            image = Image.open(self.image)

        if self.cat_dog_model is None:
            model = torchvision.models.resnet50(num_classes=2)
            model.load_state_dict(torch.load("./model/cat_dog_resnet50_re.pth", map_location=torch.device("cpu")))
        else:
            model = self.cat_dog_model

        # preprocess the image
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image = transform(image).unsqueeze(0)

        # inference
        model.eval()
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1)
            if prediction == 0:
                self.cat_dog_predicition_text.setText("Prediction: Cat")
            else:
                self.cat_dog_predicition_text.setText("Prediction: Dog")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
