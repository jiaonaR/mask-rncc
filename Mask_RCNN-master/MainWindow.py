import math
import sys
import cv2 as cv
import numpy
import numpy as np
from PIL import Image, ImageQt, ImageStat

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTableWidgetItem, QFrame

from ImageProcessUI2 import Ui_Form  ## 调用 已经设计好的界面 系统
# from ImageProcessUI_src import Ui_Form

import numpy as np
import pandas as pd
import os
import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen

from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget

class ImageLabel(QLabel):
    def __init__(self, *__args):
        super(ImageLabel, self).__init__(*__args)
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.open_mouse_flag = False
        self.select_roi_flag = False
        self.draw_roi_flag = False
        self.clear_flag = False
        self.rect = QRect()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.open_mouse_flag is True:
            self.select_roi_flag = True
            self.x0 = event.x()
            self.y0 = event.y()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.select_roi_flag = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.select_roi_flag is True:
            self.x1 = event.x()
            self.y1 = event.y()
            if self.draw_roi_flag is True:
                self.update()

    def paintEvent(self, event: QtGui.QMouseEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        if self.clear_flag is True:
            self.x0 = 0
            self.y0 = 0
            self.x1 = 0
            self.y1 = 0
        self.rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter.drawRect(self.rect)
        self.update()

from keras.models import Model


from urllib.request import urlopen

from PIL import Image
from shutil import copyfile


import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from torchvision import transforms

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1.,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def video_process(filename):
    pathOut = dir_choose + "_output_dir/"
    if not os.path.exists(pathOut): os.mkdir(pathOut)  ## 新建处理结果文件夹
    listing = os.listdir(dir_choose)
    count = 0
    counter = 1

    for vid in listing:  ## 遍历所有需要处理的视频 每个视频 抽取其中的一帧 1/3 处 作为该视频的代表
        vid1 = dir_choose + '/' + vid
        cap = cv2.VideoCapture(vid1)
        frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 3)  # 取视频的 1/3处的视频帧 作为该视频的代表帧
        count = 0
        counter += 1
        success = True
        while success:
            success, image = cap.read()
            print('read a new frame:', success)
            if count == frame:
                cv2.imwrite(pathOut + str(vid) + ".jpg", image)  ##保存  1/3处的视频帧
                print(success)
            count += 1
    n = 0
    test_files = []
    test_features = []
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  ### 利用开源resnet50 模型 以及其模型参数 来做视频的 特征分析
    images_path = pathOut
    for f in os.listdir(images_path):
        test_files.append(f)
    # Looping over every image present in the files list

    for img_path in test_files:  # 遍历所有已经 从视频里抽取出来的 视频帧
        test_img_p = images_path + str(img_path)
        if (os.stat(images_path + str(img_path)).st_size != 0):
            # if 1:
            print(str(img_path))
            n += 1
            print(n)
            # load the image and resize it
            # img = image.load_img("./test_output_dir/" + str(img_path), target_size=(224, 224))
            # img = image.load_img(images_path + str(img_path), target_size=(224, 224))
            img = load_img(images_path + str(img_path), target_size=(224, 224))  ## 读取视频帧图像 作为一个矩阵
            # extract features from each image
            # x_image = image.img_to_array(img)
            x_image = img_to_array(img)
            x_image = np.expand_dims(x_image,
                                     axis=0)  # increase dimensions of x to make it suitable for further feature extraction
            x_image = preprocess_input(x_image)  # 图送进模型之前 需要做一个归一化 适应原始模型 确保预测的准确
            x_features = model.predict(x_image)  # extract image features from model
            x_features = np.array(x_features)  # convert features list to numpy array
            x_flatten = x_features.flatten()  # flatten out the features in x
            test_features.append(x_flatten)  # this list contains the final features of the test images
    ss = StandardScaler()
    train_features = np.load("train_features_rnn_noss.npy")
    train_features = ss.fit_transform(train_features)  ## 在训练数据集上 做一个 拟合，获取所有训练数据的特征 参数
    test_features = ss.transform(test_features)  # scale the test video frames
    # test_features = self.ss.fittransform(test_features)  # scale the test video frames
    np.save("test_features_rnn_test14.npy", test_features)  ## 把测试数据的 拟合结果 保存下来 后续进行分析
    QMessageBox.warning(self, "info", "视频处理完毕")


class MainWindow(QMainWindow, Ui_Form):  # 主界面 由 ui_form 产生
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.current_open_image = None     # 当前打开图片
        # self.current_process_image = None  # 当前正在被操作的图片
        # self.current_cropped_image = None

        # self.open_keyboard_flag = False
        # self.image_rotated = False
        # self.image_cropped = False

        self.init_signals_slots()  ## pyqt 信号槽机制 前端控制 信号 通过槽 到后端 与后端算法 交互
        self.label_Image = ImageLabel(self)
        self.MyTable = QtWidgets.QTableWidget()
        self.init_ui()
        self.mp4name = ''


    def init_signals_slots(self):
        self.btnReadFile.clicked.connect(self.btnReadFile_clicked)  ## 点击 视频处理按钮 产生点击响应信号 该信号 触发回调 函数 self.btnReadFile_clicked
        # self.btnCropFile.clicked.connect(self.btnCropFile_clicked)
        self.btnShowOriginalFile.clicked.connect(self.btnShowOriginalFile_clicked)
        # self.btnimgali2.clicked.connect(self.imgalign)
        # self.btnRotateFile.clicked.connect(self.btnRotateFile_clicked)
        self.btnSave.clicked.connect(self.btnSaveFile_clicked)
        # self.horizontalSlider_rotangle.valueChanged.connect(self.sliderRotangle_valuechanged)

    def init_ui(self):
        # pass
        self.label_Image.setGeometry(370, 10, 530, 580)
        self.label_Image.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.label_Image.setText("显示视频帧")
        self.label_Image.setAlignment(Qt.AlignCenter)
        # self.player = QMediaPlayer()
        # self.vw = QVideoWidget()
        # self.vw.show()
        #
        # self.player.setVideoOutput(self.vw)  # 视频播放输出的widget，就是上面定义的



    def show_image_info(self, q_img):
        pil_img = ImageQt.fromqimage(q_img)
        self.label_resolution.setText(f"{q_img.size().width()}px x {q_img.size().height()}px")  # 显示分辨率
        # 亮度，使用perceived brightness亮度算法
        # https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
        stat = ImageStat.Stat(pil_img)
        r, g, b = stat.mean
        self.label_brightness.setText(f"{math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2)):.2f} nits")
        # 对比度，使用RMS对比度算法
        # https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
        self.label_contrast.setText(f"{int(cv.cvtColor(self.current_process_image, cv.COLOR_BGR2GRAY).std())} : 1 ")

    def btnReadFile_clicked(self):
        filename,  _ = QFileDialog.getOpenFileName(self, '选取待分割视频')
        # dir_choose = QFileDialog.getExistingDirectory(self, '待分类视频文件夹') ## 获取待分类的文件夹 路径 并且该路径是全局变量 其他按钮也需要用到改路径 变量
        print(filename)

        # # self.player.setMedia(QMediaContent(filename))  # 选取视频文件
        # self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
        # self.player.play()
        #
        # sys.exit(app.exec_())

        self.mp4name = filename.split('.')[0] +"_seg.mp4"
        cap = cv2.VideoCapture(filename)
        frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) )  # 取视频的 1/3处的视频帧 作为该视频的代表帧
        count = 0
        print(frame)
        # counter += 1
        success = True
        # QMessageBox.warning(self, "info", "视频处理开始")
        for vid in range(frame):
            success, image = cap.read()
            print('read a new frame:', success)
            cv2.imwrite('./test_video_res/' + str(vid) + ".jpg", image)  ##保存  1/3处的视频帧


            if vid ==0:
                rows, cols, channels = image.shape
                q_img = QImage(image, cols, rows, channels * cols, QImage.Format_RGB888)
                self.label_Image.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # else:
            #     QMessageBox.warning(self, "info", "视频处理结束")
            # count += 1

        QMessageBox.warning(self, "info", "视频读入结束")
        # QMessageBox.warning(self, "info", "视频处理结束")
        # exit(0)
        # n = 0
        # test_files = []
        # test_features = []
        # model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  ### 利用开源resnet50 模型 以及其模型参数 来做视频的 特征分析
        # images_path = pathOut
        # for f in os.listdir(images_path):
        #     test_files.append(f)
        # # Looping over every image present in the files list


    def btnSaveFile_clicked(self):
        # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
        savedir = './test_video_seg/'
        size = (640, 368)
        # size = (368, 640)
        videoWriter = cv2.VideoWriter('./test_seg.mp4', fourcc, 24, size )
        ilne = len(os.listdir(savedir))
        print(ilne)
        for i in range(ilne):
            print(savedir+str(i)+'.jpg')
            frame = cv2.imread(savedir+str(i)+'.jpg')
            videoWriter.write(frame)

        # for item in filelist:
        #     if item.endswith('.jpg'):  # 判断图片后缀是否是.jpg
        #         item = path + item
        #         img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        #         # print(type(img))  # numpy.ndarray类型
        #         videoWriter.write(img)  # 把图片写进视频

        videoWriter.release()  # 释放
        QMessageBox.warning(self, "info", "视频保存完毕")


    def sliderRotangle_valuechanged(self):
        if not self.current_open_image is None:
            rotate_angle = self.horizontalSlider_rotangle.value()
            image_pil = Image.fromarray(self.current_process_image)  # cv image -> pillow image
            image_pil = image_pil.rotate(rotate_angle)  # rotate image
            rotate_img = numpy.asarray(image_pil)  # 图片旋转后转为qt图片
            rows, cols, channels = rotate_img.shape
            q_img = QImage(rotate_img.data, cols, rows, channels * cols, QImage.Format_RGB888)
            self.show_image_info(q_img)
            self.label_Image.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        else:
            QMessageBox.warning(self, "Error", "请先导入图片，才可以显示原图")
            return



    def btnRotateFile_clicked(self):
        if not self.current_open_image is None:
            rotate_angle = self.dspinbox_rotate_angle.value()
            image_pil = Image.fromarray(self.current_process_image)  # cv image -> pillow image
            image_pil = image_pil.rotate(rotate_angle)  # rotate image
            rotate_img = numpy.asarray(image_pil)  # 图片旋转后转为qt图片
            rows, cols, channels = rotate_img.shape
            q_img = QImage(rotate_img.data, cols, rows, channels * cols, QImage.Format_RGB888)
            self.show_image_info(q_img)

            self.label_Image.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        else:
            QMessageBox.warning(self, "Error", "请先导入图片，才可以显示原图")
            return

    def btnShowOriginalFile_clicked(self):
        QMessageBox.warning(self, "info", "视频分割开始，这可能需要片刻，请稍等")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
            'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
            'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
            'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
            'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'
        ]

        # class indices corresponding to types of vehicles and bikes

        vehicles_indices = [3]
        # setting the model to evaluation mode for inference
        model.eval()
        savedir = './test_video_seg/'
        video_img_dir = './test_video_res/'
        # imgps = [ video_img_dir + i for i in os.listdir(video_img_dir)]
        imgs = os.listdir(video_img_dir)

        # print(imgps)
        # exit(0)
        # loading PIL Image from path and creating a torch tensor from it
        for img_path in imgs:
            print(video_img_dir+img_path)
            img_pil = PIL.Image.open(video_img_dir+img_path)
            img_tensor = transforms.functional.to_tensor(img_pil)

            cuda = False   ## 如果电脑有cuda就开启这个选项
            if cuda:
                img_tensor = img_tensor.cuda()
                model.cuda()
            else:
                img_tensor = img_tensor.cpu()
                model.cpu()

            predictions = model([img_tensor])

            # saving the image in cv2 to place bounding boxes later
            img_cv = cv2.imread(video_img_dir+img_path)
            masks = []
            print(predictions[0]["scores"])
            for i in range(predictions[0]["boxes"].shape[0]):
                # set the threshold for the prediction as you like. Here is 0.5
                if predictions[0]["scores"][i] > 0.5:
                    label_id = predictions[0]["labels"][i].item()

                    # check if the class predicted is a vehicle. If yes, get the bboxes and masks
                    if label_id in vehicles_indices:
                        bbox = predictions[0]["boxes"][i].detach().cpu().numpy()

                        # draw a rectangle from the bbox on the cv2 image
                        cv2.rectangle(img_cv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                        # cv2.imwrite(savedir + img_path, img_cv)
                        mask = predictions[0]["masks"][i].detach().cpu().numpy().squeeze()

                        # the model returns a mask with values from 0 to 1. Numpy masks ignoring every pixel that's lower than 0.5.
                        _, mask = cv2.threshold(mask, 0.5, 1, 0)
                        seg_img = apply_mask(img_cv, mask, (1, 1, 0))

            cv2.imwrite(savedir + img_path, seg_img)

        QMessageBox.warning(self, "info", "视频分割完毕")





    def imgalign(self):
        if not self.current_open_image is None:
            self.current_process_image = self.current_open_image.copy()  # 重新布置内存
            rows, cols, channels = self.current_open_image.shape
            # current_align_image =  cv.resize(self.current_open_image, (600, 600), interpolation=cv.INTER_CUBIC)
            current_align_image =  self.current_open_image
            aligned = np.zeros((600, 800, 3),dtype=np.uint8) #,interpolation = cv.INTER_CUBIC
            barlist = []
            for j in range(800):
                i = min(int(600*j/799), 599)
                key = current_align_image[i, j ,0]
                if key>200:
                    barlist.append(key)
                    # aligned[:,j,:] = current_align_image[i, j ,:]
                else:barlist.append(0)
                # else:aligned[:,j,:]=0
            for j  in range(800):
                aligned[:, j, :] = barlist[j]

            self.current_process_image = aligned
            # self.current_open_image = aligned

            q_img = QImage(self.current_process_image.data, cols, rows, channels * cols, QImage.Format_RGB888)
            self.show_image_info(q_img)
            self.label_Image.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.label_Image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            QMessageBox.warning(self, "Error", "请先导入图片，才可以显示原图")
            return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  ###### 新建app 来承载我们设计的界面
    md = MainWindow()
    md.show()     ###   显示设计的系统界面
    sys.exit(app.exec_())   ## 如果 代码出现 弹出信号 不管出错或是 异常 都会导致 app 运行结束
