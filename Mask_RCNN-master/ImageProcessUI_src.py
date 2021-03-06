# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ImageProcessUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(943, 623)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        Form.setFont(font)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 410, 311, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_resolution = QtWidgets.QLabel(self.layoutWidget)
        self.label_resolution.setMinimumSize(QtCore.QSize(200, 0))
        self.label_resolution.setObjectName("label_resolution")
        self.horizontalLayout_2.addWidget(self.label_resolution)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.layoutWidget_2 = QtWidgets.QWidget(Form)
        self.layoutWidget_2.setGeometry(QtCore.QRect(20, 470, 311, 41))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.label_contrast = QtWidgets.QLabel(self.layoutWidget_2)
        self.label_contrast.setMinimumSize(QtCore.QSize(200, 0))
        self.label_contrast.setObjectName("label_contrast")
        self.horizontalLayout_3.addWidget(self.label_contrast)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Form)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 350, 309, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_brightness = QtWidgets.QLabel(self.layoutWidget1)
        self.label_brightness.setMinimumSize(QtCore.QSize(200, 0))
        self.label_brightness.setObjectName("label_brightness")
        self.horizontalLayout.addWidget(self.label_brightness)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(Form)
        self.layoutWidget2.setGeometry(QtCore.QRect(20, 90, 271, 208))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btnReadFile = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnReadFile.setObjectName("btnReadFile")
        self.verticalLayout.addWidget(self.btnReadFile)

        self.btnimgali2 = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnimgali2.setObjectName("btnimgali2")
        self.verticalLayout.addWidget(self.btnimgali2)


        self.btnShowOriginalFile = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnShowOriginalFile.setObjectName("btnShowOriginalFile")
        self.verticalLayout.addWidget(self.btnShowOriginalFile)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.btnRotateFile = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnRotateFile.setObjectName("btnRotateFile")
        self.horizontalLayout_5.addWidget(self.btnRotateFile)
        self.dspinbox_rotate_angle = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.dspinbox_rotate_angle.setMinimum(-180.0)
        self.dspinbox_rotate_angle.setMaximum(180.0)
        self.dspinbox_rotate_angle.setObjectName("dspinbox_rotate_angle")
        self.horizontalLayout_5.addWidget(self.dspinbox_rotate_angle)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.horizontalSlider_rotangle = QtWidgets.QSlider(self.layoutWidget2)
        self.horizontalSlider_rotangle.setMinimum(-180)
        self.horizontalSlider_rotangle.setMaximum(180)
        self.horizontalSlider_rotangle.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_rotangle.setObjectName("horizontalSlider_rotangle")
        self.horizontalLayout_4.addWidget(self.horizontalSlider_rotangle)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.btnCropFile = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnCropFile.setObjectName("btnCropFile")
        self.verticalLayout.addWidget(self.btnCropFile)
        self.btnSave = QtWidgets.QPushButton(self.layoutWidget2)
        self.btnSave.setObjectName("btnSave")
        self.verticalLayout.addWidget(self.btnSave)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "视频分类"))
        self.label_3.setText(_translate("Form", "图片分辨率："))
        self.label_resolution.setText(_translate("Form", "0"))
        self.label_5.setText(_translate("Form", "图片对比度："))
        self.label_contrast.setText(_translate("Form", "0"))
        self.label.setText(_translate("Form", "图片亮度   ："))
        self.label_brightness.setText(_translate("Form", "0"))
        self.btnReadFile.setText(_translate("Form", "导入图片"))
        self.btnimgali2.setText(_translate("Form", "条纹对齐"))


        self.btnShowOriginalFile.setText(_translate("Form", "显示原图"))
        self.btnRotateFile.setText(_translate("Form", "旋转图片"))
        self.label_2.setText(_translate("Form", "旋转图片"))
        self.btnCropFile.setText(_translate("Form", "裁剪图片"))
        self.btnSave.setText(_translate("Form", "保存图片"))
