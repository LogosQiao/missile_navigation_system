# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1378, 1025)
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(40, 10, 581, 571))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.splitter = QtWidgets.QSplitter(self.frame)
        self.splitter.setGeometry(QtCore.QRect(10, 10, 561, 551))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.label_63 = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_63.sizePolicy().hasHeightForWidth())
        self.label_63.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_63.setFont(font)
        self.label_63.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_63.setTextFormat(QtCore.Qt.AutoText)
        self.label_63.setAlignment(QtCore.Qt.AlignCenter)
        self.label_63.setObjectName("label_63")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.text_input = QtWidgets.QTextBrowser(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.text_input.sizePolicy().hasHeightForWidth())
        self.text_input.setSizePolicy(sizePolicy)
        self.text_input.setMinimumSize(QtCore.QSize(0, 22))
        self.text_input.setMaximumSize(QtCore.QSize(560, 35))
        self.text_input.setObjectName("text_input")
        self.gridLayout.addWidget(self.text_input, 0, 1, 1, 1)
        self.butt_input = QtWidgets.QPushButton(self.layoutWidget)
        self.butt_input.setObjectName("butt_input")
        self.gridLayout.addWidget(self.butt_input, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.text_output = QtWidgets.QTextBrowser(self.layoutWidget)
        self.text_output.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.text_output.sizePolicy().hasHeightForWidth())
        self.text_output.setSizePolicy(sizePolicy)
        self.text_output.setMinimumSize(QtCore.QSize(70, 35))
        self.text_output.setMaximumSize(QtCore.QSize(560, 35))
        self.text_output.setObjectName("text_output")
        self.gridLayout.addWidget(self.text_output, 1, 1, 1, 1)
        self.butt_output = QtWidgets.QPushButton(self.layoutWidget)
        self.butt_output.setObjectName("butt_output")
        self.gridLayout.addWidget(self.butt_output, 1, 2, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.splitter)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.layoutWidget1 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 4, 1, 1)
        self.err_lambda = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_lambda.setObjectName("err_lambda")
        self.gridLayout_2.addWidget(self.err_lambda, 0, 3, 1, 1)
        self.err_vn = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_vn.setObjectName("err_vn")
        self.gridLayout_2.addWidget(self.err_vn, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 1, 4, 1, 1)
        self.err_phi = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_phi.setObjectName("err_phi")
        self.gridLayout_2.addWidget(self.err_phi, 2, 5, 1, 1)
        self.err_h = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_h.setObjectName("err_h")
        self.gridLayout_2.addWidget(self.err_h, 0, 5, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 2, 1, 1)
        self.err_gamma = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_gamma.setObjectName("err_gamma")
        self.gridLayout_2.addWidget(self.err_gamma, 2, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 2, 1, 1)
        self.err_l = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_l.setObjectName("err_l")
        self.gridLayout_2.addWidget(self.err_l, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_25.setObjectName("label_25")
        self.gridLayout_2.addWidget(self.label_25, 2, 4, 1, 1)
        self.err_vu = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_vu.setObjectName("err_vu")
        self.gridLayout_2.addWidget(self.err_vu, 1, 3, 1, 1)
        self.err_sigma = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_sigma.setObjectName("err_sigma")
        self.gridLayout_2.addWidget(self.err_sigma, 2, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 2, 2, 1, 1)
        self.err_ve = QtWidgets.QLineEdit(self.layoutWidget1)
        self.err_ve.setObjectName("err_ve")
        self.gridLayout_2.addWidget(self.err_ve, 1, 5, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.splitter)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.layoutWidget2 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 0, 0, 1, 1)
        self.ax_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ax_bd.setObjectName("ax_bd")
        self.gridLayout_3.addWidget(self.ax_bd, 0, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_18.setObjectName("label_18")
        self.gridLayout_3.addWidget(self.label_18, 0, 2, 1, 1)
        self.ax_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ax_lp.setObjectName("ax_lp")
        self.gridLayout_3.addWidget(self.ax_lp, 0, 3, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_26.setObjectName("label_26")
        self.gridLayout_3.addWidget(self.label_26, 0, 4, 1, 1)
        self.ax_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ax_wdx.setObjectName("ax_wdx")
        self.gridLayout_3.addWidget(self.ax_wdx, 0, 5, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 1, 0, 1, 1)
        self.ay_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ay_bd.setObjectName("ay_bd")
        self.gridLayout_3.addWidget(self.ay_bd, 1, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_19.setObjectName("label_19")
        self.gridLayout_3.addWidget(self.label_19, 1, 2, 1, 1)
        self.ay_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ay_lp.setObjectName("ay_lp")
        self.gridLayout_3.addWidget(self.ay_lp, 1, 3, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_27.setObjectName("label_27")
        self.gridLayout_3.addWidget(self.label_27, 1, 4, 1, 1)
        self.ay_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.ay_wdx.setObjectName("ay_wdx")
        self.gridLayout_3.addWidget(self.ay_wdx, 1, 5, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 2, 0, 1, 1)
        self.az_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.az_bd.setObjectName("az_bd")
        self.gridLayout_3.addWidget(self.az_bd, 2, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_20.setObjectName("label_20")
        self.gridLayout_3.addWidget(self.label_20, 2, 2, 1, 1)
        self.az_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.az_lp.setObjectName("az_lp")
        self.gridLayout_3.addWidget(self.az_lp, 2, 3, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_28.setObjectName("label_28")
        self.gridLayout_3.addWidget(self.label_28, 2, 4, 1, 1)
        self.az_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.az_wdx.setObjectName("az_wdx")
        self.gridLayout_3.addWidget(self.az_wdx, 2, 5, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 3, 0, 1, 1)
        self.wx_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wx_bd.setObjectName("wx_bd")
        self.gridLayout_3.addWidget(self.wx_bd, 3, 1, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_21.setObjectName("label_21")
        self.gridLayout_3.addWidget(self.label_21, 3, 2, 1, 1)
        self.wx_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wx_lp.setObjectName("wx_lp")
        self.gridLayout_3.addWidget(self.wx_lp, 3, 3, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_29.setObjectName("label_29")
        self.gridLayout_3.addWidget(self.label_29, 3, 4, 1, 1)
        self.wx_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wx_wdx.setObjectName("wx_wdx")
        self.gridLayout_3.addWidget(self.wx_wdx, 3, 5, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 4, 0, 1, 1)
        self.wy_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wy_bd.setObjectName("wy_bd")
        self.gridLayout_3.addWidget(self.wy_bd, 4, 1, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_22.setObjectName("label_22")
        self.gridLayout_3.addWidget(self.label_22, 4, 2, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_30.setObjectName("label_30")
        self.gridLayout_3.addWidget(self.label_30, 4, 4, 1, 1)
        self.wy_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wy_wdx.setObjectName("wy_wdx")
        self.gridLayout_3.addWidget(self.wy_wdx, 4, 5, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 5, 0, 1, 1)
        self.wz_bd = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wz_bd.setObjectName("wz_bd")
        self.gridLayout_3.addWidget(self.wz_bd, 5, 1, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_23.setObjectName("label_23")
        self.gridLayout_3.addWidget(self.label_23, 5, 2, 1, 1)
        self.wz_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wz_lp.setObjectName("wz_lp")
        self.gridLayout_3.addWidget(self.wz_lp, 5, 3, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_31.setObjectName("label_31")
        self.gridLayout_3.addWidget(self.label_31, 5, 4, 1, 1)
        self.wz_wdx = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wz_wdx.setObjectName("wz_wdx")
        self.gridLayout_3.addWidget(self.wz_wdx, 5, 5, 1, 1)
        self.wy_lp = QtWidgets.QLineEdit(self.layoutWidget2)
        self.wy_lp.setObjectName("wy_lp")
        self.gridLayout_3.addWidget(self.wy_lp, 4, 3, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.splitter)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.layoutWidget3 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_16 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_2.addWidget(self.label_16)
        self.start_point = QtWidgets.QLineEdit(self.layoutWidget3)
        self.start_point.setObjectName("start_point")
        self.horizontalLayout_2.addWidget(self.start_point)
        self.label_24 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_2.addWidget(self.label_24)
        self.duration = QtWidgets.QLineEdit(self.layoutWidget3)
        self.duration.setObjectName("duration")
        self.horizontalLayout_2.addWidget(self.duration)
        self.label_32 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_32.setObjectName("label_32")
        self.horizontalLayout_2.addWidget(self.label_32)
        self.count = QtWidgets.QLineEdit(self.layoutWidget3)
        self.count.setObjectName("count")
        self.horizontalLayout_2.addWidget(self.count)
        self.para_set = QtWidgets.QPushButton(self.layoutWidget3)
        self.para_set.setObjectName("para_set")
        self.horizontalLayout_2.addWidget(self.para_set)
        self.layoutWidget4 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget4)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)
        self.calculate = QtWidgets.QPushButton(self.layoutWidget4)
        self.calculate.setObjectName("calculate")
        self.horizontalLayout.addWidget(self.calculate)
        self.plot = QtWidgets.QPushButton(self.layoutWidget4)
        self.plot.setObjectName("plot")
        self.horizontalLayout.addWidget(self.plot)
        self.line = QtWidgets.QFrame(self.splitter)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.layoutWidget5 = QtWidgets.QWidget(self.splitter)
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.loss_analysis = QtWidgets.QPushButton(self.layoutWidget5)
        self.loss_analysis.setMaximumSize(QtCore.QSize(80, 16777215))
        self.loss_analysis.setObjectName("loss_analysis")
        self.horizontalLayout_3.addWidget(self.loss_analysis)
        self.loss_display = QtWidgets.QTextBrowser(self.layoutWidget5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loss_display.sizePolicy().hasHeightForWidth())
        self.loss_display.setSizePolicy(sizePolicy)
        self.loss_display.setMaximumSize(QtCore.QSize(615, 80))
        self.loss_display.setObjectName("loss_display")
        self.horizontalLayout_3.addWidget(self.loss_display)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Navigation"))
        self.label_63.setText(_translate("Form", "捷联惯性导航仿真系统"))
        self.label.setText(_translate("Form", "输入文件: "))
        self.butt_input.setText(_translate("Form", "选择文件"))
        self.label_2.setText(_translate("Form", "输出文件:"))
        self.butt_output.setText(_translate("Form", "选择路径"))
        self.label_7.setText(_translate("Form", "高度误差:"))
        self.err_lambda.setPlaceholderText(_translate("Form", "0.0"))
        self.err_vn.setPlaceholderText(_translate("Form", "0.0"))
        self.label_8.setText(_translate("Form", "东速误差:"))
        self.err_phi.setPlaceholderText(_translate("Form", "0.0"))
        self.err_h.setPlaceholderText(_translate("Form", "0.0"))
        self.label_9.setText(_translate("Form", "俯仰误差:"))
        self.label_5.setText(_translate("Form", "纬度误差:"))
        self.err_gamma.setPlaceholderText(_translate("Form", "0.0"))
        self.label_6.setText(_translate("Form", "天速误差:"))
        self.err_l.setPlaceholderText(_translate("Form", "0.0"))
        self.label_4.setText(_translate("Form", "北速误差:"))
        self.label_25.setText(_translate("Form", "偏航误差:"))
        self.err_vu.setPlaceholderText(_translate("Form", "0.0"))
        self.err_sigma.setPlaceholderText(_translate("Form", "0.0"))
        self.label_17.setText(_translate("Form", "滚转误差:"))
        self.err_ve.setPlaceholderText(_translate("Form", "0.0"))
        self.label_3.setText(_translate("Form", "经度误差:"))
        self.label_10.setText(_translate("Form", "a_x 标度:"))
        self.ax_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_18.setText(_translate("Form", "a_x 零偏:"))
        self.ax_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_26.setText(_translate("Form", "a_x 零偏稳定性:"))
        self.ax_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.label_11.setText(_translate("Form", "a_y 标度:"))
        self.ay_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_19.setText(_translate("Form", "a_y 零偏:"))
        self.ay_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_27.setText(_translate("Form", "a_y 零偏稳定性:"))
        self.ay_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.label_12.setText(_translate("Form", "a_z 标度:"))
        self.az_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_20.setText(_translate("Form", "a_z 零偏:"))
        self.az_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_28.setText(_translate("Form", "a_z 零偏稳定性:"))
        self.az_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.label_13.setText(_translate("Form", "ω_x 标度:"))
        self.wx_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_21.setText(_translate("Form", "ω_x 零偏:"))
        self.wx_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_29.setText(_translate("Form", "ω_x 零偏稳定性:"))
        self.wx_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.label_14.setText(_translate("Form", "ω_y 标度:"))
        self.wy_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_22.setText(_translate("Form", "ω_y 零偏:"))
        self.label_30.setText(_translate("Form", "ω_y 零偏稳定性:"))
        self.wy_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.label_15.setText(_translate("Form", "ω_z 标度:"))
        self.wz_bd.setPlaceholderText(_translate("Form", "0.0"))
        self.label_23.setText(_translate("Form", "ω_z 零偏:"))
        self.wz_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_31.setText(_translate("Form", "ω_z 零偏稳定性:"))
        self.wz_wdx.setPlaceholderText(_translate("Form", "0.0"))
        self.wy_lp.setPlaceholderText(_translate("Form", "0.0"))
        self.label_16.setText(_translate("Form", "仿真起点:"))
        self.start_point.setPlaceholderText(_translate("Form", "0"))
        self.label_24.setText(_translate("Form", "仿真时长:"))
        self.duration.setPlaceholderText(_translate("Form", "0"))
        self.label_32.setText(_translate("Form", "仿真次数:"))
        self.count.setPlaceholderText(_translate("Form", "0"))
        self.para_set.setText(_translate("Form", "设置"))
        self.calculate.setText(_translate("Form", "开始计算"))
        self.plot.setText(_translate("Form", "画图"))
        self.loss_analysis.setText(_translate("Form", "误差分析"))
