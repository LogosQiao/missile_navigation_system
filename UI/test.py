# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1352, 1048)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.chooseA = QtWidgets.QPushButton(self.centralwidget)
        self.chooseA.setGeometry(QtCore.QRect(290, 40, 89, 25))
        self.chooseA.setObjectName("chooseA")
        self.FileA = QtWidgets.QTextBrowser(self.centralwidget)
        self.FileA.setGeometry(QtCore.QRect(10, 30, 256, 41))
        self.FileA.setObjectName("FileA")
        self.set = QtWidgets.QPushButton(self.centralwidget)
        self.set.setGeometry(QtCore.QRect(300, 240, 89, 25))
        self.set.setObjectName("set")
        self.lineA = QtWidgets.QLineEdit(self.centralwidget)
        self.lineA.setGeometry(QtCore.QRect(30, 150, 113, 25))
        self.lineA.setObjectName("lineA")
        self.lineB = QtWidgets.QLineEdit(self.centralwidget)
        self.lineB.setGeometry(QtCore.QRect(30, 200, 113, 25))
        self.lineB.setObjectName("lineB")
        self.lineC = QtWidgets.QLineEdit(self.centralwidget)
        self.lineC.setGeometry(QtCore.QRect(30, 250, 113, 25))
        self.lineC.setObjectName("lineC")
        self.FileB = QtWidgets.QTextBrowser(self.centralwidget)
        self.FileB.setGeometry(QtCore.QRect(10, 90, 256, 41))
        self.FileB.setObjectName("FileB")
        self.chooseB = QtWidgets.QPushButton(self.centralwidget)
        self.chooseB.setGeometry(QtCore.QRect(290, 100, 89, 25))
        self.chooseB.setObjectName("chooseB")
        self.calculate = QtWidgets.QPushButton(self.centralwidget)
        self.calculate.setGeometry(QtCore.QRect(310, 430, 89, 25))
        self.calculate.setObjectName("calculate")
        self.result = QtWidgets.QTextBrowser(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(20, 310, 256, 192))
        self.result.setObjectName("result")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1352, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.chooseA.setText(_translate("MainWindow", "choose file"))
        self.set.setText(_translate("MainWindow", "set para"))
        self.chooseB.setText(_translate("MainWindow", "choose file"))
        self.calculate.setText(_translate("MainWindow", "calculation"))
