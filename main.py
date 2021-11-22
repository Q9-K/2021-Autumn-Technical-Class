from typing import Text
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QFileDialog
from PyQt5.QtGui import QImage, QPainter, QColor, QFont, QPixmap
from PyQt5.QtCore import QPointF, QSizeF, QRectF, pyqtPickleProtocol
import cv2 as cv
import numpy as np
# from PyQt5 import Qt

# from PyQt5.uic import loadUi #这个引用应该是将ui文件转换为py文件用到的

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.scale = 1
        self.ratio_pre = 1
        self.ratio_now = 1
        self.originscale = 1
        self.origin = QPointF(0,0)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(self.rect(),QColor(1,1,1))

        painter.drawPixmap(QRectF(self.origin, QSizeF(self.pixmap.width()*self.scale, self.pixmap.height()*self.scale)), self.pixmap, QRectF(self.pixmap.rect()))

        painter.end()

    def wheelEvent(self, event):#定义滚轮滚动角度实现缩放
        #这里只是做到了窗口最大化时的居中缩放，暂时还不知道如果窗口本身未最大化时如何实现窗口居中显示
        # pass
        angle = event.angleDelta() / 8
        angleY = angle.y()
        w = self.pixmap.width()
        h = self.pixmap.height()
        if angleY > 0: 
            self.scale*=1.1
            self.ratio_now*=1.1
        else:
            self.scale/=1.1
            self.ratio_now/=1.1
        self.changeorigin(self.pixmap)
        self.update()

    def changeorigin(self,img):#设置wheelevent的origin坐标，实现居中缩放
        self.pixmap = img
        self.origin.setX(self.origin.x()+(self.ratio_pre-self.ratio_now)*self.pixmap.width()*self.originscale/2)#通过数学几何计算得到的表达式，比较直观
        self.origin.setY(self.origin.y()+(self.ratio_pre-self.ratio_now)*self.pixmap.height()*self.originscale/2)
        self.ratio_pre=self.ratio_now

    def setPixmap(self, img): #设置pixmap展示像素,且为初始化像素
        self.pixmap = img

        r1 = self.width() / self.pixmap.width()
        r2 = self.height() / self.pixmap.height()
        if r1 < r2:
            self.scale = r1
        else:
            self.scale = r2       
        self.originscale = self.scale
        self.origin.setX((self.width()-self.pixmap.width()*self.scale)/2) 
        self.origin.setY((self.height()-self.pixmap.height()*self.scale)/2)
        self.update()

    # def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        # return super().mousePressEvent(a0)
    # 这里想要实现一个当图片尺寸超出画框外时，能够拖动图片查看细节的功能，由于时间关系先不实现了


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fn = None       
        self.setWindowTitle('异常检测')

        hblayout = QHBoxLayout()
        spacer = QSpacerItem(150, 10, QSizePolicy.Expanding)
        hblayout.addSpacerItem(spacer)
        btn1 = QPushButton()
        btn1.clicked.connect(self.open)
        btn1.setText('打开')
        hblayout.addWidget(btn1)
        btn2 = QPushButton()
        btn2.clicked.connect(self.detect)
        btn2.setText('检测')
        hblayout.addWidget(btn2)
        btn3 = QPushButton()
        btn3.clicked.connect(self.save)
        btn3.setText('保存结果')
        hblayout.addWidget(btn3)

        self.image_viewer = ImageViewer()
        vblayout = QVBoxLayout()
        vblayout.addWidget(self.image_viewer)
        vblayout.addLayout(hblayout)

        self.setLayout(vblayout)
        self.wheelEvent = self.image_viewer.wheelEvent

    def open(self):
        # fn = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg *.gif)")
        self.fn = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg *.png)")
        if len(self.fn[0]) > 0:
            self.image_viewer.setPixmap(QPixmap(self.fn[0]))

    def detect(self):#这里应该引用padim的函数，这里暂时用自己随便写的一个函数替代
        # pass
        painter_1 = QPainter()
        painter_1.begin(QPixmap(self.fn[0]))#错误提示不让paint一个已经paint的device,查询不到pyqt直接对图片操作的method，考虑使用opencv库
        painter_1.fillRect(self.rect(),QColor(1,1,1))

        painter_1.drawText(1, 2, 3, 4, "detect")
        painter_1.drawPixmap(QRectF(self.origin, QSizeF(self.pixmap.width()*self.scale, self.pixmap.height()*self.scale)), self.pixmap, QRectF(self.pixmap.rect()))

        painter_1.end()

        
        

    def save(self):#这里我没查getsavefilename的具体参数的要求，大概试了一下应该没问题
        fd, type = QFileDialog.getSaveFileName(self.image_viewer, "保存图片", "", "*.png")
        QPixmap(self.fn[0]).save(fd)

        

if  __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    # window.show()
    sys.exit(app.exec_())#将app.exex_替换为该行可避免打开一张图片后再次点击打开图片时异常