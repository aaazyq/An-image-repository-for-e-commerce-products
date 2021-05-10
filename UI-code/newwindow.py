import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi

class Window2(QMainWindow):
    def __init__(self):
        super(Window2, self).__init__()
        loadUi("newwindow.ui", self)
        self.Browse.clicked.connect(self.browsefiles)
        self.similar.clicked.connect(self.press_submit2)

    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self,
                                            'Open file',
                                            'F:/20Fall-master/ML-cs680/kaggle/uw-cs480-fall20-new/images/shuffled-images')
        self.FileName.setText(fname[0])
        File = fname[0]
        print(File)
        self.inputimage.setPixmap(QtGui.QPixmap(File))

        testimage = Image.open(File).convert('RGB')
        testimagetensor = transform_test(testimage).to(device)
        outputs = net_total(testimagetensor.reshape([1, 3, 48, 48]))
        _, predicted = torch.max(outputs.data, 1)

        category= category_mapping[int(predicted)]
        filtered_data = train_data
        filtered_data = filtered_data.loc[filtered_data["category"] == category,]
        global image_num, image_list
        image_num = len(filtered_data)
        print(image_num)
        self.textResult.setText("Prediction Result:\n  Category: " + category +
                                "\n We got %d images for you!"%image_num)
        self.update()
        image_list = list(filtered_data['id'])



    def press_submit2(self):
        random.shuffle(image_list)
        dir = 'F:/20Fall-master/ML-cs680/kaggle/uw-cs480-fall20-new/images/shuffled-images/'

        if len(image_list) > 0:
            print(image_list[0])
            image_dir = dir + str(image_list[0]) + '.jpg'
            print('image', image_dir)
            self.image1.setPixmap(QtGui.QPixmap(image_dir))
            self.id1.setText("id:%d" % image_list[0])

        if len(image_list) > 1:
            print(image_list[1])
            image_dir = dir + str(image_list[1]) + '.jpg'
            print('image', image_dir)
            self.image2.setPixmap(QtGui.QPixmap(image_dir))
            self.id2.setText("id:%d" % image_list[1])

        if len(image_list) > 2:
            print(image_list[2])
            image_dir = dir + str(image_list[2]) + '.jpg'
            print('image', image_dir)
            self.image3.setPixmap(QtGui.QPixmap(image_dir))
            self.id3.setText("id:%d" % image_list[2])

        if len(image_list) > 3:
            print(image_list[3])
            image_dir = dir + str(image_list[3]) + '.jpg'
            print('image', image_dir)
            self.image4.setPixmap(QtGui.QPixmap(image_dir))
            self.id4.setText("id:%d" % image_list[3])



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from keras.preprocessing.sequence import pad_sequences
import warnings
import random
from PIL import Image
warnings.filterwarnings("ignore")
import pandas as pd
train_data = pd.read_csv("../data/train.csv")
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, Block, num_classes=27):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Block, 64,  2, stride=1)
        self.layer2 = self.make_layer(Block, 128, 2, stride=2)
        self.layer3 = self.make_layer(Block, 256, 2, stride=2)
        self.layer4 = self.make_layer(Block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(Block)

transform_test = transforms.Compose([
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_total = ResNet18().eval().to(device)
net_total.load_state_dict(torch.load('../model/net_total.pth', map_location=torch.device('cpu')))

category_mapping = {0: 'Accessories',
 1: 'Apparel Set',
 2: 'Bags',
 3: 'Belts',
 4: 'Bottomwear',
 5: 'Cufflinks',
 6: 'Dress',
 7: 'Eyewear',
 8: 'Flip Flops',
 9: 'Fragrance',
 10: 'Free Gifts',
 11: 'Headwear',
 12: 'Innerwear',
 13: 'Jewellery',
 14: 'Lips',
 15: 'Loungewear and Nightwear',
 16: 'Makeup',
 17: 'Nails',
 18: 'Sandal',
 19: 'Saree',
 20: 'Scarves',
 21: 'Shoes',
 22: 'Socks',
 23: 'Ties',
 24: 'Topwear',
 25: 'Wallets',
 26: 'Watches'}


app = QApplication(sys.argv)
mainWindow = Window2()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedHeight(450)
widget.setFixedWidth(500)
widget.show()
sys.exit(app.exec_())