import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        loadUi("v1.ui", self)

        '''
        change here
        '''
        self.submit1.clicked.connect(self.press_submit)
        self.submit2.clicked.connect(self.press_submit2)
        self.changebutton.clicked.connect(self.press_submit)
        self.changebutton_2.clicked.connect(self.press_submit2)

    def update(self):
        self.textResult.adjustSize()

    def search1(self):
        category = self.comboBox_c.currentText()
        season = self.comboBox_s.currentText()
        gender = self.comboBox_g.currentText()

        print(category, season, gender)

        filtered_data = train_data
        if category != "All":
            filtered_data = filtered_data.loc[filtered_data["category"] == category,]

        if season != "All":
            filtered_data = filtered_data.loc[filtered_data["season"] == season,]

        if gender != "All":
            filtered_data = filtered_data.loc[filtered_data["gender"] == gender,]

        # image_num, image_id_list = filter_data_by_choice(train_data, category, season, gender)

        image_num = len(filtered_data)
        print(image_num)
        self.textResult.setText("We got %d images for you!"%image_num)
        self.update()

        image_list = list(filtered_data['id'])
        return image_num, image_list

    def press_submit(self):
        image_num, image_list = self.search1()
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

    def text_classifying(self):
        testtext = self.text_description.toPlainText()
        cleaned_text = clean_text(testtext)
        X = tokenizer.texts_to_sequences([cleaned_text])
        X = pad_sequences(X, maxlen=20)
        test_pred = model_lstm.predict(X)
        pred_label = np.argmax(test_pred)
        pred_catogory = category_mapping[int(pred_label)]

        pred_season = "All"
        pred_gender = "All"

        if ("men" in  cleaned_text) or ("man" in  cleaned_text) or ("boy" in  cleaned_text):
            pred_gender = "Men"
        if ("women" in  cleaned_text) or ("woman" in  cleaned_text) or ("girl" in  cleaned_text):
            pred_gender = "Women"

        if "summer" in cleaned_text:
            pred_season = "Summer"
        if "fall" in cleaned_text:
            pred_season = "Fall"
        if "winter" in cleaned_text:
            pred_season = "Winter"
        if "spring" in cleaned_text:
            pred_season = "Spring"
        return pred_catogory, pred_season, pred_gender

    def search2(self):
        category, season, gender = self.text_classifying()

        filtered_data = train_data
        if category != "All":
            filtered_data = filtered_data.loc[filtered_data["category"] == category,]

        if season != "All":
            filtered_data = filtered_data.loc[filtered_data["season"] == season,]

        if gender != "All":
            filtered_data = filtered_data.loc[filtered_data["gender"] == gender,]


        image_num = len(filtered_data)
        print(image_num)
        self.textResult.setText("Prediction Result:\n  Category: " + category +
                                "\n  Season: " + season +
                                "\n  gender: " + gender +
                                "\n We got %d images for you!"%image_num)
        self.update()
        image_list = list(filtered_data['id'])
        return image_num, image_list

    def press_submit2(self):
        image_num, image_list = self.search2()
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


if __name__ == "__main__":
    import sys
    import random
    import pandas as pd
    import pickle
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    import xgboost as xgb
    import re
    from nltk.corpus import stopwords
    import keras.preprocessing.text
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    import keras
    # from keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential, load_model

    model_lstm = load_model('../model/best_model_lstm.h5')

    with open('../tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    def clean_text(text):
        """
            text: a string

            return: modified initial string
        """
        text = text.lower()  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ',
                                       text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = BAD_SYMBOLS_RE.sub('',
                                  text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.

        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
        return text


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


    train_data = pd.read_csv("../data/train.csv")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QStackedWidget()
    ui = Ui_MainWindow()
    MainWindow.addWidget(ui)
    MainWindow.setFixedHeight(500)
    MainWindow.setFixedWidth(600)
    MainWindow.show()
    sys.exit(app.exec_())

