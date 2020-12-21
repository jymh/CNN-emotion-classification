# CNN-emotion-classification
数据由48x48像素的面部灰度图像组成。面部已自动对齐，一般ROC应该居中，并且在每个图像中脸占据大约相同的空间。任务是根据面部表情中显示的情感classify为以下七个类别之一（0 =愤怒，1 =厌恶，2 =恐惧，3 =快乐，4 =悲伤，5 =惊奇，6 =中性）。

train.csv包含两列，“emotion”和“pixels”。 “emotion”列包含图像中存在的情感的数字代码，范围从0到6。 “emotion”列包含每个图像用引号引起来的字符串。该字符串的内容以行主要顺序以空格分隔。 test.csv仅包含“pixels”列，你的任务是预测emotion列。

训练集包含28709。测试集包含7178。

train.py中有模仿AlexNet手动搭建的CNN，根据图片大小对参数做了一些调整，并添加了dropout解决一些过拟合的问题。Myresnet.py中有自己搭建的resnet18。

数据train.csv test.csv：链接：https://pan.baidu.com/s/17e1NrKuIxVwf57y7cMgMyg  提取码：1234 

