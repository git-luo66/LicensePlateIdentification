# LicensePlateIdentification
本模型的数据为一个开源的车牌生成器生成车牌，车牌基本都有倾斜、加噪处理
本案例属于单样本，多标签问题，七个位置，七个模型
体征提取部分，卷积层共享，构造七个不同的分类网络，逼迫其学习检测不同的位置