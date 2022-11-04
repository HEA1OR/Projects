import numpy as np

from sklearn import preprocessing

from skimage import io, transform

import matplotlib.pyplot as plt

import skimage

from skimage import color

import time

import os

# path = 'D:\\python\\Lib\\site-packages\\PyInstaller\\image'

# os.makedirs(path, 0o700)

img0 = io.imread('D:\\丁真.jpeg')

# print('img-1彩色图像的形状为：',img0.shape)
img0 = skimage.color.rgb2gray(img0)
# img_gray = skimage.color.rgb2gray(transform.rescale(img0, 0.3, True))   #压缩比例
img0 = transform.rescale(img0, 0.2)


#  print('img_gray灰度图像的形状为：',img_gray.shape)


#二值化灰度图像数据，threshold为二值化灰度阈值

bina = preprocessing.Binarizer(threshold=0.50)

bina.fit(img0)


img_gray_binary = bina.transform(img0)

#  io.imshow(img_gray_binary)





scatter_idx = np.array(np.where(img_gray_binary == 0))

#绘制灰度值为-1的元素散点图

plt.figure(figsize=(10, 10), frameon=False)

plt.scatter(scatter_idx[1], scatter_idx[0], marker='.')#显示灰度图像

ax = plt.gca()

ax.invert_yaxis()#y轴反向

plt.axis('off')#关闭坐标轴

plt.show()
time.sleep(0.2)



print("悦刻五代是悦刻系列最新发布的一款电子烟产品,单杆为267元,渐变色为278元,")
print("目前采取的销售策略是杆弹分离，处于换弹小烟的中上价格区间。" )
print("悦刻五代的烟杆上设计了一个呼吸灯,可以清晰地看到电子烟的电量情况，以及查看充电进度，使用体验更好")

time.sleep(9)