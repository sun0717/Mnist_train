import cv2 as cv
# 1. 调整图像大小为新的尺寸
img = cv.imread('image.jpg')
resized_img = cv.resize(img, (28, 28))
# 2. 将图像转换为灰度图像：
# reshaped_image = img.reshape((1, 1, 28, 28))
gray_image = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
# 3. 调整图像的形状以符合目标形状：
reshaped_image = gray_image.reshape((1, 1, 28, 28))
cv.imwrite('input.jpg', gray_image)