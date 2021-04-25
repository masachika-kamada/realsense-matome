import cv2

img = cv2.imread("image.jpg")
dst = cv2.resize(img, (640, 480))

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("dst.jpg", dst)
