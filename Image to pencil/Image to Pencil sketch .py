import cv2

image=cv2.imread("dog.jpg")
cv2.imshow("Normal Image",image)
cv2.waitKey(0)


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image",gray)
cv2.waitKey(0)


invers=255-gray
cv2.imshow("Inverse GrayScale Image",invers)
cv2.waitKey(0)

blur=cv2.GaussianBlur(invers,(21,21),0)
inver_blur=255-blur

pencil=cv2.divide(gray,inver_blur,scale=256.0)
cv2.imshow("Normal Image",image)
cv2.imshow("Pencil sketch",pencil)
cv2.waitKey(0)

