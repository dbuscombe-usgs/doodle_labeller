import cv2
img_path = "image.jpg"
img = cv2.imread(img_path)
cv2.imshow("image window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
