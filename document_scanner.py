from transform import transform
from skimage.filters import threshold_local
import numpy as np
import cv2

imgPath = "doc.jpg"
img = cv2.imread(imgPath)
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# detecting edges
edge = cv2.Canny(blur, 75, 200)

cv2.imshow("Original Image", img)
cv2.imshow("Edge Detected Image", edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

# finding contours
contours, _ = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for contour in contours:

	perimeter = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
	
	if len(approx) == 4:
		document = approx
		break
		
cv2.drawContours(img, [document], -1, (0, 255, 0), 2)

cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = transform(orig, document.reshape(4, 2))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# calculate a threshold mask
T = threshold_local(warped, 11, offset = 10, method = "gaussian")

warped = (warped > T).astype("uint8") * 255

cv2.imshow("Original", img)
cv2.imshow("Scanned", warped)

cv2.imwrite("output.jpg", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()	