import numpy as np
import cv2

def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s= np.sum(pts, axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
	d = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(d)]
	rect[3] = pts[np.argmax(d)]
	
	return rect
	
def transform(img, pts):
	
	rect = order_points(pts)
	(tl, tr, br, bl) = rect	
	
	width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))	
	width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(width_1), int(width_2))
	
	height_1 = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))	
	height_2 = np.sqrt(((bl[0] - tl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(height_1), int(height_2))
	
	dst = np.array([
	         [0, 0],
	         [maxWidth - 1, 0],
	         [maxWidth - 1, maxHeight - 1],
	         [0, maxHeight - 1]], dtype = "float32")
	         
	M = cv2.getPerspectiveTransform(rect, dst) 
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))      
	 
	return warped	    
	  
	      
	         	         	         	         	         
	