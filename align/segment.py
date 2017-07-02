import cv2
import dlib
import numpy as np
import align.align_dlib
from align.align_dlib import TEMPLATE,MINMAX_TEMPLATE
from scipy import misc

class Segmenter:
	def __init__(self):
		pass
	def segment(self,img,box,landmarks):
		"return list of imgs segmented from the origin image"
		"retval 2darray : [img1,img2,...]"
		pass

class TwodAlignSegmenter(Segmenter):
	def __init__(self,landmarkIndices = align.align_dlib.AlignDlib.OUTER_EYES_AND_NOSE,imgDim=110,scale=0.78):
		super(TwodAlignSegmenter,self).__init__()
		self.landmarkIndices = landmarkIndices
		self.npLandmarkIndices = np.array(landmarkIndices)
		self.imgDim = imgDim
		self.scale = scale

	def segment(self,img,box,landmarks):
		left,top,right,bottom = box
		left,top,right,bottom = int(left),int(top),int(right),int(bottom)
		bb = dlib.rectangle(left,top,right,bottom)
		H = cv2.getAffineTransform(landmarks,
                               self.imgDim * MINMAX_TEMPLATE[self.npLandmarkIndices] * self.scale + self.imgDim * (1 - self.scale)/2)
		thumbnail = cv2.warpAffine(img, H, (self.imgDim, self.imgDim))
		return [('2d-align',thumbnail)]

class ExpandMarginSegmenter(Segmenter):
	def __init__(self,margin=0.5,img_size=182):
		"margin : int for pixels ,float for ratio" 
		super(ExpandMarginSegmenter,self).__init__()
		self.margin = margin
		self.img_size = img_size

	def segment(self,img,box,landmarks):
		left,top,right,bottom = box
		avglr = int((left + right) / 2)
		avgtb = int((top + bottom) / 2)
		r_cir = int(max((right - left),(bottom - top) )/ 2)
		if isinstance(self.margin,int):
			margin = self.margin
		elif isinstance(self.margin,float):
			margin = r_cir * self.margin
		bb = np.zeros(4, dtype=np.int32)
		bb[0] = np.maximum(left-margin/2, 0)
		bb[1] = np.maximum(top-margin/2, 0)
		bb[2] = np.minimum(right+margin/2, img.shape[1])
		bb[3] = np.minimum(bottom+margin/2, img.shape[0])
		cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
		scaled = misc.imresize(cropped, (self.img_size, self.img_size), interp='bilinear')
		return [('expand-align',scaled)]

class FacepartSegmenter(Segmenter):
	def __init__(self,margin=0.83):
		"margin by pixel or float as ratio"
		super(FacepartSegmenter,self).__init__()
		self.margin = margin
	
	def segment(self,img,box,landmarks):
		
		


