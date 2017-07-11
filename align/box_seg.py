import align.detect_face
from scipy import misc
import tensorflow as tf
import cv2
import numpy as np
from scipy import misc


class Boxing(object):
	def __init__(self):
		pass
	def get_multibox(self,img):
		"return [(l,t,r,b)]"
		pass
	def get_facebox(self,img,skipMulti=False):
		"return the largest facebox in all boxes"
		"retval: int list (l,t,r,b)"
		boxes = self.get_multibox(img)
		if len(boxes) == 0:
			return None
		if skipMulti == True and len(boxes) > 1:
			return None
		boxes = sorted(boxes,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]),reverse=True)
		return boxes[0]

class MtcnnBoxing(Boxing):
	def __init__(self,sess=None,threshold = [ 0.6, 0.7, 0.7 ],factor = 0.709):
		super(MtcnnBoxing,self).__init__()
		if sess == None:
			config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
			config.gpu_options.allow_growth = True
			sess = tf.Session(config=config)
		self.sess = sess
		pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
		self.pnet = pnet
		self.rnet = rnet
		self.onet = onet
		self.threshold = threshold
		self.factor = factor

	def get_multibox(self,img):
		bounding_boxes, _ = align.detect_face.detect_face(img,250, self.pnet,self.rnet,self.onet,self.threshold,self.factor)
		return [(i[0],i[1],i[2],i[3]) for i in bounding_boxes]

class OpencvBoxing(Boxing):
	def __init__(self,scaleFactor=1.1,minNeighbors=5):
		super(OpencvBoxing,self).__init__()
		facecascade = cv2.CascadeClassifier('data/haar_cascade_frontalface_default.xml')
		self.facecascade = facecascade
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
	def get_multibox(self,img):
		faces = self.facecascade.detectMultiScale(img,scaleFactor=self.scaleFactor,minNeighbors=self.minNeighbors)
		return [(i[0],i[1],i[0] + i[2],i[1] + i[3])
			for i in faces]

class CenterBlindBoxing(Boxing):
	def __init__(self):
		super(CenterBlindBoxing,self).__init__()
	def get_multibox(self,img):
		return [(
			img.shape[1] * 0.25,
			img.shape[0] * 0.25,
			img.shape[1] * 0.75,
			img.shape[0] * 0.75,
		)]

class CascadeBoxing(Boxing):
	def __init__(self):
		super(CascadeBoxing,self).__init__()
		self.mtbox = MtcnnBoxing()
		self.cvbox = OpencvBoxing()
	def get_multibox(self,img):
		mtbox_result = self.mtbox.get_multibox(img)
		if len(mtbox_result) != 0:
			return mtbox_result
		dlbox_result = self.dlbox.get_multibox(img)
		if len(dlbox_result) != 0:
			return dlbox_result
		cvbox_result = self.cvbox.get_multibox(img)
		if len(cvbox_result) != 0:
			return cvbox_result
		return []

class Segmenter(object):
	def __init__(self):
		pass
	def segment(self,img,box,landmarks):
		"return list of imgs segmented from the origin image"
		"retval 2darray : [img1,img2,...]"
		pass

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

class CascadeSegmenter(Segmenter):
	def __init__(self,segmenters=[ExpandMarginSegmenter()],prefix='cascade_segmenter_'):
		"margin by other segmenters"
		super(CascadeSegmenter,self).__init__()
		self.__segmenters = segmenters
		self.__prefix = prefix
	
	def add_segmenter(self,segmenter):
		self.__segmenters.append(segmenter)
		
	def segment(self,img,box,landmarks):
		retval = []
		for ind,seg in enumerate(self.__segmenters):
			retval += seg.segment(img,box,landmarks)
			retval[-1] = ("{}_{}".format(self.__prefix,ind) + retval[-1][0],retval[-1][1])
		return retval

