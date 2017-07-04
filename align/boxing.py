import align.align_dlib
import align.detect_face
from scipy import misc
import tensorflow as tf
import align.align_dlib
import cv2


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

class DlibBoxing(Boxing):
	def __init__(self,model_dir='data/shape_predictor_68_face_landmarks.dat'):
		super(DlibBoxing,self).__init__()
		align_tool = align.align_dlib.AlignDlib('data/shape_predictor_68_face_landmarks.dat')
		self.align_tool = align_tool
	def get_multibox(self,img):
		dlib_bounders = self.align_tool.getAllFaceBoundingBoxes(img)
		return [(i.left(),i.top(),i.right(),i.bottom()) for i in dlib_bounders]


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
		self.dlbox = DlibBoxing()
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

