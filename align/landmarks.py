import align.align_dlib
import numpy as np
import dlib

class LandmarkPredictor:
	def __init__(self):
		pass
	def get_landmarks(self,img):
		"return some landmarks "
		"retval int: [(x,y),...]"
		pass

class DlibLandmarkPredictor(LandmarkPredictor):
	def __init__(self,model_file='data/shape_predictor_68_face_landmarks.dat'):
		self.align_tool = align.align_dlib.AlignDlib(model_file)
		self.landmarkIndices = align.align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
	def get_landmarks(self,img,box=None,left=None,top=None,right=None,bottom=None):
		if box is not None:
			left,top,right,bottom = box
		left = np.long(left)
		top = np.long(top)
		right = np.long(right)
		bottom = np.long(bottom)
		bb = dlib.rectangle(left,top,right,bottom)
		landmarks = self.align_tool.findLandmarks(img,bb)
		npLandmarks = np.float32(landmarks)
		npLandmarkIndices = np.array(self.landmarkIndices)
		return npLandmarks[npLandmarkIndices]
