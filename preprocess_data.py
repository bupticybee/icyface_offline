from scipy import misc
from align import facenet
import logging
import os
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from align.boxing import *
from align.landmarks import DlibLandmarkPredictor
from align.segment import CascadeSegmenter
from utils import *

cascade_box = CascadeBoxing()
center_box = CenterBlindBoxing()
landmark_predictor = DlibLandmarkPredictor()
cascade_segmenter = CascadeSegmenter(prefix='cs')

INPUT_DIR = 'data/CASIA-WebFace/'
OUTPUT_PREFIX = 'data/CAS_aled_'
dataset = get_dataset(paths=INPUT_DIR)

total_num = sum(len(i.image_paths) for i in dataset)

pb = ProgressBar(worksum=total_num,info="processing...")
pb.startjob()


dircreated = {}
all_num = 0
uni_num = 0
for one in dataset:
	foldername = one.name
	for path in one.image_paths:
		all_num += 1
		one_img = misc.imread(path)
		if one_img.ndim < 2:
			pb.complete(1)
			continue
		elif one_img.ndim == 2:
			one_img = facenet.to_rgb(one_img)
			 
		box = cascade_box.get_facebox(one_img)
		if box is None:
			uni_num += 1
			box = center_box.get_facebox(one_img)
		landmarks = landmark_predictor.get_landmarks(one_img,box=box)
		segmented = cascade_segmenter.segment(one_img,box=box,landmarks=landmarks)
		for one_seg in segmented:
			name,pic = one_seg
			dirname = "{}{}/{}".format(OUTPUT_PREFIX,name,foldername)
			if dirname not in dircreated:
				dircreated[dirname] = 1
				if not os.path.exists(dirname):
					os.makedirs(dirname)
			picname = "{}/{}".format(dirname,path.split('/')[-1])
			if not os.path.exists(picname):
				misc.imsave(picname,pic)
	pb.info="uni rate: {}%,{}  -- pp".format(round(float(uni_num) * 100 / all_num,2),uni_num)
	pb.complete(len(one.image_paths))
