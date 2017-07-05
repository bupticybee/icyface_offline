import os
import sys
from align.facenet import get_dataset
from sklearn.cross_validation import train_test_split
import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers import *
from tflearn.models import *
import tensorflow as tf
from tflearn.metrics import top_k_op

INPUT_DATADIR = 'data/CAS_aled_cs_1expand-align/'
DATA_MORETHAN = 20
TEST_SIZE = 0.1
creat_dirs = ['data/textlabel/','models']
TRAIN_TXT = 'data/textlabel/train_align.txt'
TEST_TXT = 'data/textlabel/test_align.txt'
ID_DIR_TXT = 'data/textlabel/id_dir_align.txt'
MODEL_FILE = 'expend_align_face'
RUN_ID = MODEL_FILE

dataset = get_dataset(INPUT_DATADIR)
dset_part_softmax = [i for i in dataset if len(i) > DATA_MORETHAN]
print("softmax individual {} softmax pictures {}".format(len(dset_part_softmax),sum([len(i.image_paths) for i in dset_part_softmax  ])))

"make dirs for label"
for one in creat_dirs:
	if os.path.exists(one):
		continue
	os.mkdir(one)

labels = []
files = []
act_dirs = []
for num_ind,i in enumerate(dset_part_softmax):
	files += i.image_paths
	labels += [num_ind] * len(i.image_paths)
	act_dirs += [i] * len(i.image_paths)

train_labels,test_labels,train_files,test_files = train_test_split(labels,files,test_size=TEST_SIZE,random_state=0)

print("train saples: {} test samples {}".format(len(train_labels),len(test_labels)))

with open(TRAIN_TXT,'w') as whdl:
	for f,l in zip(train_files,train_labels):
		whdl.write("{} {}\n".format(f,l))

with open(TEST_TXT,'w') as whdl:
	for f,l in zip(test_files,test_labels):
		whdl.write("{} {}\n".format(f,l))

with open(ID_DIR_TXT,'w') as whdl:
	for i,j in zip(labels,act_dirs):
		whdl.write("{} {}".format(i,j))

train_x,train_y = image_preloader(TRAIN_TXT,(182,182))
test_x,test_y = image_preloader(TEST_TXT,(182,182))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.device("/gpu:1"):
	net = tflearn.input_data(shape=[None,182,182,3])
	net = tflearn.conv_2d(net,64,7,2)
	# [91,91,64]
	net = tflearn.max_pool_2d(net,3,2)
	# [46,46,64]
	net = tflearn.residual_block(net,2,64)
	# [46,46,64]
	net = tflearn.residual_block(net,1,128,downsample=True)
	# [23,23,128]
	net = tflearn.residual_block(net,1,128)
	# [23,23,128]
	net = tflearn.residual_block(net,1,256,downsample=True)
	# [12,12,256]
	net = tflearn.residual_block(net,1,256)
	# [12,12,256]
	net = tflearn.residual_block(net,1,512,downsample=True)
	# [6,6,512]
	net = tflearn.residual_block(net,1,512)
	# [6,6,512]
	net = tflearn.global_avg_pool(net)
	# [512]
	fully_connect = tflearn.fully_connected(net,1000,activation='relu')
	# [1000]
	net = tflearn.fully_connected(fully_connect,7211)
	# [7211]
	mom = tflearn.Momentum(0.01,lr_decay=0.1,decay_step=3086 * 20)
	reg = tflearn.regression(net,optimizer=mom,loss='categorical_crossentropy')
	model = tflearn.DNN(reg,checkpoint_path='models/{}'.format(MODEL_FILE),max_checkpoints=100,session=sess)

sess.run(tf.global_variables_initializer())
model.fit(train_x,train_y,n_epoch=200,validation_set=(test_x,test_y),snapshot_epoch=True,batch_size=128,run_id=RUN_ID,show_metric=True)


