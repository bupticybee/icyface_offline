import os
import sys
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size',type=int,help='pixel of size of image resized',default=64)
parser.add_argument('net_type',type=str,help='type of the net to use as classifier')
parser.add_argument('--batch_size',type=int,help='batch size',default=32)
parser.add_argument('--gpu_core',type=int,help='gpu core of chosen',default=0)
parser.add_argument('--hidden_size',type=int,help='hidden size of the network, not useful in vgg net',default=160)
parser.add_argument('--decay_step',type=int,help='decay used in Momentum',default=10)
parser.add_argument('--img_morethan',type=int,help='chose categories contains more images than this param',default=20)
parser.add_argument('--name',type=str,help='choose specific name for this model')
parser.add_argument('--L2',type=bool,help='whether to use L2 regulizer',default=True)
args = parser.parse_args(sys.argv[1:])

from align.facenet import get_dataset
from sklearn.cross_validation import train_test_split
import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers import *
from tflearn.models import *
import tensorflow as tf
from tflearn.metrics import top_k_op

INPUT_DATADIR = 'data/CAS_aled_cs_1expand-align/'
TEST_SIZE = 0.1
IMG_SIZE = args.image_size
NET_TYPE= args.net_type
BATCH_SIZE = args.batch_size
GPU_CORE = args.gpu_core
HIDDEN = args.hidden_size
DECAY_STEP=args.decay_step
DATA_MORETHAN = args.img_morethan
USE_L2 = args.L2
print(args.image_size)
creat_dirs = ['data/textlabel/','models']
TRAIN_TXT = 'data/textlabel/train_align_{}.txt'.format(DATA_MORETHAN)
TEST_TXT = 'data/textlabel/test_align_{}.txt'.format(DATA_MORETHAN)
ID_DIR_TXT = 'data/textlabel/id_dir_align.txt'
MODEL_FILE = "{}net_{}size_{}hidden_{}class".format(NET_TYPE,IMG_SIZE,HIDDEN,DATA_MORETHAN)
if args.name is not None:
	MODEL_FILE = "{}_{}".format(MODEL_FILE,args.name)
RUN_ID = MODEL_FILE
assert(NET_TYPE in ['resnet34','resnet50','vgg16','res-cp'])

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


train_x,train_y = image_preloader(TRAIN_TXT,(IMG_SIZE,IMG_SIZE))
test_x,test_y = image_preloader(TEST_TXT,(IMG_SIZE,IMG_SIZE))
LEN_OUT = len(train_y[0])

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if NET_TYPE == 'resnet34':
	with tf.device("/gpu:{}".format(GPU_CORE)):
		net = tflearn.input_data(shape=[None,IMG_SIZE,IMG_SIZE,3])
		if USE_L2:
			net = tflearn.conv_2d(net,64,7,2, regularizer='L2', weight_decay=0.0001)
		else:
			net = tflearn.conv_2d(net,64,7,2)
		# [91,91,64]
		net = tflearn.max_pool_2d(net,3,2)
		# [46,46,64]
		net = tflearn.residual_block(net,3,64)
		# [46,46,64]
		net = tflearn.residual_block(net,1,128,downsample=True)
		# [23,23,128]
		net = tflearn.residual_block(net,3,128)
		# [23,23,128]
		net = tflearn.residual_block(net,1,256,downsample=True)
		# [12,12,256]
		net = tflearn.residual_block(net,5,256)
		# [12,12,256]
		net = tflearn.residual_block(net,1,512,downsample=True)
		# [6,6,512]
		net = tflearn.residual_block(net,2,512)
		# [6,6,512]
		net = tflearn.global_avg_pool(net)
		# [512]
		fully_connect = tflearn.fully_connected(net,HIDDEN,activation='relu')
		# [1000]
		net = tflearn.fully_connected(fully_connect,LEN_OUT,activation='softmax')
		# [7211]
		#mom = tflearn.SGD(0.1,lr_decay=0.1,decay_step=3086 * 20)
		mom = tflearn.Momentum(0.01,lr_decay=0.1,decay_step=int(395000 / BATCH_SIZE) * 10,staircase=True)
		reg = tflearn.regression(net,optimizer=mom,loss='categorical_crossentropy')
		model = tflearn.DNN(reg,checkpoint_path='models/{}'.format(MODEL_FILE),max_checkpoints=100,session=sess)
elif NET_TYPE == 'resnet50':
	with tf.device("/gpu:{}".format(GPU_CORE)):
		net = tflearn.input_data(shape=[None,IMG_SIZE,IMG_SIZE,3])
		net = tflearn.conv_2d(net,32,4,1, regularizer='L2', weight_decay=0.0001)
		# [64,64,32]
		net = tflearn.max_pool_2d(net,3,2)
		# [32,32,32]
		net = tflearn.residual_bottleneck(net,3,32,128)
		# [32,32,64]
		net = tflearn.residual_bottleneck(net,1,64,256,downsample=True)
		# [16,16,128]
		net = tflearn.residual_bottleneck(net,3,64,256)
		# [16,16,128]
		net = tflearn.residual_bottleneck(net,1,128,512,downsample=True)
		# [16,16,256]
		net = tflearn.residual_bottleneck(net,5,128,512)
		# [16,16,256]
		net = tflearn.residual_bottleneck(net,1,256,1024,downsample=True)
		# [6,6,512]
		net = tflearn.residual_bottleneck(net,2,256,1024)
		# [6,6,512]
		net = tflearn.global_avg_pool(net)
		# [512]
		fully_connect = tflearn.fully_connected(net,HIDDEN,activation='relu')
		# [1000]
		net = tflearn.fully_connected(fully_connect,LEN_OUT,activation='softmax')
		# [7211]
		#mom = tflearn.SGD(0.1,lr_decay=0.1,decay_step=3086 * 20)
		mom = tflearn.Momentum(0.01,lr_decay=0.1,decay_step=int(395000 / BATCH_SIZE) * DECAY_STEP,staircase=True)
		reg = tflearn.regression(net,optimizer=mom,loss='categorical_crossentropy')
		model = tflearn.DNN(reg,checkpoint_path='models/{}'.format(MODEL_FILE),max_checkpoints=100,session=sess)
elif NET_TYPE == 'res-cp':
	with tf.device("/gpu:{}".format(GPU_CORE)):
		input_layer = tflearn.input_data([None, IMG_SIZE, IMG_SIZE, 3])
		net = tflearn.conv_2d(input_layer, 64, 7, strides=2)
		net = tflearn.max_pool_2d(net, 3, strides=2)
		net = tflearn.residual_block(net, 3, 64)
		net = tflearn.residual_block(net, 1, 128, downsample=True)
		net = tflearn.residual_block(net, 3, 128)
		net = tflearn.residual_block(net, 1, 256, downsample=True)
		net = tflearn.residual_block(net, 5, 256)
		net = tflearn.residual_block(net, 1, 512, downsample=True)
		net = tflearn.residual_block(net, 2, 512)
		net = tflearn.global_avg_pool(net)
		fully_connected = tflearn.fully_connected(net, HIDDEN, activation="relu")
		result = tflearn.fully_connected(fully_connected, LEN_OUT, activation='softmax')
		mom = tflearn.Momentum(0.01, lr_decay=0.1, decay_step=int(395000 / BATCH_SIZE) * DECAY_STEP, staircase=True)
		net = tflearn.regression(result, optimizer=mom, loss="categorical_crossentropy")
		model = tflearn.DNN(net,checkpoint_path='models/{}'.format(MODEL_FILE),session=sess,max_checkpoints=100,
                        tensorboard_verbose=0)
	
elif NET_TYPE == 'vgg16':
	with tf.device("/gpu:{}".format(GPU_CORE)):
		import tflearn
		from models import vgg16
		input_data = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3]) 
		hidden = vgg16(input_data)
		softmax = tflearn.fully_connected(hidden, LEN_OUT, activation='softmax', scope='fc8',restore=False)
		#momem = tflearn.optimizers.SGD(learning_rate=0.01)#ecay_step=200,lr_decay=0.1)
		momem = tflearn.Momentum(0.01)
		net = tflearn.regression(softmax, optimizer=momem,
                         loss='categorical_crossentropy')
		model = tflearn.DNN(net, checkpoint_path='models/{}'.format(MODEL_FILE),session=sess,max_checkpoints=100,tensorboard_verbose=0)
		



sess.run(tf.global_variables_initializer())
model.fit(train_x,train_y,n_epoch=200,validation_set=(test_x,test_y),snapshot_epoch=True,shuffle=True,batch_size=BATCH_SIZE,run_id=RUN_ID,show_metric=True)


