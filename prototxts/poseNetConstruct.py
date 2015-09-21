import argparse, sys
sys.path.append("../external/caffe/python")
from caffe import netSpec as NS
from caffe.proto import caffe_pb2
from collections import OrderedDict
from caffe import netBuildUtils as nbUtils
numClasses = 21
import numpy as np
import os, errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def poseNet(genMode, \
    learn_fc = 1, learn_conv = 1, \
    quat_loss = 0, softmax_loss = 1, vgg = 1, azimuth_net = 0, class_agnostic = 1, \
    warp = 1, cpad = 16, flip = 1, batch_size = 20, crop_size = 224, \
            window_file_folder = '/data1/shubhtuls/code/poseInduction/cachedir/rcnnFinetuneVps/binnedJoint/'):

    net = NS.NetSpec()
    if warp:
        crop_mode = "warp"
    else:
        crop_mode = "square"

    dataOut = ['data','label'] #output layers from data_layer
    poseOut = ['e1','e2','e3','e1c','e2c','e3c']
    dataOut = dataOut + poseOut
    if quat_loss:
        dataOut = dataOut + ['quat']

    if (genMode == 0):
        net.addInputBlobs(OrderedDict([
            ('data',[batch_size,3,crop_size,crop_size]),
            ('label',[batch_size, 1])
            ]))
    elif (genMode == 1):
        transform_param = dict(crop_size=crop_size, mirror=flip)
        window_data_param = dict(source=window_file_folder + 'Train.txt', batch_size = batch_size, context_pad = 16, crop_mode = crop_mode, fg_fraction = 1.0, fg_threshold = 0.5, bg_threshold = 0.5)
        net.addLayer('data', 'WindowPoseData', [], dataOut,
            transform_param = transform_param,
            window_data_param = window_data_param)
    elif (genMode == 2):
        transform_param = dict(crop_size=crop_size, mirror=flip)
        window_data_param = dict(source=window_file_folder + 'Val.txt', batch_size = batch_size, context_pad = 16, crop_mode = crop_mode, fg_fraction = 1.0, fg_threshold = 0.5, bg_threshold = 0.5)
        net.addLayer('data', 'WindowPoseData', [], dataOut,
            transform_param = transform_param,
            window_data_param = window_data_param)

    if(vgg):
        conv5_name = nbUtils.addVggTillConv(net, learn_conv, img_blob_name = 'data', suffix='')
    else:
        conv5_name = nbUtils.addAlexnetTillConv(net, learn_conv, img_blob_name = 'data', suffix='')

    top_name = nbUtils.addAlexnetFcLayers(net, learn_fc, inputBlobName = conv5_name, suffix='')
    fc7 = top_name

    if(quat_loss):
        if(class_agnostic):
            net.addLayer('quatMask', 'InnerProduct',fc7, 'quatMask',
                num_output = 4, weight_filler={'type':"gaussian",'std':0.01},
                bias_filler={'type':"constant", 'value':0})
        else:
            net.addLayer('quatPred', 'InnerProduct',fc7, 'quatPred',
                num_output = 4*numClasses, weight_filler={'type':"gaussian",'std':0.01},
                bias_filler={'type':"constant", 'value':0})
            net.addLayer('quatMask', 'MaskOutputs', ['quatPred','label'], 'quatMask', convolution_param = dict(kernel_size = 4))

        net.addLayer('quatNorm', 'QuatNormalization', 'quatMask', 'quatNorm')
        if(genMode > 0):
            net.addLayer('quatLoss', 'EuclideanAntipodalLoss', ['quatNorm','quat'], 'quatLoss')

    if(softmax_loss):
        if(not azimuth_net):
            poseLayerSizes = dict(e1 = 21, e2 = 21, e3 = 21, e1c = 7, e2c = 7, e3c = 7)
        else:
            poseLayerSizes = dict(e1 = 24, e2 = 16, e3 = 8, e1c = 4, e2c = 4, e3c = 4)

        for pl in poseOut:
            if(class_agnostic):
                kernel = poseLayerSizes[pl]
                net.addLayer(pl + 'Mask', 'InnerProduct',fc7, pl + 'Mask',
                    num_output = kernel, weight_filler={'type':"gaussian",'std':0.01},
                    bias_filler={'type':"constant", 'value':0})
            else:
                kernel = poseLayerSizes[pl]
                net.addLayer(pl + 'Pred', 'InnerProduct',fc7, pl + 'Pred',
                    num_output = kernel*numClasses, weight_filler={'type':"gaussian",'std':0.01},
                    bias_filler={'type':"constant", 'value':0})
                net.addLayer(pl + 'Mask', 'MaskOutputs', [pl + 'Pred','label'], pl + 'Mask', convolution_param = dict(kernel_size = kernel))
            if(genMode > 0):
                net.addLayer(pl + 'Loss', 'SoftmaxWithLoss', [pl + 'Mask',pl], pl + 'Loss')
                if(genMode == 2):
                    net.addLayer(pl + 'Accuracy', 'Accuracy', [pl + 'Mask',pl], pl + 'Accuracy')

        if(genMode == 0):
            net.addLayer('poseClassify','Concat',[pl + 'Mask' for pl in poseOut],'poseClassify')
    else:
        if(genMode > 0):
            net.addLayer('junkYardPose','Silence',poseOut,[])

    if(class_agnostic):
        net.addLayer('junkYardLabel','Silence','label' ,[])
    return net.toProto()

def constructNet(folder,\
	filename = 'train.prototxt', valfilename = 'test.prototxt', testfilename = 'deploy.prototxt', \
	learn_fc = 1, learn_conv = 1, \
	quat_loss = 0, softmax_loss = 1, vgg = 1, azimuth_net = 0, class_agnostic = 1, \
	warp = 1, cpad = 16, flip = 1, batch_size = 20, crop_size = 224, \
	window_file_folder = '/data1/shubhtuls/code/poseInduction/cachedir/rcnnFinetuneVps/binnedJoint/'):

	## trainNet
	trainFile = folder + filename
	with open(trainFile, 'w') as f:
		net = poseNet(1, \
			learn_fc = learn_fc, learn_conv = learn_conv, \
			quat_loss = quat_loss, softmax_loss = softmax_loss, vgg = vgg, azimuth_net = azimuth_net, class_agnostic = class_agnostic, \
			warp = warp, cpad = cpad, flip = flip, batch_size = batch_size, crop_size = crop_size, \
			window_file_folder = window_file_folder);
		print >> f, net

	valFile = folder + valfilename
	with open(valFile, 'w') as f:
		net = poseNet(2, \
			learn_fc = learn_fc, learn_conv = learn_conv, \
			quat_loss = quat_loss, softmax_loss = softmax_loss, vgg = vgg, azimuth_net = azimuth_net, class_agnostic = class_agnostic, \
			warp = warp, cpad = cpad, flip = flip, batch_size = batch_size, crop_size = crop_size, \
			window_file_folder = window_file_folder);
		print >> f, net


	## testNet
	testFile = folder + testfilename
	with open(testFile, 'w') as f:
		net = poseNet(0, \
			learn_fc = learn_fc, learn_conv = learn_conv, \
			quat_loss = quat_loss, softmax_loss = softmax_loss, vgg = vgg, azimuth_net = azimuth_net, class_agnostic = class_agnostic, \
			warp = warp, cpad = cpad, flip = flip, batch_size = batch_size, crop_size = crop_size, \
			window_file_folder = window_file_folder);
		print >> f, net

def constructSolver(folder,\
	filename = 'train.prototxt', valfilename = 'test.prototxt',
	max_iter = 70000, snapshot = 10000, test_iter = 100, test_interval = 1000, \
	base_lr = 0.001, lr_policy= "step", gamma = 0.1 ,stepsize = 20000 , momentum = 0.9 ,weight_decay = 0.0005, \
	display= 200, \
	snapshot_folder = '/work5/shubhtuls/snapshots/instancePose/', snapshot_subdir = '', snapshot_prefix = 'net'):

	mkdir_p(snapshot_folder + snapshot_subdir)
	prefix = snapshot_folder + snapshot_subdir + snapshot_prefix
	solverFile = folder + 'solver.prototxt'
	with open(solverFile, 'w') as f:
		f.write("train_net : \"%s\"\n" %os.path.abspath(folder + filename) )
		f.write("test_net : \"%s\"\n" %os.path.abspath(folder + valfilename) )
		f.write("max_iter : %d\n" %max_iter)
		f.write("snapshot : %d\n" %snapshot)
		f.write("test_iter : %d\n" %test_iter)
		f.write("test_interval : %d\n" %test_interval)
		f.write("base_lr : %f\n" %base_lr)
		f.write("lr_policy : \"%s\"\n" %lr_policy)
		f.write("gamma : %f\n" %gamma)
		f.write("stepsize : %d\n" %stepsize)
		f.write("momentum : %f\n" %momentum)
		f.write("weight_decay : %f\n" %weight_decay)
		f.write("display : %d\n" %display)
		f.write("snapshot_prefix : \"%s\"\n" %prefix)

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a pose prediction network')

	## prototxt output parameters
	parser.add_argument('--folder', help='where to output the prototxt file',
		default=None, type=str)
	parser.add_argument('--filename', help='where to output the file',
                        default="train.prototxt", type=str)
	parser.add_argument('--valfilename', help='where to output the file',
                        default="test.prototxt", type=str)
	parser.add_argument('--testfilename', help='where to print test prototxt',
                         default="deploy.prototxt", type=str)

	## Learning parameters
	parser.add_argument('--learn_fc', default=1, type=float)
	parser.add_argument('--learn_conv', default=1, type=float)

	## Net Architecture
	parser.add_argument('--quat_loss', default=0, type=float)
	parser.add_argument('--softmax_loss', default=1, type=float)
	parser.add_argument('--vgg', default=1, type=float, help='whether to use VGG net or Alexnet')
	parser.add_argument('--azimuth_net', default = 0, type = float, help = 'whether to use joint or azimuth net')
	parser.add_argument('--class_agnostic', default = 1, type = float, help = 'whether to use same fc8 units across classes')

	## Input data parameters
	parser.add_argument('--warp', default=1, type=float, help='use warp or square crop')
	parser.add_argument('--cpad', default=16, type=float, help='input context paddding')
	parser.add_argument('--flip', default=1, type=float, help='whether to use VGG net or Alexnet')
	parser.add_argument('--batch_size', default=20, type=int, help='number of images in 1 minibatch')
	parser.add_argument('--crop_size', default=224, type=int, help='input size')
	parser.add_argument('--window_file_folder', default='/data1/shubhtuls/code/poseInduction/cachedir/rcnnFinetuneVps/binnedJoint/', type=str)

	## Solver parameters
	parser.add_argument('--max_iter', default=70000, type=int)
	parser.add_argument('--snapshot', default=10000, type=int)
	parser.add_argument('--test_iter', default=100, type=int)
	parser.add_argument('--test_interval', default=1000, type=int)
	parser.add_argument('--base_lr', default=0.001, type=float)
	parser.add_argument('--lr_policy', default='step')
	parser.add_argument('--gamma', default=0.1, type=float)
	parser.add_argument('--stepsize', default=20000, type=int)
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--weight_decay', default=0.005, type=float)
	parser.add_argument('--display', default=200, type=int)
	parser.add_argument('--snapshot_folder', default='/data1/shubhtuls/code/poseInduction/cachedir/snapshots/')
	parser.add_argument('--snapshot_subdir', default='')
	parser.add_argument('--snapshot_prefix', default='net')

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)
	assert(args.folder != None), 'output folder not specified'
	assert(args.snapshot_subdir != ''), 'snapshot subdirectory not specified'
	mkdir_p(args.folder)
	constructNet(args.folder,
		filename = args.filename,
		valfilename = args.valfilename,
		testfilename = args.testfilename,
		learn_fc = args.learn_fc, learn_conv = args.learn_conv,
		quat_loss = args.quat_loss, softmax_loss = args.softmax_loss, azimuth_net = args.azimuth_net, class_agnostic = args.class_agnostic,
		vgg = args.vgg,
		warp = args.warp, cpad = args.cpad,
		flip = args.flip, batch_size = args.batch_size, crop_size = args.crop_size,
		window_file_folder = args.window_file_folder)

	constructSolver(args.folder,
		filename = args.filename, valfilename = args.valfilename,
		max_iter = args.max_iter, snapshot = args.snapshot, test_iter = args.test_iter, test_interval = args.test_interval,
		base_lr = args.base_lr, lr_policy= args.lr_policy, gamma = args.gamma ,stepsize = args.stepsize , momentum = args.momentum ,weight_decay = args.weight_decay,
		display= args.display,
		snapshot_folder = args.snapshot_folder, snapshot_subdir = args.snapshot_subdir, snapshot_prefix = args.snapshot_prefix)
