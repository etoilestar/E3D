from torch import nn, optim
import torch
#import C3D_model
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
from lib import resnet
from config import params
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
save_path = r'/data/zhengrui/dataset/camimg'
image_path = r'/data/zhengrui/dataset/ucf101/valid'
pre_model = '/data/zhengrui/dataset/pretrain/ef0_rgb1.pth.tar'
i=0
resnets = EfficientNet.from_name(params['pretrained'], data='rgb',override_params={'num_classes': 101})
#resnets = resnet.resnet50(num_classes=101)
pretrained_dict = torch.load(pre_model, map_location='cpu')
model_dict = resnets.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if '_fc1' in k}
print("fc:"+str(pretrained_dict))
model_dict.update(pretrained_dict)
resnets.load_state_dict(model_dict)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#resnets = resnets.cuda(params['gpu'][0])
#resnets.to(device)
#resnet = C3D_model.C3Dnet()#这里单独加载一个包含全连接层的resnet50模型
image = []
class FeatureExtractor():
	""" Class for extracting activations and
	registering gradients from targetted intermediate layers """
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x, name0):
		outputs = []
		self.gradients = []
		f = open('importance.txt','a')
		for name, module in self.model._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
			print('name=',name)
			print('module=',module)
			print(str(type(module)))
			if 'ModuleList' in str(type(module)):
				for num in range(len(module)):
					x = module[num].forward(x)
			else:
				x = module.forward(x)
		#            print('name=',name)
#			print('x.size()=',x.size())
			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
#		x1 = x.cpu().detach().numpy().reshape(20480)
#		x1 =x1.tolist()
#		f.write(name0+'     ')
#		Inf = 0
#		temp = []
#		for i in range(10):
		#            print(x1)
#			temp.append(x1.index(max(x1)))
#			x1[x1.index(max(x1))]=Inf
#		temp.sort()
#		f.write(str(temp))
#		print(temp)
#		f.write('\n')
#		f.close()
		return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x, name):
		target_activations, output  = self.feature_extractor(x, name)
		output = output.squeeze()
#		output = output.view(output.size(0), -1)
#		print('classfier=',output.size())
		if self.cuda:
			output = output.cpu()
			# output = C3D_model.C3Dnet().classifier(output).cuda()##这里就是为什么我们多加载一个resnet模型进来的原因，因为后面我们命名的model不包含fc层，但是这里又偏偏要使用。
#			output = resnets.avgpool(output).cuda()
			output = F.adaptive_avg_pool2d(output, 1).squeeze(-1).squeeze(-1)
			output = resnets._fc1(output).cuda()
		else:
#			output = resnets.avgpool(output)
			output = resnets.fc(output)
			# output = C3D_model.C3Dnet().classifier(output)##这里对应use-cuda上更正一些bug,不然用use-cuda的时候会导致类型对不上,这样保证既可以在cpu上运行,gpu上运行也不会出问题.
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	# preprocessed_img = torch.from_numpy(preprocessed_img)
	# preprocessed_img.unsqueeze_(0)
	# input = torch.Tensor(preprocessed_img)
	input = preprocessed_img
	return input

def show_cam_on_image(img, mask,name, val_path, vname):
	# mask = mask.transpose((1, 0, 2, 3))
	if not os.path.exists(os.path.join(save_path, val_path, vname)):
		os.makedirs(os.path.join(save_path, val_path, vname))
	for i in range(16):
		heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		cam = heatmap + np.float32(img[i, :, :, :])
		cam = cam / np.max(cam)
		cv2.imwrite(os.path.join(save_path, val_path, vname, vname+"cam_{}.jpg".format(i)), np.uint8(255 * cam))
		print(os.path.join(save_path, val_path, vname, vname+"cam_{}.jpg".format(i)))
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, name, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda(),name)
		else:
			features, output = self.extractor(input,name)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）
		x = self.extractor.get_gradients()
		print(x)
		y = x[-1]
		grads_val = y.cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :, :]

		cam = np.maximum(cam, 0)
		cam = cam.reshape((7, 7))
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		# cam = cam.reshape((1,224,224))
		# cam1 = cam
		# for _ in range(16):
		# 	cam1 = np.concatenate((cam1, cam))
		return cam

class GuidedBackpropReLU(Function):

	def forward(self, input):
		positive_mask = (input > 0).type_as(input)
		output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
		self.save_for_backward(input, output)
		return output

	def backward(self, grad_output):
		input, output = self.saved_tensors
		grad_input = None

		positive_mask_1 = (input > 0).type_as(grad_output)
		positive_mask_2 = (grad_output > 0).type_as(grad_output)
		grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

		return grad_input
	
class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = resnet#这里同理，要的是一个完整的网络，不然最后维度会不匹配。
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

			# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

			# self.model.features.zero_grad()
			# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=True,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image_path', type=str, default=r'F:\dataset\train\ApplyEyeMakeup',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args

def makeclip(path):
	capture = cv2.VideoCapture(path)
	retaining, frame = capture.read()
	frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	start = 0
	end = start + 32
	buffer = np.empty((32, 224, 224, 3), np.dtype('float32'))
	sample_count = 0

	for j in range(end):
		retaining, frame = capture.read()
		if retaining is False:
			capture = cv2.VideoCapture(path)
			retaining, frame = capture.read()
#            print('retain False')
		if retaining:
			if j >= start:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
				frame = frame[int(frame_height/2-224/2):int(frame_height/2-224/2)+224, int(frame_width/2-224/2):int(frame_width/2-224/2)+224, :]
				buffer[sample_count] = frame
				sample_count = sample_count + 1
	capture.release()
	return buffer


if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()
	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
#	model = C3D_model.C3Dnet()
	# model = models.resnet50(pretrained=False)#这里相对vgg19而言我们处理的不一样，这里需要删除fc层，因为后面model用到的时候会用不到fc层，只查到fc层之前的所有层数。
	# del model.fc
#	model = resnet.resnet50(num_classes=101)
	model = EfficientNet.from_name(params['pretrained'], data='rgb',override_params={'num_classes': 101})
	del model._fc1
	pretrained_dict = torch.load(pre_model, map_location='cpu')
	model_dict = model.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.cuda(params['gpu'][0])
#	if torch.cuda.device_count() > 1:
#		model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu
#	model.to(device)

#	print(model)
	#modules = list(resnet.children())[:-1]
	#model = torch.nn.Sequential(*modules)
	print('OK1')
	#print(model) layer4
#	grad_cam = GradCam(model , \
#					target_layer_names = ["layer4"], use_cuda=args.use_cuda)##这里改成layer4也很简单，我把每层name和size都打印出来了，想看哪层自己直接嵌套就可以了。（最后你会在终端看得到name的）
	grad_cam = GradCam(model , \
                                        target_layer_names = ["_conv_head"], use_cuda=args.use_cuda)
	print('OK2')
	for val_path in os.listdir(image_path):
		for root,dirs,filename in os.walk(os.path.join(image_path,val_path)):
			print(filename)
			for s in filename:
				input = []
				image = []
				vname, ext = os.path.splitext(s)
				clip = makeclip(os.path.join(image_path, val_path, s))
		#		if not os.path.exists(os.path.join(save_path, vname)):
	#				os.mkdir(os.path.join(save_path, vname))
				for i in range(32):
					img = clip[i, :, :, :]
			# for s in filename:
			# 	clip = makeclip(os.path.join(args.image_path, s))
			# 	image.append(cv2.imread(args.image_path+s,1))
			#img = cv2.imread(filename, 1)
					img = np.float32((cv2.resize(img, (224, 224))-128.0)/128.0)
					image.append(img)
					input.append(preprocess_image(img))
				input = np.array(input)
				input = input.transpose((1, 0, 2, 3))
				input = torch.from_numpy(input)
				input.unsqueeze_(0)
				input = torch.Tensor(input)
				image = np.array(image)
	#			image = image.transpose((1, 0, 2, 3))
				print('input.size()=',input.size())
				target_index =None
				mask = grad_cam(input, vname, target_index)
#				print(image.shape)
#				print(mask.shape)
			#print(type(mask))
			show_cam_on_image(image, mask, i, val_path, vname)

				#gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
				#gb = gb_model(input, index=target_index)
				#utils.save_image(torch.from_numpy(gb), 'gb.jpg')

				#cam_mask = np.zeros(gb.shape)
				#for i in range(0, gb.shape[0]):
					#	cam_mask[i, :, :] = mask

				#cam_gb = np.multiply(cam_mask, gb)
				#utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
