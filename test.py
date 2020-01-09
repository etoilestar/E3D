import os
import numpy as np
import torch
from config import params
from torch import nn
from lib.dataset import VideoDataset
from lib.datasetflow import VideoFlow
from lib.test_dataset import VideoDataTest
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import glob
import math
import cv2
from tqdm import tqdm
import multiprocessing
from thop import profile
import collections

flow_dir = params['dataset_flow']
rgb_dir = params['dataset_rgb']

def get_model():
    model1 = EfficientNet.from_name(params['pretrained'], data='flow',
                                    override_params={'num_classes': params['num_classes']})
    pretrained_dict1 = torch.load(params['pretrained3d'], map_location='cpu')
    try:
        model_dict1 = model1.module.state_dict()
    except AttributeError:
        model_dict1 = model1.state_dict()
    #    pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in model_dict1}
    pretrained_dict1 = {k: v for k, v in pretrained_dict1.items()}
    #    print(pretrained_dict1.keys())
    model_dict1.update(pretrained_dict1)
    model1.load_state_dict(model_dict1)
    model1 = model1.cuda(params['gpu'][0])
    model1 = nn.DataParallel(model1, device_ids=params['gpu'])
    model1.eval()
    model2 = EfficientNet.from_name(params['pretrained'], data='rgb',
                                    override_params={'num_classes': params['num_classes']})
    pretrained_dict2 = torch.load(params['pretrained2d'], map_location='cpu')
    try:
        model_dict2 = model2.module.state_dict()
    except AttributeError:
        model_dict2 = model2.state_dict()
    pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict2}
    model_dict2.update(pretrained_dict2)
    model2.load_state_dict(model_dict2)
    model2 = model2.cuda(params['gpu'][0])
    model2 = nn.DataParallel(model2, device_ids=params['gpu'])
    model2.eval()
    return model1, model2

def to_tensor(buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose((3, 0, 1, 2))

def normalize(buffer):
    # Normalize the buffer
    # buffer = (buffer - 128)/128.0
    for i, frame in enumerate(buffer):
        # frame = (frame - np.array([[[128.0, 128.0]]]))/128.0
        frame = (frame - 128.0)/128.0
        buffer[i] = frame
    return buffer

def deal(buffer):
    buffer = normalize(buffer)
    buffer = to_tensor(buffer)
    return buffer

def loadvideo(fname, x):
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    retaining, frame = capture.read()
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = x*params['gap']
    end = start + 32
    buffer = np.empty((32, 224, 224, 3), np.dtype('float32'))
    sample_count = 0
   
    for j in range(end):
        retaining, frame = capture.read()
        if retaining is False:
            capture = cv2.VideoCapture(fname)
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
    if np.any(np.isnan(buffer)):
        print('buffer NAN')

    return buffer

def loadflow(fname_u, fname_v, x):
    u_list = os.listdir(fname_u)
    v_list = os.listdir(fname_v)
    u_list1, v_list1 = u_list, v_list
#        print(u_list)
    video = cv2.imread(os.path.join(fname_u, u_list[0]), 0)
    h = video.shape[0]
    w = video.shape[1]
    start = x*params['gap']
    end = start + 32
    if end > len(u_list):
        u_list += u_list1
        v_list += v_list1

    buffer = np.empty((32, 224, 224, 2), np.dtype('float32'))
    y = 0
    for i in range(start, end):
        u = cv2.imread(os.path.join(fname_u, u_list[i]), 0)
        v = cv2.imread(os.path.join(fname_v, v_list[i]), 0)
        u = np.array(u)[int(h/2-224/2):int(h/2-224/2)+224, int(w/2-224/2):int(w/2-224/2)+224]
        v = np.array(v)[int(h/2-224/2):int(h/2-224/2)+224, int(w/2-224/2):int(w/2-224/2)+224]
        u = np.expand_dims(u, axis=2)
        v = np.expand_dims(v, axis=2)
        total = np.concatenate([u, v], axis=2)
        buffer[y] = total
        y += 1
    return buffer


def get_data(data_flow_u, data_flow_v, data_rgb, x):
    buffer_flow = loadflow(data_flow_u, data_flow_v, x)
    buffer_flow = deal(buffer_flow)

    buffer_rgb = loadvideo(data_rgb, x)
    buffer_rgb = deal(buffer_rgb)
    return buffer_flow, buffer_rgb


def cal(label_name2label, model_flow, model_rgb,class_TorF_dict,data_flow_u_list1, data_flow_v_list1, data_rgb_list1):
#    print(len(data_flow_u_list1))
    success_num_flow, success_num_rgb, success_num = 0, 0, 0
    total_num = 0
    fail_num = 0
#    flops, params = 0, 0
    with torch.no_grad():
        for data_flow_u, data_flow_v, data_rgb in zip(data_flow_u_list1, data_flow_v_list1, data_rgb_list1):
            flow_len = len(os.listdir(data_flow_u))
            data_flow_u, data_flow_v, data_rgb = os.path.abspath(data_flow_u), os.path.abspath(data_flow_v), os.path.abspath(data_rgb)
#        print(data_flow_u, data_flow_v, data_rgb)
            if data_flow_u.split('/')[-2] == data_rgb.split('/')[-2]:
                label_id = label_name2label[data_flow_u.split('/')[-2]]
            else:
                print('Error label!')
                return
            if flow_len <= 32:
                amount = 1
            else:
                amount = math.ceil((flow_len - 32) / params['gap'])+1
            for x in tqdm(range(amount)):
                inputs_flow, inputs_rgb = get_data(data_flow_u, data_flow_v, data_rgb, x)
                inputs_flow, inputs_rgb = torch.from_numpy(inputs_flow).unsqueeze_(0).cuda(params['gpu'][1]), torch.from_numpy(inputs_rgb).unsqueeze_(0).cuda(params['gpu'][0])
                outputs_flow = model_flow(inputs_flow)
                outputs_rgb = model_rgb(inputs_rgb)
                if x == 0:
                    outputs_flow_add, outputs_rgb_add = outputs_flow.cpu(), outputs_rgb.cpu()
       #             flops_flow, params_flow = profile(model_flow, inputs=(inputs_flow,))
       #             flops_rgb, params_rgb = profile(model_rgb, inputs=(inputs_rgb,))
                else:
                    outputs_flow_add, outputs_rgb_add = torch.add(outputs_flow_add, outputs_flow.cpu()), torch.add(outputs_rgb_add, outputs_rgb.cpu())
                del inputs_flow
                del inputs_rgb
                del outputs_flow
                del outputs_rgb
            outputs_add = torch.add(outputs_flow_add, outputs_rgb_add)
            pred_flow = torch.argmax(outputs_flow_add, 1)
            pred_rgb = torch.argmax(outputs_rgb_add, 1)
            pred = torch.argmax(outputs_add, 1)
            TorF_flow = pred_flow == label_id
            TorF_rgb = pred_rgb == label_id
            TorF = pred == label_id
            if TorF_flow:
                success_num_flow += 1
 #               print('flow success')
            if TorF_rgb:
                success_num_rgb += 1
            if TorF:
                success_num += 1
                class_TorF_dict[data_flow_u.split('/')[-2]] += np.array([1,0,0])
                print(data_rgb+'  success!')
                with open('res_hmdb.txt', 'a') as f:
                    f.write(data_rgb+'  success!\n')
            elif not TorF:
                fail_num += 1
                class_TorF_dict[data_flow_u.split('/')[-2]] += np.array([0, 1, 0])
                with open('res_ucf.txt', 'a') as f:
                    print(data_rgb+ '  fail!')
                    print(pred, label_id)
                    f.write(data_rgb+'  fail!\n')
            class_TorF_dict[data_flow_u.split('/')[-2]] += ([0,0,1])
            total_num += 1
#    print('flow acc:', success_num_flow/total_num)
#    print('rgb acc:', success_num_rgb/total_num)
#    print('flow-rgb acc:', success_num/total_num)
#    print(flops_rgb+flops_flow, params_rgb+params_flow)
    return success_num_flow, success_num_rgb,success_num,total_num, class_TorF_dict

if __name__ == '__main__':
    thread_num = 4
    pool = multiprocessing.Pool(processes=thread_num)
    result = []
    model_flow, model_rgb = get_model()
    label_name2label = {}
 #   label_name2label = collections.OrderedDict()
    success_num_flow, success_num_rgb, success_num, total_num = 0, 0, 0, 0
    class_TorF_dict = {}
    class_TorF_dict = collections.OrderedDict()
    with torch.no_grad():
        for label, label_name in enumerate(sorted(os.listdir(os.path.join(params['dataset_flow'],'valid')))):
            label_name2label[label_name] = label
            class_TorF_dict[label_name] = np.array([0, 0, 0])
        data_flow_u_list = sorted(glob.glob(os.path.join(flow_dir, 'valid/*/*_u')))
        data_flow_v_list = sorted(glob.glob(os.path.join(flow_dir, 'valid/*/*_v')))
        data_rgb_list = sorted(glob.glob(os.path.join(rgb_dir, 'valid/*/*.avi')))
        n = int(math.ceil(len(data_flow_u_list) / float(thread_num)))
#        cal(label_name2label,model_flow, model_rgb,data_flow_u_list,data_flow_v_list,data_rgb_list)
        for i in range(0, len(data_flow_u_list), n):
            result.append(pool.apply_async(cal, (label_name2label,model_flow, model_rgb,class_TorF_dict,data_flow_u_list[i: i+n], data_flow_v_list[i: i+n], data_rgb_list[i: i+n],)))
        for s in range(len(result)):
            success_num_flow1, success_num_rgb1,success_num1,total_num1, class_TorF_dict1 = result[s].get()
            for key, value in class_TorF_dict.items():
                class_TorF_dict[key] += class_TorF_dict1[key]
#           print(success_num_flow1, success_num_rgb1,success_num1,total_num1)
            success_num_flow, success_num_rgb,success_num,total_num = success_num_flow+success_num_flow1, success_num_rgb+success_num_rgb1,success_num+success_num1,total_num+total_num1
        pool.close()
        pool.join()
        with open('hmdbresult.txt', 'a') as f:
            for key, value in class_TorF_dict.items():
                  print('total {} class number:{}'. format(key, value[2]))
                  print('success to predict {} class number:{}'. format(key, value[0]))
                  print('fail to predict {} class number:{}'. format(key, value[1]))
                  f.write('total {} class number:{}\n'. format(key, value[2]))
                  f.write('success to predict {} class number:{}\n'. format(key, value[0]))
                  f.write('fail to predict {} class number:{}\n'. format(key, value[1]))
            print('flow acc:', success_num_flow/total_num)
            print('rgb acc:', success_num_rgb/total_num)
            print('flow-rgb acc:', success_num/total_num)
            f.write('flow acc:'+str(success_num_flow/total_num)+'\n')
            f.write('rgb acc:'+str(success_num_rgb/total_num)+'\n')
            f.write('flow-rgb acc:'+str(success_num/total_num)+'\n')

#    print(result)




