import os
from pathlib import Path
import random
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory, mode='train', clip_len=64, frame_sample_rate=1):
        folder = os.path.join(directory, mode)  # get the directory of the specified split
        print(folder)
        self.clip_len = clip_len

        self.short_side = [230, 256]
        self.crop_size = 224
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
#        lines = []
#        f = open('/data/zhengrui/train/SlowFastNetworks/lib/k400.txt', 'r')
#        lines1 = f.readlines()
#        for line in lines1:
#            lines.append(line[:-1])
#        print(lines)
#        f.close()
        self.fnames, labels = [], []
        i1 = 0
        for label in sorted(os.listdir(folder)):
#            if i1<101 and (label in lines):
            if i1<600:
                for fname in sorted(os.listdir(os.path.join(folder, label))):
               #     if 'v_BasketballDunk_g07_c02.avi' != fname and 'v_BlowingCandles_g05_c03.avi' != fname and 'v_SkyDiving_g02_c01.avi' != fname: 
#                        print(fname)
                    if True:
                        self.fnames.append(os.path.join(folder, label, fname))
                        labels.append(label)
                    else:
                        print('pass:',fname)
                i1 += 1
        print(str(i1)+'\n')
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_file = str(len(os.listdir(folder)))+'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def addGaussianNoise(self, frame, percetage):
        G_Noiseimg = frame.copy()

        w = frame.shape[1]
        h = frame.shape[0]
        G_NoiseNum = int(percetage * frame.shape[0] * frame.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        return G_Noiseimg

    def randomaddG(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        for i, frame in enumerate(buffer):
            buffer[i] = self.addGaussianNoise(frame,0.01)
        return buffer

    def deal(self, buffer):
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return buffer

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        try:    
#            print(self.fnames[index])
            buffer = self.loadvideo(self.fnames[index])
            while buffer.shape[0]<self.clip_len+2 :
                index = np.random.randint(self.__len__())
                buffer = self.loadvideo(self.fnames[index])
            if self.mode == 'train' or self.mode == 'training':
                if np.random.random() < 0.5:
                    buffer = self.randomflip(buffer)
#            if np.random.random() < 0.5:
#                buffer = self.randomaddG(buffer)
#            if np.random.random() < 0.5:
#                buffer = self.randomrotate(buffer)
            buffer = self.deal(buffer)
            return buffer, self.label_array[index], self.fnames[index]
        except ZeroDivisionError:
            return np.asarray([1]), -1, self.fnames[index]
        except cv2.error:
            return np.asarray([1]), -1, self.fnames[index]
        except ValueError:
            return np.asarray([1]), -1, self.fnames[index]

    def rotate(self, frame, angle=15, scale=0.9):
        w = frame.shape[1]
        h = frame.shape[0]  # rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle*random.randint(1, 3), scale)    #rotate
        frame = cv2.warpAffine(frame,M,(w,h))
        return frame

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
#        print(fname)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            crop1 = int((frame_width-(7/6)*frame_height)/2)
            crop2 = frame_width - crop1
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * (frame_width-2*crop1))
        else:
            crop1 = int((frame_height-(7/6)*frame_width)/2)
            crop2 = frame_height - crop1
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * (frame_height-2*crop1))

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count%self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                if frame_height < frame_width:
                   frame = frame[crop1:crop2, :, :]
                if frame_height > frame_width:
                   frame = frame[:, crop1:crop2, :]
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer

    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def randomrotate(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        for i, frame in enumerate(buffer):
            buffer[i] = self.rotate(frame)
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        for i, frame in enumerate(buffer):
            buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    datapath = '/data/zhengrui/dataset/ucf101'
    with open('vbrokenvideo.txt', 'a') as f:
        flist = []
        train_dataloader = \
            DataLoader( VideoDataset(datapath, mode='train'), batch_size=4, shuffle=False, num_workers=0)
       #print(enumerate(train_dataloader))
        for step, (buffer, label, fname) in enumerate(train_dataloader):
            if label == -1:
                print('fail:',fname[0])
                flist.append(fname[0])
                f.write('fail:'+fname[0]+'\r\n')
            else:
                print('success:', fname[0])
    print('fail:', flist)
#    imagepath = os.path.join(datapath, 'copy')
#    if not os.path.exists(imagepath):
#        os.mkdir(imagepath)
#    for i in range(30):
#        for step, (buffer, label, fname) in enumerate(train_dataloader):
#            if len(fname) == 10:
                #for j in range(10):
#                _, v_name = os.path.split(fname[0])
#                name, ext = os.path.splitext(v_name)
#                imageclassdir = os.path.join(imagepath, name)
#                if not os.path.exists(imageclassdir):
##                    os.mkdir(imageclassdir)
#                for k in range(16):
#                    print(os.path.join(imageclassdir, name+'_'+str(i)+'_'+str(k)+'.jpg'))
#                    frame = buffer.numpy()[0].transpose(1, 2, 3, 0)[k]
#                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                    frame = frame*128.0+128.0
#                    cv2.imwrite(os.path.join(imageclassdir, name+'_'+str(i)+'_'+str(k)+'.jpg'), frame)
#            else:
#                print('-------------------------------------------')
               # print("label: ", label)
               # print(buffer.numpy().shape)
               # print(fname)
