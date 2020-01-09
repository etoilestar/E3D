import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib.datasetflow import VideoFlow
from lib.MYdataset import MYDataset
from lib import resnet
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F
from lib import EfficientNet
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, way, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels, file) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda(params['gpu'][0])
        labels = labels.cuda(params['gpu'][0])
        outputs = model(inputs)


        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, 1, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (step+1) % params['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)
    return loss

def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()



    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels, files) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda(params['gpu'][0])
            labels = labels.cuda(params['gpu'][0])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(outputs.data, labels, 2, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)

            for name, layer in model._modules.items():
                x = inputs.view(x.size(0), -1) if "fc" in name else inputs
                x = layer(x)
                x = F.relu(x) if 'conv' in name else x
                if 'layer' in name or 'conv' in name:
                    img_grid = vutils.make_grid(x.transpose(0, 1), normalize=True, scale_each=True, nrow=4)
                    writer.add_image('{}_feature_maps'.format(name), img_grid, global_step=0)
            for param in model.named_parameters():
                if 'conv' in name and 'weight' in name:
                    inchannels = param.size()[1]
                    outchannels = param.size()[0]
                    kw, kh = param.size()[3], param.size()[2]
                    kernel_all = param.view(-1, 1, kw, kh)
                    kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=inchannels)
                    writer.add_image('{}_all'.format(name), kernel_grid, global_step=0)
    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)
    return loss

def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    Data = params['data']

    if Data =='rgb':
        dataset = params['dataset_rgb']
    elif Data == 'flow':
        dataset = params['dataset_flow']


    train_dataloader = \
        DataLoader(
            MYDataset(dataset, data=Data, mode='train',clip_len=params['clip_len'], method='total_random'),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            MYDataset(dataset, data=Data, mode='valid',clip_len=params['clip_len'], method='total_random'),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])


    print("load model")
#    model = resnet.resnet50(num_classes=params['num_classes'])
  
    model = EfficientNet.from_name(params['pretrained'], data=Data, override_params={'num_classes': params['num_classes']})
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained3d'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda(1)
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.6, patience=6, verbose=True)
    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        loss = train(model, train_dataloader, epoch, criterion, optimizer, writer)
        validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        scheduler.step(loss)
        if epoch % 1 == 0:
            checkpoint = os.path.join(model_save_dir,
                                      "clip_len_" + str(params['clip_len']) + "frame_sample_rate_" +str(params['frame_sample_rate'])+ "_checkpoint_" + str(epoch) + ".pth.tar")
            torch.save(model.module.state_dict(), checkpoint)

    writer.close

if __name__ == '__main__':
    main()
