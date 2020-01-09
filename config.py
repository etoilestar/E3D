params = dict()

params['num_classes'] =101
params['data'] = 'rgb'
params['dataset_flow'] = '/data/zhengrui/dataset/ucf-flow2/'
params['dataset_rgb'] = '/data/zhengrui/dataset/ucf101/'
params['epoch_num'] = 300
params['gap'] = 16
params['batch_size'] = 32
params['step'] = 30
params['num_workers'] = 8
params['learning_rate'] = 1e-3
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = '/data/zhengrui/dataset/pretrain/efficientnet-b0.pth'

params['pretrained3d'] = '/data/zhengrui/train/SlowFastNetworks/ucf101/2019-10-10-22-12-44/clip_len_32frame_sample_rate_1_checkpoint_49.pth.tar'#'/data/zhengrui/dataset/pretrain/efficientnet-b0.pth'#'/data/zhengrui/train/SlowFastNetworks/hmdb51/2019-10-05-10-34-07/clip_len_32frame_sample_rate_1_checkpoint_31.pth.tar'#'/data/zhengrui/dataset/pretrain/ef0_rgb1.pth.tar'#'/data/zhengrui/train/SlowFastNetworks/ucf101/2019-09-25-22-25-43/clip_len_32frame_sample_rate_1_checkpoint_32.pth.tar'#'/data/zhengrui/dataset/pretrain/ef0_flow5.pth.tar'#'/data/zhengrui/dataset/pretrain/efficientnet-b0.pth'#'/data/zhengrui/dataset/pretrain/ef0_flow_hmdb7.pth.tar'#'/data/zhengrui/dataset/pretrain/efficientnet-b0-flow.pth'#'/data/zhengrui/dataset/pretrain/ef0_flow_hmdb2.pth.tar'#'/data/zhengrui/dataset/pretrain/ef0_flow5.pth.tar'#'/data/zhengrui/dataset/pretrain/efficientnet-b0-flow.pth'#'/data/zhengrui/dataset/pretrain/efficientnet-b0.pth'#'/data/zhengrui/dataset/pretrain/ef0_flow5.pth.tar'#'/data/zhengrui/dataset/pretrain/efficientnet-b0-flow.pth'
params['pretrained2d'] = '/data/zhengrui/train/SlowFastNetworks/hmdb51/2019-10-05-15-47-34/clip_len_32frame_sample_rate_1_checkpoint_15.pth.tar'#'/data/zhengrui/train/SlowFastNetworks/ucf101/2019-09-27-18-56-42/clip_len_32frame_sample_rate_1_checkpoint_81.pth.tar'#'/data/zhengrui/dataset/pretrain/ef0_rgb.pth.tar'
params['gpu'] = [0, 1]
params['log'] = 'log'
params['save_path'] = 'ucf101'
params['clip_len'] = 32
params['frame_sample_rate'] = 1
params['n_sample'] = 3
