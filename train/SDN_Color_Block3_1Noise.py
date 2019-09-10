import re
import os, glob, datetime, time
import numpy as np
from skimage import io
from skimage.measure import compare_psnr, compare_ssim

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 16
gamma = 0.5
sigmaU = 1
sigmaW = 10
epochs = 100
LR = 1e-4
file_mul = 2
total_images = 33600
# total_images = 47046
train_data = './train_128_color.h5'
model_name = 'SDN_Color_Block3_1Noise_gamma_%.1f_sigmaU_%.1f_sigmaW_%d'%(gamma, sigmaU, sigmaW)

save_dir = os.path.join('Models', model_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
class Block3(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super(Block3, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        
        c1 = self.conv1(x)
        c2 = self.conv2(c1+x)
        c3 = self.conv3(c2+x)
        
        return c3

class SDNCNN(nn.Module):
    
    def __init__(self, filters=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(SDNCNN, self).__init__()
        kernel_size = 3
        padding = 1
        self.conv0 = nn.Conv2d(in_channels=image_channels, out_channels=filters, kernel_size=kernel_size, padding=padding)
        
        self.convOut = nn.Conv2d(in_channels=filters, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True)
        
        self.ResBlock1 = Block3(filters)
        self.ResBlock2 = Block3(filters)
        self.ResBlock3 = Block3(filters)
        self.ResBlock4 = Block3(filters)
        self.ResBlock5 = Block3(filters)
        self.ResBlock6 = Block3(filters)
        self.ResBlock7 = Block3(filters)
        
        self._initialize_weights()

    def forward(self, x):

        c0 = self.conv0(x)
        
        c1 = self.ResBlock1(c0)
        
        c2 = self.ResBlock2(c1)
        
        c3 = self.ResBlock3(c2)
        
        c4 = self.ResBlock4(c3)

        c5 = self.ResBlock5(c4+c3)
        
        c6 = self.ResBlock6(c5+c2)
        
        c7 = self.ResBlock7(c6+c1)
        
        noise = self.convOut(c7)
        
        rec = x - noise
        
        return rec, noise

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def calculate_psnr_fast(prediction, target):
    assert prediction.max().cpu().data.numpy() <= 1
    assert prediction.min().cpu().data.numpy() >= 0
    psnr_list = []
    for i in range(prediction.size(0)):
        mse = torch.mean(torch.pow(prediction.data[i]-target.data[i], 2))
        mse = mse.cpu()
        try:
            psnr_list.append(10 * np.log10(1**2 / mse))
        except:
            print('error in psnr calculation')
            continue
    return psnr_list


model = SDNCNN()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

model.cuda()
criterion = sum_squared_error()

optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-6)

output_file_name = 'Log/Log_output_%s.txt'%(model_name)

for epoch in range(initial_epoch, epochs):
    start_time = time.time()
#     if epoch > 20:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-4
#     if epoch > 40:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-7
    for param_group in optimizer.param_groups:
        print('learning rate is %f'%param_group['lr'])
    
    DDataset = Dataset(train_data, file_mul=file_mul)
    DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
  
    model.train()
    
    psnr_list = []
    epoch_loss = 0
    for n_count, data in enumerate(DLoader):
        
        batch_x = data
        u = torch.FloatTensor(batch_x.size()).normal_(0, sigmaU)
        w = torch.FloatTensor(batch_x.size()).normal_(0, sigmaW)
        noise = batch_x.pow(gamma)*u + w
        batch_y = batch_x + noise
        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        noise = noise.cuda()
        
        optimizer.zero_grad()
        out, noise_out = model(batch_y)
        
        loss = criterion(out, batch_x)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        psnr_list += calculate_psnr_fast(out.clamp(0, 255)/255.0, batch_x.clamp(0, 255)/255.0)
        if n_count % 10 == 0:
            print('%4d %4d / %4d loss = %2.4f, psnr is %.2f' % (epoch+1, n_count, total_images//file_mul//batch_size, loss.item()/batch_size, psnr_list[-1]))
    
    
    mean_psnr = np.array(psnr_list)
    mean_psnr = mean_psnr[mean_psnr != np.inf].mean()
    
    torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
    
    elapsed_time = time.time() - start_time
    output_data = 'Epoch[%d/%d] - Loss: %.4f, Train: %.3f, Time is %.2f    ' % (epoch+1, epochs, epoch_loss*file_mul/total_images, mean_psnr, elapsed_time)
    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()
    
    print('Epoch[%d/%d] - Loss: %.4f, Train: %.3f, Time is %.2f' % (epoch+1, epochs, epoch_loss*file_mul/total_images, mean_psnr, elapsed_time))
    
    with torch.no_grad():
        
        test_data = './Test_datasets/CBSD68/*.png'
        test_dir = glob.glob(test_data)

        initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
        model = model.cuda()
        model.eval()

        psnr_list = []
        ssim_list = []
        
        for i in range(len(test_dir)):
            
            batch_x = io.imread(test_dir[i])
            u = np.random.normal(0, sigmaU, batch_x.shape)
            w = np.random.normal(0, sigmaW, batch_x.shape)
            noise = np.power(batch_x, gamma)*u + w
            batch_y = batch_x + noise
            
            batch_x = torch.from_numpy(batch_x.transpose(2,0,1).astype('float32'))[None,:,:,:].cuda()
            batch_y = torch.from_numpy(batch_y.transpose(2,0,1).astype('float32'))[None,:,:,:].cuda()

            out, noise_out = model(batch_y)
            
            batch_x = batch_x.clamp(0, 255)[0,...].cpu().detach().numpy().transpose(1,2,0)/255.0
            out = out.clamp(0, 255)[0,...].cpu().detach().numpy().transpose(1,2,0)/255.0

            psnr_list += [compare_psnr(out, batch_x)]
            ssim_list += [compare_ssim(out, batch_x, multichannel=True)]

        output_data = 'PSNR: %.2f, SSIM: %.4f\n' % (np.mean(psnr_list), np.mean(ssim_list))
        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        print('PSNR: %.2f, SSIM: %.4f\n' % (np.mean(psnr_list), np.mean(ssim_list)))