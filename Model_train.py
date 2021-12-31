#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import torch
from network.Model_define_pytorch import DatasetFolder, NMSE
from network.Model_define_pytorch import AutoEncoder
# from network.model import AutoEncoder
import os
import torch.nn as nn

from utils.losses import NMSELoss


def Score(NMSE):
    score = (1 - NMSE) * 100
    if score < 0:
        score = 0
    return score


def train_on_epoch():
    # model training
    model.train()
    for i, input in enumerate(train_loader):
        # adjust learning rate
        if epoch == 300:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * 0.1
        input = input.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, input)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\tLoss {loss:.6f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
    scheduler.step()


def validation():
    # model evaluating
    model.eval()
    total_loss = 0
    x_test = []
    y_test = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            x_test.append(input)
            output = model(input)
            y_test.append(output)

            total_loss += criterion(output, input).item() * input.size(0)

        average_loss = total_loss / len(test_dataset)

    x_test = torch.cat(x_test, dim=0).cpu().numpy()
    y_test = torch.cat(y_test, dim=0).cpu().numpy()
    NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)), np.transpose(y_test, (0, 2, 3, 1)))
    scr = Score(NMSE_test)
    print(f'The NMSE is {NMSE_test}, score={scr}')

    return NMSE_test


if __name__ == '__main__':
    # Parameters for training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_single_gpu = True  # select whether using single gpu or multiple gpus
    torch.manual_seed(1)
    batch_size = 16
    epochs = 1000
    learning_rate = 1e-3
    num_workers = 4
    print_freq = 10  # print frequency (default: 60)
    # parameters for data
    feedback_bits = 512
    efn_name = 'efficientnet-b1'
    data_load_address = '/data1/chenby/dataset/NAIC/train'
    save_root = f'./output/{efn_name}/weights'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Model construction

    model = AutoEncoder(feedback_bits, efn_name=efn_name)
    # encoder_path = f'{save_root}/encoder.pth.tar'
    # model.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu')['state_dict'])
    # print(f'Load encoder in {encoder_path}')
    # decoder_path = f'{save_root}/decoder.pth.tar'
    decoder_path = './output/baseline/weights/decoder.pth.tar'
    model.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu')['state_dict'])
    print(f'Load decoder in {decoder_path}')

    if use_single_gpu:
        model = model.cuda()
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        autoencoder = torch.nn.DataParallel(model).cuda()
    import scipy.io as scio

    criterion = nn.MSELoss().cuda()
    # criterion = NMSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-9,
                                                                     last_epoch=-1)

    mat = scio.loadmat(data_load_address + '/Htrain.mat')
    x_train = mat['H_train']  # shape=8000*126*128*2
    x_train = np.transpose(x_train.astype('float32'), [0, 3, 1, 2])
    mat = scio.loadmat(data_load_address + '/Htest.mat')
    x_test = mat['H_test']  # shape=2000*126*128*2
    x_test = np.transpose(x_test.astype('float32'), [0, 3, 1, 2])
    print(x_train.shape, x_test.shape)

    # Data loading
    # dataLoader for training
    train_dataset = DatasetFolder(x_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    # dataLoader for training
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    is_training = True
    if is_training:
        best_loss = 1
        for epoch in range(epochs):
            train_on_epoch()
            average_loss = validation()
            if average_loss < best_loss:
                # model save
                # save encoder
                modelSave1 = f'{save_root}/encoder.pth.tar'
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
                # save decoder
                modelSave2 = f'{save_root}/decoder.pth.tar'
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
                print("Model saved")
                best_loss = average_loss
            print(f'Current loss:{average_loss}, Best loss:{best_loss}')
    else:
        average_loss = validation()
