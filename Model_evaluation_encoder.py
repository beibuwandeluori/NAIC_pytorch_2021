#!/usr/bin/env python3
import numpy as np
# from network.Model_define_pytorch import AutoEncoder, DatasetFolder
from submissions.project.Model_define_pytorch import AutoEncoder, DatasetFolder
import torch
import os

if __name__ == '__main__':
    # Parameters for training
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    batch_size = 64
    num_workers = 4
    # Data parameters
    save_root = './output/baseline'
    img_height = 16
    img_width = 32
    img_channels = 2
    feedback_bits = 512
    model_ID = 'CsiNet'  # Model Number
    import scipy.io as scio

    # load test data
    data_load_address = '/data1/chenby/dataset/NAIC/train'
    mat = scio.loadmat(data_load_address + '/Htest.mat')
    x_test = mat['H_test']  # shape=2000*126*128*2
    x_test = np.transpose(x_test.astype('float32'), [0, 3, 1, 2])

    # load model
    model = AutoEncoder(feedback_bits).cuda()
    model_encoder = model.encoder
    model_path = f'{save_root}/weights/encoder.pth.tar'
    model_encoder.load_state_dict(torch.load(model_path)['state_dict'])
    print("weight loaded")

    # dataLoader for test
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # test
    model_encoder.eval()
    encode_feature = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model_encoder(input)
            output = output.cpu().numpy()
            if i == 0:
                encode_feature = output
            else:
                encode_feature = np.concatenate((encode_feature, output), axis=0)
    print("feedbackbits length is ", np.shape(encode_feature)[-1])
    np.save(f'{save_root}/results/encoder_output.npy', encode_feature)