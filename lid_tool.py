import os
import time
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s]： %(message)s',
                    )

import torch
import torch.utils.data as Data

from torch.utils.data import DataLoader
from data.dataloader import AudioDataset, CollateFunc
from network import LanNet

# ======================================
# 配置文件和参数
# 数据列表
train_list = "../dataset/train.txt"
test_list   = "../dataset/test.txt"

# 基本配置参数
use_cuda = False 
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 保存模型地址
model_dir = "./models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
    
# 网络参数
dimension = 40
language_nums = 3
learning_rate = 0.1
batch_size = 8
train_iteration = 10
display_fre = 5
half = 4

# 构建数据迭代器
train_dataset = AudioDataset(train_list, batch_size=batch_size, shuffle=True)
test_dataset = AudioDataset(test_list, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=CollateFunc())
test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=CollateFunc())

logging.info('finish reading all train data')

# 设计网络优化器
train_module = LanNet(input_dim=dimension, hidden_dim=32, bn_dim=30, output_dim=language_nums)
print(train_module)

optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

# 将模型放入GPU中
if use_cuda:
    train_module = train_module.to(device)

# 模型训练
for epoch in range(train_iteration):
    if epoch >= half:
        learning_rate /= 2.
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

    train_module.train()
    epoch_tic = time.time()
    train_loss = 0.
    train_acc = 0.

    sum_batch_size = 0
    curr_batch_size = 0
    curr_batch_acc = 0
    tic = time.time()
    for step, batch_data in enumerate(train_loader): 
        # import pdb; pdb.set_trace()
        
        batch_max_frames = max(batch_data[-1])
        batch_size = len(batch_data[-1])
        batch_mask = torch.zeros(batch_size, batch_max_frames)
        for ii in range(batch_size):
            frames = batch_data[-1][ii]
            batch_mask[ii, :frames] = 1.

        batch_train_data = batch_data[1]
        batch_train_target = batch_data[2].view(-1, 1).long()

        # 将数据放入GPU中
        if use_cuda:
            batch_train_data = batch_train_data.to(device)
            batch_mask       = batch_mask.to(device)
            batch_train_target     = batch_train_target.to(device)

        acc, samples, loss = train_module(batch_train_data, batch_mask, batch_train_target)

        backward_loss = loss
        optimizer.zero_grad()
        backward_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc
        curr_batch_acc += acc
        sum_batch_size += samples
        curr_batch_size += samples
        if (step+1) % display_fre == 0:
            toc = time.time()
            step_time = toc-tic
            tic = time.time()
            logging.info('Epoch:%d, Batch:%d, acc:%.6f, loss:%.6f, cost time: %.6fs', epoch, step+1, curr_batch_acc/curr_batch_size, loss.item(), step_time)
            curr_batch_acc = 0.
            curr_batch_size = 0
 
    # import pdb; pdb.set_trace()

    # 模型存储
    modelfile = '%s/model%d.model'%(model_dir, epoch)
    torch.save(train_module.state_dict(), modelfile)
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    logging.info('Epoch:%d, train-acc:%.6f, train-loss:%.6f, cost time: %.6fs', epoch, train_acc/sum_batch_size, train_loss/sum_batch_size, epoch_time)

    # 模型验证
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 

    for step, batch_data in enumerate(test_loader): 
        batch_max_frames = max(batch_data[-1])
        batch_size = len(batch_data[-1])
        batch_mask = torch.zeros(batch_size, batch_max_frames)
        for ii in range(batch_size):
            frames = batch_data[-1][ii]
            batch_mask[ii, :frames] = 1.

        batch_train_data = batch_data[1]
        batch_train_target = batch_data[2].view(-1, 1).long()

        # 将数据放入GPU中
        if use_cuda:
            batch_train_data = batch_train_data.to(device)
            batch_mask       = batch_mask.to(device)
            batch_train_target     = batch_train_target.to(device)
            
        with torch.no_grad():
            acc, samples, loss = train_module(batch_train_data, batch_mask, batch_train_target)
        
        dev_loss += loss.item()
        dev_acc += acc
        dev_batch_num += samples
    
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    logging.info('Epoch:%d, dev-acc:%.6f, dev-loss:%.6f, cost time: %.6fs\n', epoch, dev_acc/dev_batch_num, dev_loss/dev_batch_num, epoch_time)
