
DATASET_PATH ='~/data'


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")
import torch
import torchvision
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import random
#from logger import SummaryLogger
import utils
import network_Densenet
import torch






parser = argparse.ArgumentParser(description='Quantization finetuning for CIFAR100')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar100/Densenet_40_12', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='400', type=int)
parser.add_argument('--decay_epoch', default=[150, 225], nargs="*", type=int)
parser.add_argument('--w_decay', default='2e-4', type=float)
parser.add_argument('--cu_num', default='1', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained', default='models/ResNet82.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--depth', type=int, default=40, help='Model depth.')
parser.add_argument('--widen_factor', type=int, default=12, help='Model width.')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
args = parser.parse_args()


print(args)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.manual_seed(int(args.seed))

#### random Seed #####
# random.seed(random.randint(1, 10000))
# torch.manual_seed(args.manualSeed)
#####################

batch_size = 128
trainloader, valloader, testloader = utils.get_cifar100_dataloaders(batch_size, 100)
num_classes = 100

#Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr
 
model = network_Densenet.CrossKDDenseNet(num_classes=num_classes)
# According to CIFAR
# module = network_VGG.BRUModule(1024,num_classes=num_classes)
module = network_Densenet.Fusion_module(688, num_classes, 1)
model.to(DEVICE)
module.to(DEVICE)

# Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion_CE = nn.CrossEntropyLoss()
criterion_kl = utils.KLLoss().cuda()
optimizer_FM = optim.SGD(module.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
scheduler_FM = optim.lr_scheduler.MultiStepLR(optimizer_FM, milestones=DECAY_EPOCH, gamma=0.1)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return utils.sigmoid_rampup(epoch, args.consistency_rampup)

def eval(net, module, epoch, test_flag=False):
    loader = valloader if not test_flag else testloader
    flag = 'Val.' if not test_flag else 'Test'

    epoch_start_time = time.time()
    net.eval()
    module.eval()
    val_loss = 0

    correct_sub1 = 0
    correct_sub2 = 0
    correct_fused = 0

    total = 0
    consistency_weight = get_current_consistency_weight(epoch)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs1, outputs2, crossFusionKnowledge, fmap, (net1_2_output,net2_2_output) = model(inputs)

        fused_logit = module(fmap[0].unsqueeze(-1).unsqueeze(-1), fmap[1].unsqueeze(-1).unsqueeze(-1))

        loss_cross = criterion_CE(outputs1, targets) + criterion_CE(outputs2, targets) + criterion_CE(fused_logit, targets)
        # MKD
        # loss_kl = (criterion_kl(outputs1, outputs2) + criterion_kl(outputs2, outputs1) + utils.lossOfKnowledge(criterion_kl, crossFusionKnowledge) + criterion_kl(fused_logit, ensemble_logit))
        loss_knowledge = utils.lossOfKnowledge(criterion_kl, crossFusionKnowledge)
        loss_kl = consistency_weight*(criterion_kl(outputs1, outputs2) + criterion_kl(outputs2, outputs1) + loss_knowledge + criterion_kl(outputs1, fused_logit) + criterion_kl(outputs2, fused_logit) +  criterion_kl(fused_logit, outputs1) + criterion_kl(fused_logit, outputs2))

        loss = loss_cross + loss_kl
        # loss = loss_cross
        #val_loss += loss.item()
        val_loss=val_loss.clone()+loss.item()


        _, predicted_sub1 = torch.max(outputs1.data, 1)
        _, predicted_sub2 = torch.max(outputs2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        #total += targets.size(0)
        total=total.clone()+targets.size(0)

        #correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        #correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        #correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()
        correct_sub1=correct_sub1.clone()+predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2=correct_sub2.clone()+predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused=correct_fused.clone()+predicted_fused.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx
    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (train_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total, 100. * correct_fused / total))

    return val_loss / (b_idx + 1),  correct_sub1 / total,  correct_sub2 / total,  correct_fused / total

def train(model, module, epoch):
    torch.autograd.set_detect_anomaly(True)
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.train()
    module.train()

    train_loss = 0
    correct_sub1 = 0 
    correct_sub2 = 0
    correct_fused = 0

    total = 0

    global optimizer
    global optimizer_FM


    consistency_weight = get_current_consistency_weight(epoch)
    train_loss_save = open('train_loss.txt', 'a')
    train_loss_save.write('epoch: {}\n'.format(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        optimizer_FM.zero_grad()
        batch_size = inputs.shape[0]

        #自监督输入、标签
        size = inputs.shape[1:]
        inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1).view(-1, *size)
        labels = torch.stack([targets * 4 + i for i in range(4)], 1).view(-1) #联合标签
        # 原始数据
        nor_index = (torch.arange(4 * batch_size) % 4 == 0).cuda()

        # 增强数据
        aug_index = (torch.arange(4 * batch_size) % 4 != 0).cuda()
        ###################################################################################
        
        outputs1, outputs2, crossFusionKnowledge, fmap,(net1_2_output,net2_2_output)= model(inputs)
        rep = net2_2_output
        # print('skjdh')
        nor_rep_ = rep[nor_index]
        aug_rep_ = rep[aug_index]

        nor_rep = nor_rep_.unsqueeze(2).expand(-1, -1, 3 * batch_size).transpose(0, 2)
        aug_rep = aug_rep_.unsqueeze(2).expand(-1, -1, 1 * batch_size)
        # 论文里的A矩阵
        simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)  #余弦相似矩阵

        sim_target = torch.arange(batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
        loss_sim  = F.cross_entropy(simi, sim_target)  #simi={Tensor:(384,128)}  sim_target={Tensor:(384,)}




        fused_logit = module(fmap[0].unsqueeze(-1).unsqueeze(-1).clone(), fmap[1].unsqueeze(-1).unsqueeze(-1).clone())  #融合模块函数
        #fused_logit = module(fmap[0], fmap[1])

        #自监督Loss
        # 2表示2-layer
        loss_net1_2 = criterion_CE(net1_2_output, labels) #net1_2_output={Tensor:(512,400)} labels={Tensor:(512,)}
        # loss_net2_2 = criterion_CE(net2_2_output, labels)

        # for i in range(len(ss_logits)):
        #     loss_ss = criterion_CE(ss_logits[i], labels)


        loss_cross = criterion_CE(outputs1, targets) + criterion_CE(outputs2, targets) + criterion_CE(fused_logit, targets) #交叉熵损失


        # MKD
        # net1和net2的逐层对齐
        # loss_knowledge = utils.lossOfKnowledge(criterion_kl, crossFusionKnowledge)
        loss_knowledge=0
        # 输出logit之前的最后一层的对齐
        loss_kl = consistency_weight*(criterion_kl(outputs1, outputs2) + criterion_kl(outputs2, outputs1)
                                     + criterion_kl(outputs1, fused_logit) + criterion_kl(outputs2, fused_logit) +  criterion_kl(fused_logit, outputs1) + criterion_kl(fused_logit, outputs2))






        loss = loss_cross + loss_kl + loss_net1_2  + 5 * loss_sim

        # loss = loss_cross + loss_kl+ loss_knowledge + loss_net1_2 + loss_net2_2
        ###################################################################################

        loss.backward()


        optimizer.step()
        optimizer_FM.step()
        #train_loss += loss.item()
        train_loss=train_loss.clone()+loss.item()

        #模型预测结果
        _, predicted_sub1 = torch.max(outputs1.data, 1)
        _, predicted_sub2 = torch.max(outputs2.data, 1)
        _, predicted_fused = torch.max(fused_logit.data, 1)

        #total += targets.size(0)  #计算总样本数
        total = total.clone() + targets.size(0)

        #计算预测正确的样本数
        #correct_sub1 += predicted_sub1.eq(targets.data).cpu().sum().float().item()
        #correct_sub2 += predicted_sub2.eq(targets.data).cpu().sum().float().item()
        #correct_fused += predicted_fused.eq(targets.data).cpu().sum().float().item()
        correct_sub1 = correct_sub1.clone() + predicted_sub1.eq(targets.data).cpu().sum().float().item()
        correct_sub2 = correct_sub2.clone() + predicted_sub2.eq(targets.data).cpu().sum().float().item()
        correct_fused = correct_fused.clone() + predicted_fused.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx
        train_loss_save.write('batch_index: %d | loss_kl: %.3f | loss_ce: %.3f | loss_knowledge: %.3f\n' % (b_idx, loss_kl, loss_cross, loss_knowledge))
    train_loss_save.write('\n')
    train_loss_save.close()
    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc sub-1: %.3f%% | Acc sub-2: %.3f%% | Acc fused: %.3f%% |' % (train_loss / (b_idx + 1), 100. * correct_sub1 / total, 100. * correct_sub2 / total, 100. * correct_fused / total))

    #返回平均损失和准确率
    return train_loss / (b_idx + 1), correct_fused / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d_%H:%M')
    if int(args.log_time) :
        folder_name = 'CrossKD_{}'.format(time_log)


    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    #logger = SummaryLogger(path)



    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss,  acc  = train(model, module, epoch)
        scheduler.step()
        # scheduler_FM.step()

        ### Evaluate FFL  ###
        val_loss, accuracy_sub1, accuracy_sub2, accuracy_fused = eval(model, module, epoch, test_flag=True)

        max_acc1 = 0
        max_acc2 = 0
        max_accS = 0
        if (max_acc1 < accuracy_sub1):
            max_acc1 = accuracy_sub1
            os.system('rm ckpt/{}/acc1Max*'.format(path))
            utils.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, True, 'ckpt/' + path, filename='acc1Max_{}.pth'.format(epoch))
        if (max_acc2 < accuracy_sub2):
            max_acc2 = accuracy_sub2
            os.system('rm ckpt/{}/acc2Max*'.format(path))
            utils.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, True, 'ckpt/' + path, filename='acc2Max_{}.pth'.format(epoch))
        if (max_accS < accuracy_sub1 + accuracy_sub2):
            max_accS = accuracy_sub1 + accuracy_sub2
            os.system('rm ckpt/{}/accSummaryMax*'.format(path))

            utils.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, True, 'ckpt/' + path, filename='accSummaryMax_{}.pth'.format(epoch))

        '''utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': module.state_dict(),
                    'optimizer' : optimizer_FM.state_dict(),
                }, True, 'ckpt/' + path, filename='Module_{}.pth'.format(epoch))
        '''


        f.write('EPOCH {epoch} \t'
                'ACC_sub-1 : {acc_sub1:.4f} \t ACC_sub-2 : {acc_sub2:.4f}\t' 
                'ACC_fused : {acc_fused:.4f} \t \n'.format(
                    epoch=epoch, acc_sub1=accuracy_sub1, acc_sub2=accuracy_sub2, acc_fused=accuracy_fused)
                )
        f.close()

