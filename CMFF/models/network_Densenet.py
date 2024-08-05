import os
import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Tuple, Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import math
import torch.nn.init as init
import numpy as np

##################################################### densenet #############################################
norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]



class CrossTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)


            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))


        return [rgb_fea_flat, ir_fea_flat]

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).view(b, c)
class CrossAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.LN3 = nn.LayerNorm(d_model)
        self.LN4 = nn.LayerNorm(d_model)
        self.atten1 = SELayer(512)
        self.atten2 = SELayer(512)
        self.final_1 = nn.Linear(1024,512)
        self.final_2 = nn.Linear(1024,512)

        self.mlp_vis = nn.Sequential(nn.Linear(688, 128),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(128, 688),
                                     nn.Dropout(0.1),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(688, 128),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(128, 688),
                                     nn.Dropout(0.1),
                                     )
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, x2, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        # rgb_fea_flat = x[0]
        # ir_fea_flat = x[1]
        # b_s, nq = rgb_fea_flat.shape[:2]
        # nk = rgb_fea_flat.shape[1]

        b_s = x.shape[0]

        # Self-Attention
        rgb_fea_flat = self.LN1(x)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, self.h, self.d_k)   # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, self.h, self.d_k)  # (b_s, h, nk, d_v)
        # print("hello")
        ir_fea_flat = self.LN2(x2)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis.transpose(-2, -1)) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir.transpose(-2, -1)) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).contiguous().view(b_s, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).contiguous().view(b_s,  self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)


        # out_vis = self.atten1(torch.cat((x, out_vis), dim=1))
        # out_ir = self.atten2(torch.cat((x2, out_ir), dim=1))

        # out_vis = self.atten1(out_vis)
        # out_ir = self.atten2(out_ir)
        out_vis = self.coefficient1(x) + self.coefficient2(out_vis)
        out_ir = self.coefficient3(x2) + self.coefficient4(out_ir)

        out_vis = self.coefficient5(out_vis) + self.coefficient6(self.mlp_vis(self.LN3(out_vis)))
        out_ir = self.coefficient7(out_ir) + self.coefficient8(self.mlp_ir(self.LN4(out_ir)))
        # out_vis = x + out_vis
        # out_ir = x2 + out_ir
        return [out_vis, out_ir]

class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)


        self.sptial = sptial


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        atmap = []
        input = torch.cat((x,y),1)   #特征融合

        # print("------------------------------"+ str(x.size())+"," + str(y.size())+ ","+ str(input.size()))


        x = F.relu(self.bn1((self.conv1(input))))
        # print("----------------1--------------"+ str(x.size()))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        # print("----------------2--------------"+ str(x.size()))

        atmap.append(x)
        # print("----------------3--------------"+ str(x.size()))
        x = F.avg_pool2d(x, self.sptial)
        # print("----------------4--------------"+ str(x.size()))
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out




class CrossKDDenseNet(nn.Module):
    def __init__(self, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=1000, growthRate=4, compressionRate=2,dropout = 0.0,):
        super(CrossKDDenseNet, self).__init__()
        self.total_feature_maps = {}


        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if depth==40 else (depth - 4) // 6
        transition = Transition

        self.covcfg=cov_cfg

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2



        self.conv1_1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False) # 3->24
        self.conv1_2 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False) # 3->24
        # self.dense1_1 = self._make_denseblock(block, n)
        # self.trans1_1 = self._make_transition(transition, compressionRate)
        # self.dense2_1 = self._make_denseblock(block, n)
        # self.trans2_1 = self._make_transition(transition, compressionRate)
        # self.dense3_1 = self._make_denseblock(block, n) #在cifar上只有三个block
        # self.trans3_1 = self._make_transition(transition, compressionRate)
        self.bn_1 = nn.BatchNorm2d(self.inplanes)
        self.bn_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(43, num_classes)
        self.fc2 = nn.Linear(129, num_classes)

        blocksSize = [(112,56), (56, 28), (28, 14)]
        channels = [28,38,43]
        # strides = [1, 2, 2, 2]
        self.net1_2_classifier = nn.Sequential(
            nn.Linear(2432, 1024),
            #nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 512),
            #nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(512, num_classes * 4),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(2432, 2432),
            #nn.ReLU(True),
            nn.Linear(2432, 2432),

        )
        self.cross_attention = CrossAttention(688, 128, 0, 4)
        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()


        for stage in range(3):
            bkplanes = self.inplanes
            self.net1Blocks.append(nn.Sequential(*[self._make_denseblock(block, n),self._make_transition(transition, compressionRate)]))
            self.inplanes = bkplanes
            self.net2Blocks.append(nn.Sequential(*[self._make_denseblock(block, n),self._make_transition(transition, compressionRate)]))
            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage + 1, 3):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1],
                                                              int(blocksSize[stage][1] / blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.reset_parameters()
        self.register_hook()

    def _make_denseblock(self, block, blocks):
        layers = []
        inplanes = self.inplanes
        for i in range(blocks):
            layers.append(
                block(inplanes + i * self.growthRate, outplanes=int(self.growthRate), dropRate=self.dropRate))

        self.inplanes = inplanes + blocks * self.growthRate  # 到下一层了
        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def reset_parameters(self):

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def register_hook(self):

        self.extract_layers = ['dense1', 'dense2', 'relu']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)



    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x1 = self.conv1_1(x)

        x2 = self.conv1_2(x)

        fmap = []
        crossFusionKnowledge = []
        net1Knowledge = []
        net2Knowledge = []
        for stage in range(3):
            x1 = self.net1Blocks[stage](x1)
            # print(x1.size())
            x2 = self.net2Blocks[stage](x2)


            temp1 = x1
            temp2 = x2
            for preNum in range(0, stage):
                temp1 = temp1 + net1Knowledge[preNum][stage - preNum - 1]
                temp2 = temp2 + net2Knowledge[preNum][stage - preNum - 1]
            crossFusionKnowledge.append((torch.flatten(temp1, 1), torch.flatten(temp2, 1)))

            stageKnowledge1 = []
            stageKnowledge2 = []
            for to in range(stage + 1, 3):
                stageKnowledge1.append(self.net1CrossNet[stage][to - stage - 1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to - stage - 1](x2))
            net1Knowledge.append(stageKnowledge1)
            net2Knowledge.append(stageKnowledge2)
        if self.training:
            x1 = x1[0::4]
            x2 = x2[0::4]
        x1_, x2_ = self.cross_attention(x1.view(x1.size(0), -1),x2.view(x1.size(0), -1))


        fmap.append(x1_)
        fmap.append(x2_)

        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        '''
        # print(x1.size())
        x1 = self.relu(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)

        x2 = self.relu(x2)
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)

        net1_2 = crossFusionKnowledge[1][0]
        net2_2 = crossFusionKnowledge[1][1]

        net1_2_output = self.net1_2_classifier(net1_2)
        net2_2_output = self.proj_head(net2_2)

        return x1, x2, crossFusionKnowledge, fmap, (net1_2_output,net2_2_output)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List, List]:
        return self._forward_impl(x)
