'''
VGG16 for CIFAR-10/100 Dataset.
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

__all__ = ['cross_vgg16', 'cross_vgg19']




#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}


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

        self.mlp_vis = nn.Sequential(nn.Linear(512, 2 * 512),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(2 * 512, 512),
                                     nn.Dropout(0.1),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(512, 2 * 512),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(2 * 512, 512),
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

        ir_fea_flat = self.LN2(x2)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, self.h, self.d_k) # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis.transpose(-2, -1)) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir.transpose(-2, -1)) / np.sqrt(self.d_k)


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



        out_vis = self.coefficient1(x) + self.coefficient2(out_vis)
        out_ir = self.coefficient3(x2) + self.coefficient4(out_ir)

        out_vis = self.coefficient5(out_vis) + self.coefficient6(self.mlp_vis(self.LN3(out_vis)))
        out_ir = self.coefficient7(out_ir) + self.coefficient8(self.mlp_ir(self.LN4(out_ir)))

        return [out_vis, out_ir]





#原来的融合模块
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


class CrossVGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, dropout = 0.0, KD= False):
        super(CrossVGG, self).__init__()
        self.KD = KD
        self.inplances = 64
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # self.auxiliary_classifier = Auxiliary_Classifier(cfg, batch_norm=batch_norm, num_classes=num_classes * 4)
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        blocksSize = [(16, 8), (8, 4), (4, 2), (2, 1)]
        channels = [128, 256, 512, 512]
        self.net1Blocks = nn.ModuleList()
        self.net1CrossNet = nn.ModuleList()
        self.net2Blocks = nn.ModuleList()
        self.net2CrossNet = nn.ModuleList()
        for stage in range(4):
            bkplances = self.inplances
            self.net1Blocks.append(self._make_layers(channels[stage], num_layer))
            self.inplances = bkplances
            self.net2Blocks.append(self._make_layers(channels[stage], num_layer))
            stageCrossNet1 = nn.ModuleList()
            stageCrossNet2 = nn.ModuleList()
            for to in range(stage+1, 4):
                stageCrossNet1.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
                stageCrossNet2.append(self._make_fusion_layer(channels[stage], channels[to], blocksSize[stage][1], int(blocksSize[stage][1]/blocksSize[to][1])))
            self.net1CrossNet.append(stageCrossNet1)
            self.net2CrossNet.append(stageCrossNet2)      
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
        )

        self.net1_2_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes * 4),
        )

        self.net2_2_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(512, num_classes * 4),
        )



        self.proj_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),

        )

        self.cross_attention = CrossAttention(512,128,0,4)


        self.LN3 = nn.LayerNorm(512)
        self.LN4 = nn.LayerNorm(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _make_fusion_layer(self, in_planes, out_planes, in_size, minification):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, minification, minification, padding=0, bias=False))
        # layers.append(nn.AvgPool2d(minification, minification))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)


        fmap = []
        crossFusionKnowledge = []
        net1Knowledge = []
        net2Knowledge = []

        for stage in range(4):
            x1 = self.net1Blocks[stage](x1)
            x2 = self.net2Blocks[stage](x2)
            
            temp1 = x1
            temp2 = x2
            for preNum in range(0, stage):
                temp1 = temp1 + net1Knowledge[preNum][stage-preNum-1]
                temp2 = temp2 + net2Knowledge[preNum][stage-preNum-1]
            crossFusionKnowledge.append((torch.flatten(temp1,1), torch.flatten(temp2,1)))  #进行知识融合

            stageKnowledge1 = []
            stageKnowledge2 = []
            for to in range(stage+1, 4):
                stageKnowledge1.append(self.net1CrossNet[stage][to-stage-1](x1))
                stageKnowledge2.append(self.net2CrossNet[stage][to-stage-1](x2))
            net1Knowledge.append(stageKnowledge1)
            net2Knowledge.append(stageKnowledge2)
        if self.training:
            x1 = x1[0::4]
            x2 = x2[0::4]
        x1_, x2_ = self.cross_attention(x1.view(x1.size(0), -1),x2.view(x1.size(0), -1))


        # x1 = x1 + self.mlp_vis(self.LN3(x1))
        # x2 = x2 + self.mlp_ir(self.LN4(x2)
        fmap.append(x1_)
        fmap.append(x2_)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier(x1)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.classifier(x2)


        net1_2 =  crossFusionKnowledge[2][0]
        net2_2 =  crossFusionKnowledge[2][1]

        net1_2_output = self.net1_2_classifier(net1_2)
        # net2_2_output = self.net2_2_classifier(net2_2)

        net2_2_output = self.proj_head(net2_2)
        return x1, x2, crossFusionKnowledge, fmap, (net1_2_output,net2_2_output)
    
def cross_vgg16(pretrained=False, path=None, **kwargs):
    """
    Constructs a CrossVGG16 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = CrossVGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def cross_vgg19(pretrained=False, path=None, **kwargs):
    """
    Constructs a CrossVGG19 model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = CrossVGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


