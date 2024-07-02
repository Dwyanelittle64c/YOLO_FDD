import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.attention import se_block, cbam_block, eca_block, ca_block, SpatialAttention
from nets.head import YOLOXHead_new, YOLO_Decouple_Head, YOLOXHead_lite, YOLOXHead_lite_s
from nets.mobilenet_v3 import mobilenetv3_large
from nets.mobilevit import mobilevit_s

attentions_block=[se_block,cbam_block,eca_block,ca_block]

from nets.backbone import Backbone, Multi_Concat_Block, Conv, SiLU, Transition_Block, autopad, EfficientNet, BaseConv


# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支add操作
# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.register_parameter("weight_in_BiFPNcat2", self.w)
        # self.w = torch.ones(2, dtype=torch.float32)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.register_parameter("weight_in_BiFPNcat3", self.w)

        # self.w = torch.ones(3, dtype=torch.float32)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        # m1 = self.m[0](x1)
        # m2 = self.m[1](x1)
        # m3 = self.m[2](x1)
        # m0=[m(x1) for m in self.m]
        # mm=[x1] + [m(x1) for m in self.m]
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11  = autopad(k, p) - k // 2
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std     = (bn.running_var + bn.eps).sqrt()
        bias    = bn.bias - bn.running_mean * bn.weight / std

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        
        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None
            
def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False, phi_attention=0, DCN_fpn=False, backbone=0,
                 focus=False, sod=False,head=0, DCN_backbone=False, MPCA_backbone=False, MPCA_fpn=False, BiFPN=False,
                 SE_concat=False, MPCBAM_fpn=False, MPCBAM_backbone=False, DW=False, SA_concat=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7版本的参数
        #-----------------------------------------------#
        transition_channels = {'l' : 32, 'x' : 40}[phi]
        block_channels      = 32
        panet_channels      = {'l' : 32, 'x' : 64}[phi]
        e       = {'l' : 2, 'x' : 1}[phi]
        n       = {'l' : 4, 'x' : 6}[phi]
        ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]
        conv    = {'l' : RepConv, 'x' : Conv}[phi]

        self.sod=sod
        self.SE_concat=SE_concat
        self.SA_concat=SA_concat
        self.phi_attention=phi_attention
        assert 0<=phi_attention<=5
        assert 0<=backbone<=3
        assert 0<=head<=4
        # 注意力机制
        if phi_attention>0:
            self.feat1_attention=attentions_block[phi_attention-1](512)
            self.feat2_attention=attentions_block[phi_attention-1](1024)
            self.feat3_attention=attentions_block[phi_attention-1](1024)
            self.P3_attention=attentions_block[phi_attention-1](128)
            self.P4_attention=attentions_block[phi_attention-1](256)
            self.P5_attention=attentions_block[phi_attention-1](512)

            if sod:
                self.feat0_attention = attentions_block[phi_attention - 1](256)
                self.P0_attention = attentions_block[phi_attention - 1](64)


            # 额外2p
            # self.P1_attention=attentions_block[phi_attention-1](256)
            # self.P2_attention=attentions_block[phi_attention-1](128)

        self.BiFPN=BiFPN


        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：

        #   128,128,256

        #   64, 64, 512
        #   32, 32, 1024
        #   16, 16, 1024
        #---------------------------------------------------#


        if backbone==0:
            self.backbone   = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained, focus=focus,
                                       sod=sod,DCN_backbone=DCN_backbone,MPCA_backbone=MPCA_backbone,MPCBAM_backbone=MPCBAM_backbone,DW=False)
        elif backbone==1:
            self.backbone   = EfficientNet(2)
        elif backbone==2:
            self.backbone   = mobilenetv3_large()
        elif backbone==3:
            self.backbone   = mobilevit_s(sod=sod)


        if BiFPN:
            self.concat0=BiFPN_Concat2()
            self.concat1 = BiFPN_Concat2()
            self.conv_for_bifpn_p4 = nn.Conv2d(256 * 3, 256 * 2, kernel_size=1, padding=0)
            if sod:
                self.concat2=BiFPN_Concat2()
                self.concat3=BiFPN_Concat3()
                self.conv_for_bifpn_p3=nn.Conv2d(128*3,128*2,kernel_size=1,padding=0)
            else:
                self.concat3=BiFPN_Concat2()

            self.concat4=BiFPN_Concat3()
            self.concat5=BiFPN_Concat2()

        if SE_concat:
            self.conv_for_P52P4=nn.Conv2d(transition_channels * 16,transition_channels * 8,kernel_size=3,padding=1)
            self.conv_for_P42P3=nn.Conv2d(transition_channels * 8,transition_channels * 4,kernel_size=3,padding=1)
            self.se_for_P52P4=se_block(transition_channels * 8, return_channel=True)
            self.se_for_P42P3=se_block(transition_channels * 4, return_channel=True)
            if sod:
                self.conv_for_P32P0 = nn.Conv2d(transition_channels * 4, transition_channels * 2, kernel_size=3,
                                                padding=1)
                self.se_for_P32P0 = se_block(transition_channels * 2, return_channel=True)


        if SA_concat:
            self.SA_for_P3=SpatialAttention()
            self.SA_for_P4=SpatialAttention()
            self.ap = nn.AvgPool2d(2)

        #------------------------加强特征提取网络------------------------#
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")



        # 20, 20, 1024 => 20, 20, 512
        self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        # 40, 40, 1024 => 40, 40, 256
        self.conv_for_feat2         = Conv(transition_channels * 32, transition_channels * 8)
        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, DCN=DCN_fpn,DW=DW)



        # 40, 40, 256 => 40, 40, 128 => 80, 80, 128
        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        # 80, 80, 512 => 80, 80, 128
        self.conv_for_feat1         = Conv(transition_channels * 16, transition_channels * 4)

        # 80, 80, 256 => 80, 80, 128
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2,
                                                      transition_channels * 4, e=e, n=n, ids=ids, DCN=DCN_fpn,DW=DW)

        if sod:
            # 160, 160, 256 => 160, 160, 128
            self.conv_for_feat0     = Conv(transition_channels * 8, transition_channels * 2)


            # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
            self.conv_for_P3        = Conv(transition_channels * 4, transition_channels * 2)



            # 40, 40, 512 => 40, 40, 256
            self.conv3_for_upsample0= Multi_Concat_Block(transition_channels * 4, panet_channels * 1,
                                                         transition_channels * 2, e=e, n=n, ids=ids, DCN=DCN_fpn,DW=DW)
            # 80, 80, 128 => 80, 80, 256
            self.rep_conv_0         = conv(transition_channels * 2, transition_channels * 8, 3, 1)

            self.down_sample0       = Transition_Block(transition_channels * 2, transition_channels * 2,use_CA=MPCA_fpn,use_CBAM=MPCBAM_fpn,DW=DW)

            self.conv3_for_downsample0=Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids, DCN=DCN_fpn,DW=DW)




        # 80, 80, 128 => 40, 40, 256
        self.down_sample1           = Transition_Block(transition_channels * 4, transition_channels * 4,use_CA=MPCA_fpn,use_CBAM=MPCBAM_fpn,DW=DW)

        # 40, 40, 512 => 40, 40, 256
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids,DCN=False,DW=DW)

        # 40, 40, 256 => 20, 20, 512
        self.down_sample2           = Transition_Block(transition_channels * 8, transition_channels * 8,use_CA=MPCA_fpn,use_CBAM=MPCBAM_fpn,DW=DW)
        # 20, 20, 1024 => 20, 20, 512
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids,DCN=False,DW=DW)

        #------------------------加强特征提取网络------------------------#


        # 80, 80, 128 => 80, 80, 256
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        # 40, 40, 256 => 40, 40, 512
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        # 20, 20, 512 => 20, 20, 1024
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        # 4 + 1 + num_classes


        if head==1:

            if sod:
                self.yolo_head_P0=YOLOXHead_new(num_classes=num_classes,in_channel=transition_channels * 8)
            # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
            self.yolo_head_P3 = YOLOXHead_new(num_classes=num_classes,in_channel=transition_channels * 8)
            # 40, 40, 512 => 40, 40, 3 * 25 & 85
            self.yolo_head_P4 = YOLOXHead_new(num_classes=num_classes,in_channel=transition_channels * 16)
            # 20, 20, 512 => 20, 20, 3 * 25 & 85
            self.yolo_head_P5 = YOLOXHead_new(num_classes=num_classes,in_channel=transition_channels * 32)

        elif head==2:
            if sod:
                self.yolo_head_P0=YOLO_Decouple_Head(num_classes=num_classes,in_channel=transition_channels * 8)
            # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
            self.yolo_head_P3 = YOLO_Decouple_Head(num_classes=num_classes,in_channel=transition_channels * 8)
            # 40, 40, 512 => 40, 40, 3 * 25 & 85
            self.yolo_head_P4 = YOLO_Decouple_Head(num_classes=num_classes,in_channel=transition_channels * 16)
            # 20, 20, 512 => 20, 20, 3 * 25 & 85
            self.yolo_head_P5 = YOLO_Decouple_Head(num_classes=num_classes,in_channel=transition_channels * 32)

        elif head==3:
            if sod:
                self.yolo_head_P0=YOLOXHead_lite(num_classes=num_classes,in_channel=transition_channels * 8)
            # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
            self.yolo_head_P3 = YOLOXHead_lite(num_classes=num_classes,in_channel=transition_channels * 8)
            # 40, 40, 512 => 40, 40, 3 * 25 & 85
            self.yolo_head_P4 = YOLOXHead_lite(num_classes=num_classes,in_channel=transition_channels * 16)
            # 20, 20, 512 => 20, 20, 3 * 25 & 85
            self.yolo_head_P5 = YOLOXHead_lite(num_classes=num_classes,in_channel=transition_channels * 32)

        elif head==4:
            if sod:
                self.yolo_head_P0=YOLOXHead_lite_s(num_classes=num_classes,in_channel=transition_channels * 8)
            # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
            self.yolo_head_P3 = YOLOXHead_lite_s(num_classes=num_classes,in_channel=transition_channels * 8)
            # 40, 40, 512 => 40, 40, 3 * 25 & 85
            self.yolo_head_P4 = YOLOXHead_lite_s(num_classes=num_classes,in_channel=transition_channels * 16)
            # 20, 20, 512 => 20, 20, 3 * 25 & 85
            self.yolo_head_P5 = YOLOXHead_lite_s(num_classes=num_classes,in_channel=transition_channels * 32)


        else:# head = 0
            if sod:
                self.yolo_head_P0=nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
            # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
            self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
            # 40, 40, 512 => 40, 40, 3 * 25 & 85
            self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
            # 20, 20, 512 => 20, 20, 3 * 25 & 85
            self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)




    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):

        #  backbone
        if self.sod:
            feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        else:
            feat1, feat2, feat3 = self.backbone.forward(x)

        # backbone输出的三个特征层添加注意力
        if self.phi_attention>0:
            feat1=self.feat1_attention(feat1)
            feat2=self.feat2_attention(feat2)
            feat3=self.feat3_attention(feat3)
            if self.sod:
                feat0=self.feat0_attention(feat0)

        #------------------------加强特征提取网络------------------------# 
        # 20, 20, 1024 => 20, 20, 512
        P5          = self.sppcspc(feat3)

        if self.SE_concat:
            P5_se          = self.conv_for_P52P4(P5)

            P5_channel  = self.se_for_P52P4(P5_se)

        # 20, 20, 512 => 20, 20, 256
        P5_conv     = self.conv_for_P5(P5)
        # 20, 20, 256 => 40, 40, 256
        P5_upsample = self.upsample(P5_conv)


        P4          = self.conv_for_feat2(feat2)

        if self.SE_concat:
            # print(P4.shape)
            # print(P5_channel.shape)

            P4 = P4 * P5_channel
            P5_se = self.upsample(P5_se)

            P4 = P4 + P5_se
            # P4      = P4 * 0

            P4_se      = self.conv_for_P42P3(P4)
            P4_channel = self.se_for_P42P3(P4_se)



        if self.BiFPN:
            P4      = self.concat0([P4, P5_upsample])
        # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        else:
            P4      = torch.cat([P4, P5_upsample], 1)
        # 40, 40, 512 => 40, 40, 256
        P4          = self.conv3_for_upsample1(P4)



        # 40, 40, 256 => 40, 40, 128
        P4_conv     = self.conv_for_P4(P4)
        # 40, 40, 128 => 80, 80, 128
        P4_upsample = self.upsample(P4_conv)


        P3          = self.conv_for_feat1(feat1)
        if self.SE_concat:
            if self.sod:
                P3_se=self.conv_for_P32P0(P3)
                P3_channel=self.se_for_P32P0(P3_se)

            P4_se=self.upsample(P4_se)

            P3 = P3 * P4_channel

            P3 = P3 + P4_se
            # P3 = P3 * 0

        if self.BiFPN:
            P3      = self.concat1([P3, P4_upsample])
        else:
        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
            P3      = torch.cat([P3, P4_upsample], 1)
        # 80, 80, 256 => 80, 80, 128
        P3          = self.conv3_for_upsample2(P3)


        if self.sod:

            P3_conv=self.conv_for_P3(P3)

            P3_upsample=self.upsample(P3_conv)

            P0=self.conv_for_feat0(feat0)

            if self.SE_concat:
                P0 = P0 * P3_channel

            if self.BiFPN:
                P0=self.concat2([P0,P3_upsample])
            else:
                P0=torch.cat([P0,P3_upsample],dim=1)

            P0=self.conv3_for_upsample0(P0)


            if self.phi_attention>0:
                P0=self.P0_attention(P0)

            P0_downsample=self.down_sample0(P0)

            if self.BiFPN:
                # print(P3.shape)
                # print(P0_downsample.shape)
                # print(self.conv_for_feat1(feat1).shape)
                if self.sod:
                    P3 = self.concat3([P3,P0_downsample,self.conv_for_feat1(feat1)])
                    P3 = self.conv_for_bifpn_p3(P3)
                else:
                    P3 = self.concat3([P3, P0_downsample])
            else:
                P3=torch.cat([P3,P0_downsample],dim=1)
            P3=self.conv3_for_downsample0(P3)

            P0=self.rep_conv_0(P0)


        if self.SA_concat:
            P3_sa=self.ap(P3)
            P3_sa=self.SA_for_P3(P3_sa)



        # 特征融合网络输出的P3分支添加注意力机制
        if self.phi_attention>0:
            P3=self.P3_attention(P3)

                # 80, 80, 128 => 40, 40, 256
        P3_downsample = self.down_sample1(P3)
            # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512

        if self.BiFPN:
            P4 = self.concat4([P3_downsample, P4,self.conv_for_feat2(feat2)])
            # print(P4.shape)
            P4 = self.conv_for_bifpn_p4(P4)
        else:

            if self.SA_concat:
                P4 = P4 * P3_sa


                P4_sa = self.ap(P4)
                P4_sa = self.SA_for_P4(P4_sa)



            P4 = torch.cat([P3_downsample, P4], 1)
            # 40, 40, 512 => 40, 40, 256
        P4 = self.conv3_for_downsample1(P4)




        # 特征融合网络输出的P4分支添加注意力机制
        if self.phi_attention>0:
            P4=self.P4_attention(P4)



        # 40, 40, 256 => 20, 20, 512
        P4_downsample = self.down_sample2(P4)

        # 20, 20, 512 cat 20, 20, 512 => 20, 20, 1024
        if self.BiFPN:
            P5 = self.concat5([P4_downsample, P5],)
        else:

            if self.SA_concat:


                P5 = P5 * P4_sa


            P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 => 20, 20, 512
        P5 = self.conv3_for_downsample2(P5)
        # 特征融合网络输出的P5分支添加注意力机制
        if self.phi_attention>0:
            P5=self.P5_attention(P5)





        #------------------------加强特征提取网络------------------------#
        # P3 80, 80, 128
        # P4 40, 40, 256
        # P5 20, 20, 512




        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)


        if self.sod:
            out3 = self.yolo_head_P0(P0)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
            #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        if self.sod:

            return [out0, out1, out2, out3]
        else:
            return [out0, out1, out2]

if __name__ == '__main__':
    # x1=torch.rand(1,256,16,16)
    # x2=torch.rand(1,512,24,24)
    x=torch.rand(1,3,512,512).cuda()
    # anchors_mask    = [[9,10,11],[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    net=YoloBody(anchors_mask, 4,'l', phi_attention=0, DCN_fpn=False, backbone=0,
                 sod=False,DCN_backbone=False,BiFPN=False,head=0,
                 SE_concat=False,DW=False,SA_concat=False,
                 MPCA_fpn=True,MPCA_backbone=True).cuda()
    out=net(x)
    # out=Backbone(32, 32, 4, 'l',sod=True).cuda()(x)
    # out=YOLOXHead(num_classes=4)([x1,x2,x3])
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    # print(out[3].shape)
    # for i in list(net.named_parameters()):
    #     print(i)
    # print())