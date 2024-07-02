# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
from thop import clever_format, profile
from torch import nn

from nets.mobilevit import mobilevit_s
from nets.yolo import YoloBody, BiFPN_Concat2

if __name__ == "__main__":
    input_shape = [512, 512]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors_mask    = [[9,10,11],[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 4
    phi = 'l'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(anchors_mask, num_classes, phi, False, phi_attention=0, DCN_fpn=False, backbone=0,
                 focus=False, sod=False,head=0, DCN_backbone=False, MPCA_backbone=True, MPCA_fpn=True, BiFPN=False,
                 SE_concat=False, MPCBAM_fpn=False, MPCBAM_backbone=False, DW=False, SA_concat=False).to(device)
    # m = nn.Sequential(BiFPN_Concat2().to(device))
    # m = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3).to(device)
    # m = mobilevit_s()
    for i in m.children():
        print(i)
        print('==============================')

    # dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    # dummy_input=[dummy_input,dummy_input]
    flops, params = profile(m.to(device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

"""
原版yolo7
Total GFLOPS: 67.305G
Total params: 37.211M


MPGAM
Total GFLOPS: 83.187G
Total params: 45.900M

MPSGE
Total GFLOPS: 75.147G
Total params: 44.143M


MPCA
Total GFLOPS: 75.176G
Total params: 44.241M


MPCBAM
Total GFLOPS: 75.283G
Total params: 46.650M


"""
