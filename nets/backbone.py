import torch
import torch.nn as nn


from nets.attention import ca_block, cbam_block, ca_block_for_MPCA, se_block, SpatialGroupEnhance, gam_block, \
    cbam_block_for_MP
from nets.efficientnet import EfficientNet as EffNet


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class DWConv(nn.Module):
    def __init__(self,inp,oup,k=3,stride=1,p=None):
        super(DWConv, self).__init__()
        self.dwconv=nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

    def forward(self,x):
        x = self.dwconv(x)

        return x



class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# 用于focus的BaseConv
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class DConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DConv, self).__init__()
        self.dconv = DeformConv2d(c1, c2, k, s, autopad(k, p))
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = act
    def forward(self, x):
        return self.act(self.bn(self.dconv(x)))

    def fuseforward(self, x):
        return self.act(self.dconv(x))

class BottleNeck_DCN(nn.Module):
    def __init__(self, c1, c_, c2,  k=3, s=1, p=None, act=SiLU(), attention=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(BottleNeck_DCN, self).__init__()
        self.cv1 = nn.Conv2d(c1,c_,1)
        self.cv2 = nn.Conv2d(c_,c2,1)
        self.dconv = DeformConv2d(c_, c_, k, s, autopad(k, p))
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c_)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = act
        self.attention=attention
        if attention:
            self.attention_block=se_block(c_)
    def forward(self, x):


        # se new
        # residual  = x
        # x=self.act(self.bn1(self.cv1(x)))
        #
        #
        #
        # x=self.bn2(self.dconv(x))
        #
        # if self.attention:
        #     x=self.attention_block(x)
        #
        # x=self.bn3(self.cv2(x))
        #
        #
        #
        # return self.act(x + residual)



        # se
        residual  = x
        x=self.act(self.bn1(self.cv1(x)))

        if self.attention:
            x=self.attention_block(x)

        x=self.act(self.bn2(self.dconv(x)))


        x=self.bn3(self.cv2(x))



        return self.act(x + residual)


# 可融合DCN或DW的Multi_Concat_Block
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0], DCN=False, DW=False):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        assert (DCN and DW) == False
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        if DW:
            self.cv3 = nn.ModuleList(
                [DWConv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
            )
        else:
            self.cv3 = nn.ModuleList(
                [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
            )
        if DCN:
            self.cv4 = DConv(c_ * 2 + c2 * (len(ids) - 2), c3, 3, 1)
        else:
            self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        # [-1, -3, -5, -6] => [5, 3, 1, 0]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Transition_Block(nn.Module):
    def __init__(self, c1, c2, use_CA=False,use_CBAM=False,DW=False):
        super(Transition_Block, self).__init__()
        assert (use_CA and use_CBAM) is not True
        self.use_CA=use_CA
        self.use_CBAM=use_CBAM
        self.cv1 = Conv(c1, c2, 1, 1)

        if use_CA:
            self.cv2 = cbam_block_for_MP(c1,c2)
            self.cv3 = Conv(c2, c2, 3, 2)
            self.cv4 = Conv(c1, c2, 3, 2)

        elif use_CBAM:
            self.cv2 = cbam_block(c1)
            self.cv3 = Conv(c1, c2, 3, 2)
            self.cv4 = Conv(c1, c2, 3, 2)

        else:
            self.cv2 = Conv(c1, c2, 1, 1)
            if DW:
                self.cv3 = DWConv(c2, c2, 3, 2)
            else:
                self.cv3 = Conv(c2, c2, 3, 2)

        self.mp = MP()

    def forward(self, x):
        # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        # 160, 160, 256 => 160, 160, 128 => 80, 80, 128

        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        if self.use_CA or self.use_CBAM:
            x_2 = x_2 + self.cv4(x)




        # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        return torch.cat([x_2, x_1], 1)
        # return x_2 + x_1


class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False, focus=False, sod=False,
                 DCN_backbone=False,MPCA_backbone=False,MPCBAM_backbone=False,DW=False,bottleneck_attention=True):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#
        ids = {
            'l': [-1, -3, -5, -6],
            'x': [-1, -3, -5, -7, -8],
        }[phi]
        # 640, 640, 3 => 640, 640, 32 => 320, 320, 64
        # if focus:
        #     self.stem = nn.Sequential(
        #         Focus(3,transition_channels * 2,ksize=3,act='silu'),
        #         Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        #     )
        # else:
        #     self.stem = nn.Sequential(
        #         Conv(3, transition_channels, 3, 1),
        #         Conv(transition_channels, transition_channels * 2, 3, 2),
        #         Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        #     )
        self.sod = sod
        if focus:
            self.stem = nn.Sequential(
                Focus(3, transition_channels * 2, ksize=3, act='silu'),
                # Conv(transition_channels, transition_channels * 2, 3, 2),
                MP(k=2),
                Conv(transition_channels * 2, transition_channels * 2, 3, 1),

            )
        else:
            if DW:
                self.stem = nn.Sequential(
                    DWConv(3, transition_channels, 3, 1),
                    DWConv(transition_channels, transition_channels * 2, 3, 2),
                    DWConv(transition_channels * 2, transition_channels * 2, 3, 1),
                )
            else:
                self.stem = nn.Sequential(
                    Conv(3, transition_channels, 3, 1),
                    Conv(transition_channels, transition_channels * 2, 3, 2),
                    Conv(transition_channels * 2, transition_channels * 2, 3, 1),
                )



        # dark2
        # 320, 320, 64 => 160, 160, 128 => 160, 160, 256
        if DW:
            self.dark2 = nn.Sequential(
                DWConv(transition_channels * 2, transition_channels * 4, 3, 2),
                Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids,DW=DW),
            )
        elif DCN_backbone:
            self.dark2 = nn.Sequential(
                Conv(transition_channels * 2, transition_channels * 4, 3, 2),
                Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids,DW=DW),
                # BottleNeck_DCN(transition_channels * 8, transition_channels * 4, transition_channels * 8,attention=bottleneck_attention),

            )

        else:
            self.dark2 = nn.Sequential(
                Conv(transition_channels * 2, transition_channels * 4, 3, 2),
                Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids,DW=DW),
            )



        # dark3
        if DCN_backbone:
            # 160, 160, 256 => 80, 80, 256 => 80, 80, 512
            self.dark3 = nn.Sequential(
                Transition_Block(transition_channels * 8, transition_channels * 4, use_CA=MPCA_backbone,
                                 use_CBAM=MPCBAM_backbone, DW=DW),
                Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids,
                                   DW=DW,DCN=False),
                BottleNeck_DCN(transition_channels * 16,transition_channels * 8,transition_channels * 16,attention=bottleneck_attention),
                # DConv(transition_channels * 16, transition_channels * 16, 3, 1),

            )
        else:
            # 160, 160, 256 => 80, 80, 256 => 80, 80, 512
            self.dark3 = nn.Sequential(
                Transition_Block(transition_channels * 8, transition_channels * 4,use_CA=MPCA_backbone,use_CBAM=MPCBAM_backbone,DW=DW),
                Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids,DW=DW),
            )


        # dark4
        if DCN_backbone:
            # 80, 80, 512 => 40, 40, 512 => 40, 40, 1024
            self.dark4 = nn.Sequential(
                Transition_Block(transition_channels * 16, transition_channels * 8, use_CA=MPCA_backbone,
                                 use_CBAM=MPCBAM_backbone, DW=DW),
                Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids,
                                   DCN=False, DW=DW,),
                BottleNeck_DCN(transition_channels * 32,transition_channels * 16,transition_channels * 32,attention=bottleneck_attention),
                # DConv(transition_channels * 32, transition_channels * 32, 3, 1),

            )
        else:
            # 80, 80, 512 => 40, 40, 512 => 40, 40, 1024
            self.dark4 = nn.Sequential(
                Transition_Block(transition_channels * 16, transition_channels * 8,use_CA=MPCA_backbone,use_CBAM=MPCBAM_backbone,DW=DW),
                Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids,DCN=False,DW=DW),
            )


        # dark5
        if DCN_backbone:
            # 40, 40, 1024 => 20, 20, 1024 => 20, 20, 1024
            self.dark5 = nn.Sequential(
                Transition_Block(transition_channels * 32, transition_channels * 16, use_CA=MPCA_backbone,
                                 use_CBAM=MPCBAM_backbone, DW=DW),
                Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids,
                                   DCN=False, DW=DW),
                BottleNeck_DCN(transition_channels * 32,transition_channels * 16,transition_channels * 32,attention=bottleneck_attention),


            )
        else:
            # 40, 40, 1024 => 20, 20, 1024 => 20, 20, 1024
            self.dark5 = nn.Sequential(
                Transition_Block(transition_channels * 32, transition_channels * 16,use_CA=MPCA_backbone,use_CBAM=MPCBAM_backbone,DW=DW),
                Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids,DCN=False,DW=DW),
            )

        if pretrained:
            url = {
                "l": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        # -----------------------------------------------#
        #   dark2的输出为160, 160, 128，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark2(x)
        feat0 = x
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x

        if self.sod:
            return feat0, feat1, feat2, feat3
        else:
            return feat1, feat2, feat3


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model
        self.conv_feat1 = Conv(48, 512)
        self.conv_feat2 = Conv(120, 1024)
        self.conv_feat3 = Conv(352, 1024)

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        out_feats = [self.conv_feat1(feature_maps[2]), self.conv_feat2(feature_maps[3]),
                     self.conv_feat3(feature_maps[4])]
        return out_feats


if __name__ == '__main__':
    x = torch.rand(1, 3, 160, 160)
    out = Backbone(32, 32, 4, 'l', focus=False,DCN_backbone=False,MPCBAM_backbone=True,DW=False)(x)
    # out = Transition_Block(c1=256,c2=128,use_CBAM=True,use_CA=False,DW=False)(x)
    # out = DWConv(3,64)(x)
    # print(out.shape)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
# 160, 160, 256 => 80, 80, 256 => 80, 80, 128