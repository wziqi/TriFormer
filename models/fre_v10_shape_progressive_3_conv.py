import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, Mlp, DropPath, lecun_normal_


class Net(nn.Module):
    def __init__(self, angRes, upscale_factor, channels, blocks_num=3):
        super(Net, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.dim = channels
        self.factor = upscale_factor
        self.blocks_num = blocks_num

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################

        self.Ang_Spa_block = self.Make_Layer(layer_num=self.blocks_num, ang_spa=True)

        self.EPI_block = self.Make_Layer(layer_num=self.blocks_num, ang_spa=False)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(3 * channels, channels * self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def Make_Layer(self, layer_num, ang_spa=True, ):
        layers = []
        for i in range(layer_num):
            if ang_spa:
                layers.append(BlockFilter(angRes=self.angRes, dim=self.dim, transtype=2))
                layers.append(BlockFilter(angRes=self.angRes, dim=self.dim, transtype=1))
            else:
                layers.append(BlockFilter(angRes=self.angRes, dim=self.dim, transtype=3))
                layers.append(BlockFilter(angRes=self.angRes, dim=self.dim, transtype=4))

        return nn.Sequential(*layers)

    def forward(self, lr, info=None):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)

        # Initial Convolution
        buffer = self.conv_init0(lr)
        buffer_init = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        buffer_Ang_Spa_out = self.Ang_Spa_block(buffer_init) + buffer_init

        buffer_EPI_out = self.EPI_block(buffer_Ang_Spa_out) + buffer_Ang_Spa_out

        buffer_all = torch.cat((buffer_init, buffer_Ang_Spa_out, buffer_EPI_out), dim=1)
        buffer_all = rearrange(buffer_all, 'b c a h w -> b a c h w')
        # buffer_out = self.Reconstruct(buffer_all

        # Up-Sampling
        buffer_out = rearrange(buffer_all, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        buffer_out = self.upsampling(buffer_out)
        out = buffer_out + lr_upscale

        return out


def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor,
                                                                      W * scale_factor)  # [B, 1, A*h*S, A*w*S]

    return x_upscale


class BlockFilter(nn.Module):
    def __init__(self, angRes, dim, transtype, drop=0.):
        super(BlockFilter, self).__init__()
        self.angRes = angRes
        self.dim =dim
        self.transtype = transtype
        self.patch_embed = PatchEmbed(img_size=224, patch_size=4, in_chans=0, embed_dim=dim, norm_layer=None,
                                      transtype=transtype)
        self.pos_drop = nn.Dropout(p=drop)
        self.block_layers = nn.ModuleList([
            Block(angRes=self.angRes, dim=self.dim, transtype=self.transtype),
            Block(angRes=self.angRes, dim=self.dim, transtype=self.transtype),
            Block(angRes=self.angRes, dim=self.dim, transtype=self.transtype)
        ])
        if transtype == 1:
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def check_image_size(self, x, transtype):
        b, c, a, h, w = x.size()

        if transtype == 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * a, c, h, w)
        elif transtype == 2:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h w) c a1 a2', a1=self.angRes, a2=self.angRes)
        elif transtype == 3:
            x = rearrange(x, 'b c (a1 a2) h w -> (b h a1) c w a2', a1=self.angRes, a2=self.angRes)
        elif transtype == 4:
            x = rearrange(x, 'b c (a1 a2) h w -> (b w a2) c h a1', a1=self.angRes, a2=self.angRes)

        return x

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):

        b, c, a, h, w = x.shape
        x = self.check_image_size(x, transtype=self.transtype)
        x = self.pos_drop(x)
        x1 = x
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        x_size = (h, w)
        for layer in self.block_layers:
            x = layer(x, x_size)

        x = x.permute(0, 3, 1, 2).contiguous()
        if self.transtype == 1:
            x = self.conv(x) + x1

        if self.transtype == 2:
            x = rearrange(x, '(b h w) c a1 a2 -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=h, w=w)
        if self.transtype == 3:
            x = rearrange(x, '(b h a1) c w a2 -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=h, w=w)
        if self.transtype == 4:
            x = rearrange(x, '(b w a2) c h a1 -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes, h=h, w=w)

        x = x.view(b, a, c, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class PA(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        if kernel_size == 3:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, groups=dim)
        elif kernel_size == 5:
            self.pa_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=2, groups=dim)

        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, with_pos=True,
                 transtype=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.with_pos = with_pos

        if transtype == 1:
            self.pos = PA(embed_dim, kernel_size=3)
        else:
            self.pos = PA(embed_dim, kernel_size=5)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.with_pos:
            x = self.pos(x)
        # x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Block(nn.Module):

    def __init__(self, angRes=5, dim=60, transtype=1, num_heads=10, mlp_ratio=2., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1
                 ):
        super(Block, self).__init__()
        self.angRes = angRes
        self.transtype = transtype
        self.norm1 = norm_layer(dim)
        self.attn = Mixer(angRes=angRes,dim=dim, transtype=transtype, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         attention_head=attention_head)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        x = x + self.drop_path(self.attn(self.norm1(x), x_size))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)
    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class FreBlock(nn.Module):
    def __init__(self, angRes, dim, transtype):
        super(FreBlock, self).__init__()
        self.angRes = angRes
        self.transtype = transtype
        self.dim = dim

        self.qkv = ComplexLinear(self.dim, self.dim * 3)
        self.fre_conv = nn.Conv2d(dim, dim, 1, 1, 0)

    def complex_matmul(self, a, b):
        """复数矩阵乘法."""
        ar = a.real
        br, bi = b.real, b.imag
        return torch.complex((ar @ br ), (ar @ bi))

    def exchange_low_frequency(self, frequency, center_fraction, x_size):
        # frequency = rearrange(frequency, '(b a1 a2) c h w -> b (a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        B, c, h, w = frequency.shape
        if self.transtype == 1:
            b = int(B // (self.angRes * self.angRes))
        elif self.transtype == 2:
            b = int(B // (x_size[0] * x_size[1]))
        else:
            b = int(B // (x_size[0] * self.angRes))

        center_h = int(h * center_fraction)
        center_w = int(w * center_fraction)
        mask = torch.zeros((h, w))
        start_h = (h - center_h) // 2
        start_w = (w - center_w) // 2
        mask[start_h:start_h + center_h, start_w:start_w + center_w] = 1
        mask = mask.to(frequency.device)

        low_frequency = (frequency * mask)
        # print(low_frequency.shape)  torch.Size([1, 25, 6, 32, 32])
        low_fre = (low_frequency[:, :, start_h:start_h + center_h, start_w:start_w + center_w])
        low_fre = rearrange(low_fre, '(b s) c h w -> b (h w s) c', h=center_h, w=center_w, b=b, s=B // b)
        # print(low_frequency.shape)  torch.Size([1, 25, 216])
        # qkv = self.qkv(low_frequency).reshape(b, -1, 3, self.angRes, self.angRes).permute(2, 0, 1, 3, 4)
        # qkv = self.qkv(low_fre).reshape(b, -1, 3, (B // b) * center_h * center_w, self.dim // 3).permute(2, 0, 1, 3, 4)
        qkv = self.qkv(low_fre).reshape(b, -1, 3, (B // b) * center_h * center_w, 1).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        # print(q.shape)  torch.Size([1, 216, 5, 5])
        # print(q.shape)  torch.Size([1, 2, 4096, 3])
        attn_logits = q @ k.transpose(-2, -1)
        attn_weights = F.softmax(attn_logits.abs(), dim=-1)
        attn_outputs = self.complex_matmul(attn_weights, v)

        low_fre = attn_outputs.view(b, -1, self.dim) + low_fre
        low_fre = low_fre.reshape(B, c, center_h, center_w)

        frequency[:, :, start_h:start_h + center_h, start_w:start_w + center_w] = low_fre
        # frequency = self.fre_conv(frequency)
        # frequency = rearrange(frequency, ' b (a1 a2) c h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        return frequency

    def forward(self, frequency, x_size):
        # h, w = x_size
        _, _, h, w = frequency.shape

        out = torch.fft.fft2(frequency + 1e-8, norm='backward')
        out = self.exchange_low_frequency(out, 0.2, x_size)
        out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))
        out = self.fre_conv(out)
        out = out + frequency
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)

        return out


class Mixer(nn.Module):
    def __init__(self, angRes, dim, transtype, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 attention_head=1,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        self.low_dim = low_dim = attention_head * head_dim
        self.fre_dim = fre_dim = self.low_dim
        self.high_dim = high_dim = dim - low_dim - low_dim

        self.fre_block = FreBlock(angRes=angRes, dim=fre_dim, transtype=transtype)

        self.high_mixer = HighMixer(angRes, high_dim, transtype)
        self.low_mixer = LowMixer(angRes, low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop)

        # self.attn_fuse_1 = AsyCA(num_features=low_dim + high_dim * 2, ratio=6)
        # self.attn_fuse_2 = AsyCA(num_features=(high_dim + fre_dim) * 2, ratio=6)
        self.conv_fuse_1 = nn.Conv2d(fre_dim + low_dim, fre_dim + low_dim, kernel_size=3, stride=1, padding=1,
                                     bias=False, groups=fre_dim + low_dim)
        self.conv_fuse_2 = nn.Conv2d((high_dim + fre_dim) * 2, (high_dim + fre_dim) * 2, kernel_size=3, stride=1, padding=1,
                                     bias=False, groups=(high_dim + fre_dim) * 2)
        self.proj = nn.Conv2d((high_dim + fre_dim) * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size):
        # B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim : self.high_dim + self.low_dim, :, :].contiguous()
        lx = self.low_mixer(lx)

        fre_x = x[:, self.high_dim + self.low_dim:, :, :].contiguous()
        fre_x = self.fre_block(fre_x, x_size)

        buffer_x = torch.cat((hx, lx, fre_x), dim=1)
        # buffer_x = buffer_x + self.attn_fuse(hx, lx, fre_x)
        coarse_x = self.conv_fuse_1(torch.cat((lx, fre_x), dim=1))
        x = buffer_x + self.conv_fuse_2(torch.cat((hx, coarse_x), dim=1))

        # x = buffer_x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class HighMixer(nn.Module):
    def __init__(self, angRes, dim, transtype, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.angRes = angRes
        self.transtype = transtype

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)

        return hx


class LowMixer(nn.Module):
    def __init__(self, angRes, dim, num_heads=6, qkv_bias=False, attn_drop=0.,):
        super().__init__()
        self.angRes = angRes
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        B, C, h, w = x.shape
        xa = x.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, C // num_head
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, h, w)

        return xa


def count_parameters(models):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)

def print_submodule_parameters(model, prefix=""):
    for name, submodule in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{full_name} 的参数量: {count_parameters(submodule)}")
        print_submodule_parameters(submodule, full_name)

if __name__ == "__main__":
    net = Net(5, 2, 60).cuda()
    print(net)
    from thop import profile

    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    # Number of parameters: 1.18M
    #
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
    # Number of FLOPs: 29.89G
    #
    for index, sub_module in enumerate(net.Ang_Spa_block[0].block_layers[0].attn.children()):
        print(f'        sub_module {index+1} params: { (count_parameters(sub_module) / 1e6)}')
    # print_submodule_parameters(net.Ang_Spa_block)



