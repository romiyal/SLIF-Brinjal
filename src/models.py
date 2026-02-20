import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0,
                 use_norm: bool = True, use_act: bool = True, act_layer: nn.Module = nn.SiLU, bias: bool = False,
                 groups: int = 1, dilation: int = 1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias,
                      groups=groups, dilation=dilation)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DilatedBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        if stride == 1:
            padding = dilation
        elif stride == 2:
            padding = dilation
        else:
            raise ValueError(f"Unsupported stride: {stride}")

        self.use_res = (in_ch == out_ch and stride == 1)

        self.block = nn.Sequential(
            ConvLayer(in_ch, mid_ch, kernel_size=1),
            ConvLayer(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=padding,
                      groups=mid_ch, dilation=dilation),
            ConvLayer(mid_ch, out_ch, kernel_size=1, use_act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res and x.shape == out.shape:
            out = out + x
        return out

class LocalRepresentationBlock(nn.Module):
    def __init__(self, Cin: int, TransformerDim: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(Cin, Cin, kernel_size=3, padding=1, groups=Cin)
        self.pointwise_conv = nn.Conv2d(Cin, TransformerDim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, attn_dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.qkv_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p_dim, n, d = x.shape
        x_reshaped_for_conv = rearrange(x, 'b p_dim n d -> b d p_dim n')
        qkv = self.qkv_proj(x_reshaped_for_conv)

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        intermediate_output = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(intermediate_output)

        out = rearrange(out, 'b d p_dim n -> b p_dim n d')
        return out

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearSelfAttention(embed_dim=dim, attn_dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h
        return y

class AffixAttentionBlock(nn.Module):
    def __init__(self, Cin: int, TransformerDim: int, Cout: int, depth: int = 2, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.TransformerDim = TransformerDim
        self.Cin = Cin

        self.local = LocalRepresentationBlock(Cin, TransformerDim)

        mlp_dim = TransformerDim * 2
        dropout = 0.25
        self.transformer = Transformer(
            dim=TransformerDim,
            depth=depth,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        self.conv_proj = nn.Conv2d(TransformerDim, Cin, kernel_size=1)
        self.coord_att = CoordinateAttention(inp=Cin, oup=Cin)
        self.fusion = nn.Conv2d(2 * Cin, Cout, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        res = x

        ph = pw = self.patch_size
        h_patches = H // ph
        w_patches = W // pw

        local_out = self.local(x)

        if C < self.TransformerDim:
            padding_channels = torch.zeros(B, self.TransformerDim - C, H, W, device=x.device, dtype=x.dtype)
            res_n_padded = torch.cat([x, padding_channels], dim=1)
        else:
            res_n_padded = x[:, :self.TransformerDim, :, :]

        local_patches = rearrange(local_out, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                                 ph=ph, pw=pw, h=h_patches, w=w_patches)
        res_patches = rearrange(res_n_padded, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                               ph=ph, pw=pw, h=h_patches, w=w_patches)

        seq = local_patches + res_patches
        seq_t = self.transformer(seq)

        fm = rearrange(seq_t, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      ph=ph, pw=pw, h=h_patches, w=w_patches)

        projected = self.conv_proj(fm)
        attention = self.coord_att(res)
        concat = torch.cat([attention, projected], dim=1)
        out = self.fusion(concat)

        return out + res

class UltraLightBlockNet_L1(nn.Module):
    def __init__(self, num_classes: int, image_size: int, dims: list[int], channels: list[int]):
        super().__init__()
        self.stem = nn.Sequential(
            ConvLayer(3, channels[0], kernel_size=3, stride=2, padding=1)
        )
        expansion = 2
        self.stage1 = nn.Sequential(
            DilatedBottleneck(channels[0], expansion * channels[0], channels[0], stride=2),
            DilatedBottleneck(channels[0], expansion * channels[0], channels[1], stride=2),
            DilatedBottleneck(channels[1], expansion * channels[1], channels[1], stride=1)
        )
        self.stage2 = nn.Sequential(
            DilatedBottleneck(channels[1], expansion * channels[1], channels[2], stride=2, dilation=2),
            AffixAttentionBlock(Cin=channels[2], TransformerDim=dims[0], Cout=channels[2], depth=2)
        )
        self.stage3 = nn.Sequential(
            DilatedBottleneck(channels[2], expansion * channels[2], channels[3], stride=1, dilation=4),
            DilatedBottleneck(channels[3], expansion * channels[3], channels[3], stride=1),
            AffixAttentionBlock(Cin=channels[3], TransformerDim=dims[1], Cout=channels[3], depth=3)
        )
        self.head = nn.Sequential(
            ConvLayer(channels[3], channels[4], kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels[4], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return self.classifier(x)
