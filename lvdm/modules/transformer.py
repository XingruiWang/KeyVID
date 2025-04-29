import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def cyclic_shift(x, shifts, dims):
    #x: B, n_contrast, H, W, C
    n = x.shape[1]
    shifted_x = []
    for i in range(n):
        shifted_x_i = torch.roll(x[:, i, :, :, :], shifts=shifts, dims=dims)
        shifted_x.append(shifted_x_i.unsqueeze(1))
    shifted_x = torch.cat(shifted_x, 1)
    return shifted_x




def window_partition(x, window_size):
    """
    Args:
        x: (B, n_contrast, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, n_contrast, H, W, C = x.shape
    windows = []
    for i in range(n_contrast):
        x_i = x[:, i, :, :, :]
        x_i = x_i.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        window_i = x_i.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, 1, window_size[0], window_size[1], C)  #(num_windows*B, 1, window_size, window_size, C)
        windows.append(window_i)
    windows = torch.cat(windows, 1) #(num_windows*B, n_contrast, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, n_contrast, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, n_contrast, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = []
    n_contrast = windows.shape[1]
    for i in range(n_contrast):
        window = windows[:, i, :, :, :]
        x_i = window.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x_i = x_i.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 1, H, W, -1)
        x.append(x_i)
    x = torch.cat(x, 1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_k, x_v, mask=None, return_attention=False):
        """
        Args:
            x: input features with shape of (num_windows*B, n_contrast, window_size, window_size, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, n_contrast_q, Wh, Ww, C = x_q.shape
        n_contrast_kv = x_k.shape[1]

        x_q = x_q.view(-1, Wh * Ww * n_contrast_q, C)   #B*nW, Wh * Ww * n_contrast_q, C
        x_k = x_k.view(-1, Wh * Ww * n_contrast_kv, C)
        x_v = x_v.view(-1, Wh * Ww * n_contrast_kv, C)

        B_, N_q, C = x_q.shape
        _, N_kv, C = x_k.shape

        q = self.q(x_q).reshape(B_, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = self.k(x_k).reshape(B_, N_kv, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = self.v(x_v).reshape(B_, N_kv, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # (1, B_, nH, N, C//n_head)
        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N_q, N_kv)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        # pdb.set_trace()   # check if the shape of relative position bias matches attn
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().repeat(1, n_contrast_q, n_contrast_kv)   # nH, Wh*Ww*c_q, Wh*Ww*c_kv
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: # nW, window_size**2, window_size**2
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.repeat(1, n_contrast_q, n_contrast_kv).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)  #B_, N_q, nH, c => B_, N_q, C
        x = self.proj(x)
        x = self.proj_drop(x).view(B_, n_contrast_q, Wh, Ww, C)
        return x, attn if return_attention else None

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        #H, W = self.input_resolution
        B, n_contrast, H, W, C = x.shape
        #assert L == H * W * n_contrast, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_merged = []
        for i in range(n_contrast):
            x0 = x[:, i, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, i, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, i, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, i, 1::2, 1::2, :]  # B H/2 W/2 C
            x_merged_i = torch.cat([x0, x1, x2, x3], -1).unsqueeze(1) # B 1 H/2 W/2 4*C
            x_merged.append(x_merged_i)
        x_merged = torch.cat(x_merged, 1) # B n_contrast H/2 W/2  4*C

        x = self.norm(x_merged)   # B n_contrast H/2 W/2  4*C
        x = self.reduction(x)  # B n_contrast H/2 W/2  2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        B, n_contrast, H, W, C = x.shape
        x = self.expand(x)    #B, n_contrast, H, W, 2C

        x = rearrange(x, 'b n h w (p1 p2 c)-> b n (h p1) (w p2) c', n=n_contrast, p1=2, p2=2, c=C//2)
        x= self.norm(x)   #B, n_contrast, 2*H, 2*W, C//2

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        # x = self.expand(x)
        n_contrast = x.shape[1]
        C = x.shape[-1]
        x = rearrange(x, 'b n h w (p1 p2 c)-> b n (h p1) (w p2) c', n=n_contrast,
                      p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = self.norm(x)  # B, n_contrast, H*dim_scale, W*dim_scale, C//dim_scale**2

        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(160, 192), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = img_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)  # B Ph Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops