"""
    复现Med-former医学图像分类模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

#   图像输入部分Patch Partitioning + Linear Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用 Conv2d 实现 patch 划分 + flatten + embedding 映射
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: patch_tokens [B, N_patches, embed_dim]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)  # [B, embed_dim, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, embed_dim]
        return x

#   LGT模块中左部分的W-MSA子模块
class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, "Input sequence length doesn't match H*W"
        shortcut = x  # for residual

        # 1. Norm & reshape to 2D
        x = self.norm(x).reshape(B, H, W, C)

        # 2. Padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # (B, H_pad, W_pad, C)
        Hp, Wp = x.shape[1], x.shape[2]

        # 3. Partition into windows
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)  # [num_win*B, Ws*Ws, C]

        # 4. Attention
        attn_out, _ = self.attn(x, x, x)  # [num_win*B, Ws*Ws, C]

        # 5. Merge windows
        x = attn_out.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # 6. Crop to original size
        x = x[:, :H, :W, :].contiguous().view(B, N, C)

        # 7. Residual connection
        return shortcut + x
#   LGT模块中右部分的SW-MSA子模块
class ShiftedWindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x

        x = self.norm(x).reshape(B, H, W, C)

        # padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # ★ Step 1: Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Step 2: Partition windows
        x_windows = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # Step 3: Attention
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)

        # Step 4: Merge
        x = attn_out.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # ★ Step 5: Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # crop + flatten
        x = x[:, :H, :W, :].contiguous().view(B, N, C)

        return shortcut + x  # residual


class LGTBlockLeft(nn.Module):
    def __init__(self, dim=768, num_heads=6, window_size_global=7, window_size_local=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.global_path = WindowAttentionBlock(dim, window_size_global, num_heads)
        self.local_path = WindowAttentionBlock(dim, window_size_local, num_heads)

        self.fusion_norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W):
        # 两条路径分别处理
        out_g = self.global_path(x, H, W)  # Global
        out_l = self.local_path(x, H, W)   # Local

        # 融合（简单平均）
        fused = (out_g + out_l) / 2

        # MLP + 残差
        mlp_out = self.mlp(self.fusion_norm(fused))
        out = fused + mlp_out
        return out

class LGTBlockRight(nn.Module):
    def __init__(self, dim=768, num_heads=6, window_size_global=7, window_size_local=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.global_path = ShiftedWindowAttentionBlock(dim, window_size_global, shift_size=window_size_global // 2, num_heads=num_heads)
        self.local_path = ShiftedWindowAttentionBlock(dim, window_size_local, shift_size=window_size_local // 2, num_heads=num_heads)

        self.fusion_norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W):
        # 双路径：SW-MSA（滑动窗口注意力）
        out_g = self.global_path(x, H, W)
        out_l = self.local_path(x, H, W)

        fused = (out_g + out_l) / 2

        # LN + MLP + Add
        mlp_out = self.mlp(self.fusion_norm(fused))
        out = fused + mlp_out
        return out

#   完整的LGT模块
class LGTBlock(nn.Module):
    def __init__(self, dim=768, num_heads=6,
                 window_size_global=7, window_size_local=3,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.left = LGTBlockLeft(
            dim=dim,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            mlp_ratio=mlp_ratio,
            drop=drop
        )
        self.right = LGTBlockRight(
            dim=dim,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

    def forward(self, x, H, W):
        x = self.left(x, H, W)
        x = self.right(x, H, W)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(input_dim * 4, out_dim)

    def forward(self, x, H, W):
        """
        x: [B, H*W, C]
        """
        B, N, C = x.shape
        assert N == H * W, "Input shape mismatch"

        x = x.view(B, H, W, C)

        # 🔧 Step 1: padding if H or W is odd
        pad_h = (H % 2 != 0)
        pad_w = (W % 2 != 0)
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, int(pad_w), 0, int(pad_h)))  # pad W, H
            H += int(pad_h)
            W += int(pad_w)

        # Step 2: extract 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 0::2, 1::2, :]  # top-right
        x2 = x[:, 1::2, 0::2, :]  # bottom-left
        x3 = x[:, 1::2, 1::2, :]  # bottom-right

        # Step 3: concat along channel
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]

        H_new, W_new = x_cat.shape[1], x_cat.shape[2]

        # Step 4: flatten & project
        x_flat = x_cat.view(B, H_new * W_new, 4 * C)
        x_out = self.reduction(x_flat)  # Linear(4C → out_dim)

        return x_out, H_new, W_new


class SAFModule(nn.Module):
    def __init__(self, dim_input_A, dim_input_B):
        """
        dim_input_A: f_A 的输入维度（例如来自 Encoding Phase，768）
        dim_input_B: f_B 的输入维度（例如来自当前阶段输出，1536）
        """
        super().__init__()

        # 将 f_A 映射到与 f_B 相同的通道数
        self.downsample = nn.Linear(dim_input_A, dim_input_B)

        # 分别对 f_A 和 f_B 做空间注意力图
        self.spatial_attn_a = nn.Sequential(
            nn.Conv2d(dim_input_B, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.spatial_attn_b = nn.Sequential(
            nn.Conv2d(dim_input_B, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_A, f_B, H_A, W_A, H_B, W_B):
        """
        f_A: [B, N1, dim_input_A] ← 较浅特征
        f_B: [B, N2, dim_input_B] ← 当前阶段输出
        H_A, W_A: f_A 的空间尺寸（如 14×14）
        H_B, W_B: f_B 的空间尺寸（如 7×7）
        """
        B, _, _ = f_A.shape

        # 1️⃣ 将 f_A 线性映射到 B 的维度
        f_A_proj = self.downsample(f_A)  # [B, N1, dim_B]

        # 2️⃣ reshape 成 2D map：[B, C, H, W]
        f_A_map = f_A_proj.transpose(1, 2).reshape(B, -1, H_A, W_A)  # [B, dim_B, H_A, W_A]
        f_B_map = f_B.transpose(1, 2).reshape(B, -1, H_B, W_B)

        # 3️⃣ 插值将 f_A_map 下采样到 f_B 的空间大小
        f_A_down = F.interpolate(f_A_map, size=(H_B, W_B), mode='bilinear', align_corners=False)

        # 4️⃣ 分别计算空间注意力图
        attn_A = self.spatial_attn_a(f_A_down)  # [B, 1, H_B, W_B]
        attn_B = self.spatial_attn_b(f_B_map)   # [B, 1, H_B, W_B]

        # 5️⃣ 对应位置相乘加权
        f_A_weighted = (f_A_down * attn_A).reshape(B, -1, H_B * W_B).transpose(1, 2)  # [B, N2, dim_B]
        f_B_weighted = (f_B_map * attn_B).reshape(B, -1, H_B * W_B).transpose(1, 2)

        # 6️⃣ 融合：逐位置相加
        fused = f_A_weighted + f_B_weighted  # [B, N2, dim_B]

        return fused

#   论文中的encoding phase
class EncodingPhase(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768,
                 num_heads=6, window_size_global=7, window_size_local=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.lgt = LGTBlock(
            dim=embed_dim,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local
        )

        self.grid_size = img_size // patch_size

    def forward(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        x = self.lgt(x, H=self.grid_size, W=self.grid_size)
        return x, self.grid_size, self.grid_size

#   论文中的stage0
class Stage0(nn.Module):
    def __init__(self, in_dim=768, out_dim=1536, num_heads=6,
                 window_size_global=7, window_size_local=3,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()

        # 1. Patch Merging: 将 H×W/4 patch 合并成一个，并升维
        self.patch_merging = PatchMerging(input_dim=in_dim, out_dim=out_dim)

        # 2. 单个 LGT Block（左+右注意力）
        self.lgt_block = LGTBlock(
            dim=out_dim,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

        # 3. SAF 模块：融合 Encoding 阶段输出和当前阶段输出
        self.saf = SAFModule(
            dim_input_A=in_dim,   # 如768
            dim_input_B=out_dim   # 如1536
        )

    def forward(self, x_prev, H_prev, W_prev):
        """
        输入:
            x_prev: 来自 EncodingPhase，shape [B, N1, in_dim]
            H_prev, W_prev: 原始 patch grid 尺寸（如 14 x 14）
        输出:
            x_fused: [B, N2, out_dim]
            H, W: 新的空间尺寸（H_prev/2, W_prev/2）
        """

        # 1️⃣ Patch Merging
        x_merged, H, W = self.patch_merging(x_prev, H_prev, W_prev)  # [B, N2, out_dim]

        # 2️⃣ LGT Block
        x_lgt = self.lgt_block(x_merged, H, W)  # [B, N2, out_dim]

        # 3️⃣ SAF 融合前一阶段信息
        x_fused = self.saf(x_prev, x_lgt, H_prev, W_prev, H, W)  # [B, N2, out_dim]

        return x_fused, H, W

#   论文中的stage1
class Stage1(nn.Module):
    def __init__(self, in_dim=1536, out_dim=3072, num_heads=6,
                 window_size_global=7, window_size_local=3,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()

        # 1️⃣ Patch Merging
        self.patch_merging = PatchMerging(input_dim=in_dim, out_dim=out_dim)

        # 2️⃣ 单个 LGT Block（左+右 注意力路径）
        self.lgt_block = LGTBlock(
            dim=out_dim,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

        # 3️⃣ SAF 融合 Stage0 输出
        self.saf = SAFModule(
            dim_input_A=in_dim,
            dim_input_B=out_dim
        )

    def forward(self, x_prev, H_prev, W_prev):
        """
        x_prev: Stage0 输出 [B, N1, in_dim]
        返回: SAF 融合后输出 [B, N2, out_dim]
        """
        # Patch Merging
        x_merged, H, W = self.patch_merging(x_prev, H_prev, W_prev)

        # 单层 LGT
        x_lgt = self.lgt_block(x_merged, H, W)

        # SAF 融合 Stage0 输出
        x_fused = self.saf(x_prev, x_lgt, H_prev, W_prev, H, W)

        return x_fused, H, W

#   MLP分类头
class MLPHead(nn.Module):
    def __init__(self, in_dim=4096, num_classes=2, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 1, C]
        return: [B, num_classes]
        """
        x = x.squeeze(1)  # [B, C]
        out = self.head(x)  # [B, num_classes]
        return out

class MedFormerBackbone(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=1,
                 dim_enc=768,
                 dim_s0=1536,
                 dim_s1=3072,
                 num_heads=8,
                 num_classes=2,  # 可以支持多分类
                 window_size_global=7,
                 window_size_local=3,
                 drop=0.0):
        super().__init__()

        # 编码阶段
        self.encoder = EncodingPhase(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim_enc,
            window_size_global=window_size_global,
            window_size_local=window_size_local
        )

        # Stage0
        self.stage0 = Stage0(
            in_dim=dim_enc,
            out_dim=dim_s0,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            drop=drop
        )

        # Stage1
        self.stage1 = Stage1(
            in_dim=dim_s0,
            out_dim=dim_s1,
            num_heads=num_heads,
            window_size_global=window_size_global,
            window_size_local=window_size_local,
            drop=drop
        )

        # 分类头：接收 Stage1 输出
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim_s1),
            nn.Dropout(drop),
            nn.Linear(dim_s1, num_classes)  # 输出 num_classes 个类别
        )

    def forward(self, x):
        x_enc, H_enc, W_enc = self.encoder(x)
        x_s0, H_s0, W_s0 = self.stage0(x_enc, H_enc, W_enc)
        x_s1, H_s1, W_s1 = self.stage1(x_s0, H_s0, W_s0)

        # 获取分类特征
        x_cls = x_s1.mean(dim=1)  # [B, 3072]

        # 最后通过一个线性层将输出维度转换为类别数
        logits = self.cls_head(x_cls)  # [B, num_classes]

        return logits

if __name__ == "__main__":
    model = MedFormerBackbone(num_classes=2, num_heads=8)
    img = torch.randn(1, 1, 224, 224)
    logits = model(img)
    print("预测logitis:", logits)  # 打印预测标签



