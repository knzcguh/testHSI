import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from sklearn.decomposition import PCA

# class MultiScaleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleConv, self).__init__()
#
#         # 不同尺度的卷积分支
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#         self.conv5x5 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         # 多尺度特征提取
#         feat_1x1 = self.conv1x1(x)
#         feat_3x3 = self.conv3x3(x)
#         feat_5x5 = self.conv5x5(x)
#
#         # 特征融合
#         concat_features = torch.cat([feat_1x1, feat_3x3, feat_5x5], dim=1)
#
#         return self.fusion(concat_features)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        # 确保out_channels能被3整除
        self.out_channels = out_channels // 3 * 3

        # 不同尺度的卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels // 3, kernel_size=1),
            nn.BatchNorm2d(self.out_channels // 3),
            nn.ReLU()
        )

        # 3x3卷积的输入通道数需要考虑前一层的输出
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels + self.out_channels // 3, self.out_channels // 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels // 3),
            nn.ReLU()
        )

        # 5x5卷积的输入通道数需要考虑前两层的输出
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * (self.out_channels // 3), self.out_channels // 3, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.out_channels // 3),
            nn.ReLU()
        )

        # 最终投影层，确保输出通道数正确
        self.proj = nn.Sequential(
            nn.Conv2d(self.out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # 1x1卷积
        out1 = self.conv1x1(x)
        # 连接原始输入和1x1卷积的输出
        concat1 = torch.cat([x, out1], dim=1)

        # 3x3卷积
        out2 = self.conv3x3(concat1)
        # 连接原始输入、1x1卷积输出和3x3卷积输出
        concat2 = torch.cat([x, out1, out2], dim=1)

        # 5x5卷积
        out3 = self.conv5x5(concat2)

        # 只连接三个卷积的输出
        out = torch.cat([out1, out2, out3], dim=1)

        # 通过投影层调整通道数
        return self.proj(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=3, dim_head=64, dropout=0.0, attention_type='spatial'):
        super(MultiHeadAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attention_type = attention_type

        # QKV投影
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 空间卷积注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=3, padding=1, groups=heads),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
            nn.Conv2d(heads, heads, kernel_size=3, padding=1, groups=heads),
            nn.BatchNorm2d(heads),
            nn.ReLU()
        )

        # 光谱卷积注意力
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(heads, heads, kernel_size=3, padding=1, groups=heads),
            nn.BatchNorm2d(heads),
            nn.ReLU(),
            nn.Conv2d(heads, heads, kernel_size=3, padding=1, groups=heads),
            nn.BatchNorm2d(heads),
            nn.ReLU()
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(heads, heads // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(heads // 2, heads, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # QKV投影
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, h, n, n]

        if self.attention_type == 'spatial':
            # 应用空间注意力
            # 将注意力图重塑为2D形式进行卷积
            attn_2d = dots.permute(0, 1, 2, 3).contiguous()  # [b, h, n, n]
            attn_2d = self.spatial_conv(attn_2d)

            # 应用通道注意力
            channel_weights = self.channel_attention(attn_2d)
            attn_2d = attn_2d * channel_weights

            # 重塑回原始形状
            dots = attn_2d
        else:
            # 应用光谱注意力
            attn_2d = dots.permute(0, 1, 2, 3).contiguous()  # [b, h, n, n]
            attn_2d = self.spectral_conv(attn_2d)
            dots = attn_2d

        # 应用softmax
        attn = F.softmax(dots, dim=-1)

        # 应用注意力到值向量
        out = torch.matmul(attn, v)

        # 重排并合并多头
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=3, dim_head=64, mlp_dim=256, dropout=0.0, attention_type='spatial'):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout, attention_type)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=3, heads=3, dim_head=64, mlp_dim=256, dropout=0.0, attention_type='spatial'):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, heads, dim_head, mlp_dim, dropout, attention_type)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PCALayer(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.fitted = False
        self.pca = None
        self.register_buffer('components', None)
        self.register_buffer('mean', None)

    def fit(self, x):
        if not self.fitted:
            B, C, H, W = x.shape
            # 将数据移到CPU并分离计算图
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()

            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(x_reshaped)
            self.fitted = True

            # 将PCA参数转换为张量并注册为缓冲区
            device = x.device
            self.register_buffer('components',
                                 torch.FloatTensor(self.pca.components_).to(device))
            self.register_buffer('mean',
                                 torch.FloatTensor(self.pca.mean_).to(device))

    def forward(self, x):
        if not self.fitted:
            self.fit(x)

        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)

        # 确保在同一设备上
        if x_reshaped.device != self.mean.device:
            self.mean = self.mean.to(x_reshaped.device)
            self.components = self.components.to(x_reshaped.device)

        # 使用注册的缓冲区进行PCA变换
        with torch.no_grad():
            x_centered = x_reshaped - self.mean
            x_transformed = torch.matmul(x_centered, self.components.t())

        # 重塑回原始维度
        x_pca = x_transformed.reshape(B, H, W, self.n_components)
        x_pca = x_pca.permute(0, 3, 1, 2)

        return x_pca

class OptimizedHSI3DTransformer(nn.Module):
    def __init__(self, input_channels, pca_channels, num_classes, patch_size=7, embedding_dim=128,
                 spatial_depth=2, spectral_depth=2, heads=4, dim_head=32,
                 mlp_dim=256, dropout=0.1):
        super(OptimizedHSI3DTransformer, self).__init__()

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.seq_length = patch_size * patch_size

        # PCA层
        self.pca_layer = PCALayer(n_components=pca_channels)

        # 添加输入归一化
        self.input_norm = nn.BatchNorm2d(input_channels)

        # 空间分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(pca_channels, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout
            MultiScaleConv(embedding_dim, embedding_dim)
        )

        # 初始化位置编码时使用较小的标准差
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.seq_length, embedding_dim) * 0.02)

        self.spatial_transformer = TransformerEncoder(
            dim=embedding_dim,
            depth=spatial_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_type='spatial'
        )

        # 光谱分支 - 使用原始输入通道数
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(input_channels, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            MultiScaleConv(embedding_dim, embedding_dim)
        )
        self.spectral_pos_embed = nn.Parameter(torch.randn(1, self.seq_length, embedding_dim))
        self.spectral_transformer = TransformerEncoder(
            dim=embedding_dim,
            depth=spectral_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_type='spectral'
        )
        # 特征融合前的规范化层
        self.spectral_norm = nn.LayerNorm(embedding_dim)
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )


    def _forward_spatial(self, x):
        B, C, H, W = x.shape

        # 多尺度特征提取
        spatial_features = self.spatial_conv(x)  # [B, embedding_dim, H, W]

        # 调整维度顺序以适应LayerNorm
        spatial_features = spatial_features.permute(0, 2, 3, 1)  # [B, H, W, embedding_dim]
        spatial_features = self.spatial_norm(spatial_features)  # 应用LayerNorm
        spatial_features = self.dropout(spatial_features)

        # Transformer处理
        spatial_tokens = rearrange(spatial_features, 'b h w c -> b (h w) c')
        # 确保序列长度匹配
        if spatial_tokens.size(1) != self.seq_length:
            spatial_tokens = F.adaptive_avg_pool1d(
                spatial_tokens.transpose(1, 2),
                self.seq_length
            ).transpose(1, 2)

        spatial_tokens = spatial_tokens + self.spatial_pos_embed
        spatial_encoded = self.spatial_transformer(spatial_tokens)

        # 全局特征
        spatial_avg = torch.mean(spatial_encoded, dim=1)
        spatial_max, _ = torch.max(spatial_encoded, dim=1)
        return torch.cat([spatial_avg, spatial_max], dim=1)

    def _forward_spectral(self, x):
        # 多尺度特征提取
        spectral_features = self.spectral_conv(x)  # [B, embedding_dim, H, W]

        # 调整维度顺序以适应LayerNorm
        spectral_features = spectral_features.permute(0, 2, 3, 1)  # [B, H, W, embedding_dim]
        spectral_features = self.spectral_norm(spectral_features)  # 应用LayerNorm
        spectral_features = self.dropout(spectral_features)

        # Transformer处理
        spectral_tokens = rearrange(spectral_features, 'b h w c -> b (h w) c')
        # 确保序列长度匹配
        if spectral_tokens.size(1) != self.seq_length:
            spectral_tokens = F.adaptive_avg_pool1d(
                spectral_tokens.transpose(1, 2),
                self.seq_length
            ).transpose(1, 2)

        spectral_tokens = spectral_tokens + self.spectral_pos_embed
        spectral_encoded = self.spectral_transformer(spectral_tokens)

        # 全局特征
        spectral_avg = torch.mean(spectral_encoded, dim=1)
        spectral_max, _ = torch.max(spectral_encoded, dim=1)
        return torch.cat([spectral_avg, spectral_max], dim=1)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            # 输入归一化
            x = self.input_norm(x)

            # 梯度裁剪
            x = torch.clamp(x, -10, 10)

            # PCA降维
            x_pca = self.pca_layer(x)
            x_pca = torch.clamp(x_pca, -10, 10)  # 防止数值不稳定

            # 特征提取
            spatial_features = self._forward_spatial(x_pca)
            spectral_features = self._forward_spectral(x)

            # 特征融合前的归一化
            spatial_features = F.normalize(spatial_features, p=2, dim=1)
            spectral_features = F.normalize(spectral_features, p=2, dim=1)

            # 特征融合
            fused_features = torch.cat([spatial_features, spectral_features], dim=1)
            fused_features = self.fusion(fused_features)

            # 分类
            output = self.classifier(fused_features)

        return output


# class MultiScaleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleConv, self).__init__()
#         # 确保out_channels能被3整除
#         self.out_channels = (out_channels // 3) * 3
#
#         self.conv1x1 = nn.Conv2d(in_channels, self.out_channels // 3, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(in_channels + self.out_channels // 3, self.out_channels // 3, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(in_channels + 2 * (self.out_channels // 3), self.out_channels // 3, kernel_size=5,
#                                  padding=2)
#
#         # 添加一个投影层来确保输出维度正确
#         self.proj = nn.Conv2d(self.out_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         out1 = self.conv1x1(x)
#         concat1 = torch.cat([x, out1], dim=1)
#         out2 = self.conv3x3(concat1)
#         concat2 = torch.cat([x, out1, out2], dim=1)
#         out3 = self.conv5x5(concat2)
#         concat3 = torch.cat([out1, out2, out3], dim=1)
#         return self.proj(concat3)
