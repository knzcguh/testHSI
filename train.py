# main.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt

import get_cls_map
from model import OptimizedHSI3DTransformer




# utils/dataset.py

# 加载数据  15 144
# data = sio.loadmat('../data/Houstondata.mat')['Houstondata']
# labels = sio.loadmat('../data/Houstonlabel.mat')['Houstonlabel']

# 加载数据  176 13
# data = sio.loadmat('../data/KSC.mat')['KSC']
# labels = sio.loadmat('../data/KSC_gt.mat')['KSC_gt']

# 读入数据 16 200
#     data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
#     labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

# 读入数据 9 103
# data = sio.loadmat('../data/PaviaU.mat')['paviaU']
# labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']

# 帕维亚中心 9 102
# data = sio.loadmat('../data/Pavia.mat')['pavia']
# labels = sio.loadmat('../data/Pavia_gt.mat')['pavia_gt']

# 萨利纳斯场景 16 204
# data = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
# labels = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']


def create_data_loader(data_path='Indian_pines.mat',
                       gt_path='Indian_pines_gt.mat',
                       batch_size=32,
                       patch_size=11,
                       train_ratio=0.1,
                       val_ratio=0.1):
    # 加载数据
    data = sio.loadmat('../data/Houstondata.mat')['Houstondata']
    labels = sio.loadmat('../data/Houstonlabel.mat')['Houstonlabel']

    # 创建patches
    patches, patch_labels = create_patches(data, labels, patch_size=patch_size)

    # 获取唯一的类别
    unique_classes = np.unique(patch_labels)

    # 按类别分组样本
    class_indices = {cls: np.where(patch_labels == cls)[0] for cls in unique_classes}

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = class_indices[cls]
        np.random.shuffle(cls_indices)  # 随机打乱每个类别的样本

        # 计算每个类别的样本数量
        n_samples = len(cls_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # 分配样本
        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

        # 打印每个类别的分配情况
        print(f"\n类别 {cls} 样本分配:")
        print(f"总样本数: {n_samples}")
        print(f"训练样本数: {n_train} ({(n_train / n_samples) * 100:.1f}%)")
        print(f"验证样本数: {n_val} ({(n_val / n_samples) * 100:.1f}%)")
        print(
            f"测试样本数: {len(cls_indices[n_train + n_val:])} ({(len(cls_indices[n_train + n_val:]) / n_samples) * 100:.1f}%)")

    # 随机打乱
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # 创建数据集和数据加载器
    train_dataset = HSIDataset(patches[train_indices], patch_labels[train_indices])
    val_dataset = HSIDataset(patches[val_indices], patch_labels[val_indices])
    test_dataset = HSIDataset(patches[test_indices], patch_labels[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 打印最终的样本分布
    print("\n最终样本分布统计:")
    for cls in unique_classes:
        n_train = sum(patch_labels[train_indices] == cls)
        n_val = sum(patch_labels[val_indices] == cls)
        n_test = sum(patch_labels[test_indices] == cls)
        total = n_train + n_val + n_test
        print(f"类别 {cls}:")
        print(f"  训练集: {n_train} 样本 ({(n_train / total) * 100:.1f}%)")
        print(f"  验证集: {n_val} 样本 ({(n_val / total) * 100:.1f}%)")
        print(f"  测试集: {n_test} 样本 ({(n_test / total) * 100:.1f}%)")

    return train_loader, val_loader, test_loader, patch_labels
# patch大小
def create_patches(data, labels, patch_size=11, pad_size=5):
    """
    创建图像patches
    data: [H, W, C]
    labels: [H, W]
    """
    height, width, bands = data.shape

    # 检查labels的形状
    print(f"Labels shape inside create_patches: {labels.shape}")  # 调试打印
    if labels.ndim != 2:
        raise ValueError(f"Labels should be a 2D array, but got {labels.ndim}D array.")

    # 添加填充
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    data_padded = np.pad(data, pad_width, mode='reflect')

    # 创建patches
    patches = []
    patch_labels = []

    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:  # 只处理有标签的像素
                patch = data_padded[i:i + 2 * pad_size + 1,
                        j:j + 2 * pad_size + 1, :]
                patches.append(patch)
                patch_labels.append(labels[i, j] - 1)  # 标签从0开始

    return np.array(patches), np.array(patch_labels)

def augment_data(patches):
    """简单的数据增强"""
    augmented = []
    for patch in patches:
        augmented.append(patch)  # 原始
        augmented.append(np.fliplr(patch))  # 水平翻转
        augmented.append(np.flipud(patch))  # 垂直翻转
    return np.array(augmented)

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        """
        data: [N, H, W, C] patches
        labels: [N] 标签
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].permute(2, 0, 1)  # [C, H, W]
        y = self.labels[idx]
        return x, y

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, num_classes]
            target: 目标值 [B]
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # 创建平滑标签
            true_dist = torch.zeros_like(pred)  # [B, num_classes]
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.

        Args:
            patience (int): Number of epochs to wait before stopping when loss is
                          not improving. Default: 10
            min_delta (float): Minimum change in the monitored quantity to qualify
                             as an improvement. Default: 0
            verbose (bool): If True, prints a message for each validation loss
                          improvement. Default: True
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Early stopping counter reset: validation loss improved to {val_loss:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
                return True

        return False


def train(model, classes, train_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.to(device)

    # 使用标签平滑
    criterion = LabelSmoothingLoss(classes=classes, smoothing=0.1)

    # 修改优化器参数
    optimizer = optim.AdamW(net.parameters(), lr=0.0001,
                            weight_decay=0.001,
                            betas=(0.9, 0.999),
                            eps=1e-8)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    if len(train_loader) == 0:
        raise ValueError("Error: Empty train loader!")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            batch_size = data.size(0)
            if batch_size == 0:
                continue

            try:
                data = data.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(data)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                # 使用 detach() 来分离计算图
                running_loss += loss.detach().item() * batch_size
                with torch.no_grad():
                    _, preds = torch.max(outputs.detach(), 1)
                    running_corrects += torch.sum(preds == labels).item()
                running_samples += batch_size

            except RuntimeError as e:
                print(f"Error in training: {str(e)}")
                continue

        if running_samples == 0:
            print("Warning: No samples processed in this epoch!")
            continue

        epoch_loss = running_loss / running_samples
        epoch_acc = 100. * running_corrects / running_samples

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{epochs}] '
              f'Loss: {epoch_loss:.4f} '
              f'Acc: {epoch_acc:.2f}% '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

    print("Training completed!")
    return net, device


def test(device, net, test_loader):
    net.eval()
    all_preds = []
    all_labels = []

    # 检查测试加载器是否为空
    if len(test_loader) == 0:
        raise ValueError("测试数据加载器为空")

    print(f"开始测试: 共有 {len(test_loader)} 个批次")

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            # 检查数据批次
            if data.size(0) == 0:
                print(f"警告：第 {batch_idx} 批次数据为空")
                continue

            try:
                # 移动数据到设备
                data = data.to(device)

                # 打印数据形状和值范围
                # print(f"批次 {batch_idx}: 输入形状 {data.shape}, "
                #       f"范围 [{data.min().item():.2f}, {data.max().item():.2f}]")

                # 前向传播
                outputs = net(data)

                # 检查输出是否有效
                if outputs is None or torch.isnan(outputs).any():
                    print(f"警告：第 {batch_idx} 批次产生无效输出")
                    continue

                _, preds = outputs.max(1)

                # 收集预测结果和标签
                all_preds.append(preds.cpu())
                all_labels.append(labels)

            except Exception as e:
                print(f"处理批次 {batch_idx} 时发生错误: {str(e)}")
                continue

    # 检查是否有收集到预测结果
    if not all_preds:
        raise RuntimeError("没有收集到任何预测结果")

    try:
        # 合并所有批次的结果
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        print(f"测试完成: 预测形状 {y_pred.shape}, 标签形状 {y_true.shape}")

        return y_pred, y_true

    except Exception as e:
        print(f"合并结果时发生错误: {str(e)}")
        raise


def acc_reports(y_true, y_pred):
    # 计算混淆矩阵
    confusion = confusion_matrix(y_true, y_pred)

    # 计算每个类别的准确率，添加防止除零的处理
    class_counts = confusion.sum(axis=1)
    each_acc = np.zeros_like(class_counts, dtype=float)

    # 只计算样本数大于0的类别的准确率
    valid_classes = class_counts > 0
    each_acc[valid_classes] = confusion.diagonal()[valid_classes] / class_counts[valid_classes]

    # 计算 AA 时只考虑有效的类别
    valid_acc = each_acc[valid_classes]
    aa = np.mean(valid_acc) if len(valid_acc) > 0 else 0.0

    # 计算 OA 和 Kappa
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return str(confusion), oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def main():
    # 创建必要的目录
    os.makedirs('cls_params', exist_ok=True)
    os.makedirs('cls_result', exist_ok=True)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置模型参数
    input_channels = 144  # 数据集的原始通道数
    pca_channels = 30  # PCA降维后的通道数
    num_classes = 15  # 数据集的类别数
    patch_size = 8   # patch大小
    embedding_dim = 64  # embedding维度

    batch_size = 128  # 批次大小
    train_ratio = 0.05
    val_ratio = 0.05
    heads = 4
    dim_head = 64
    epochs = 100

    spatial_depth = 2
    spectral_depth = 2
    mlp_dim = 256
    dropout = 0.1

    # spatial_depth = 2
    # spectral_depth = 2
    # heads = 4
    # dim_head = 32
    # mlp_dim = 256
    # dropout = 0.1
    # 加载数据
    train_loader, test_loader, all_data_loader, y_all = create_data_loader(
        batch_size=batch_size,  # 减小batch size
        patch_size=patch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    # 创建模型实例
    model = OptimizedHSI3DTransformer(
        input_channels=input_channels,  # 原始通道数
        pca_channels=pca_channels,      # PCA降维后的通道数
        num_classes=num_classes,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        spatial_depth=spatial_depth,
        spectral_depth=spectral_depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=dropout
    )
    # 训练
    tic1 = time.perf_counter()
    net, device = train(model, num_classes, train_loader, epochs=epochs)
    torch.save(net.state_dict(), 'cls_params/net_params.pth')
    toc1 = time.perf_counter()

    # 测试
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()

    # 计算评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)

    # 记录时间
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2

    # 保存结果
    with open("cls_result/classification_report.txt", 'w') as f:
        f.write(f'{Training_Time} Training_Time (s)\n')
        f.write(f'{Test_time} Test_time (s)\n')
        f.write(f'{kappa} Kappa accuracy (%)\n')
        f.write(f'{oa} Overall accuracy (%)\n')
        f.write(f'{aa} Average accuracy (%)\n')
        f.write(f'{each_acc} Each accuracy (%)\n')
        f.write(f'{classification}\n')
        f.write(f'{confusion}\n')


        # Indian Pines 数据集的实际尺寸是 145x145
        # 生成分类图
        if len(y_all.shape) == 1:
            total_pixels = len(y_all)
            print(f"Total pixels in ground truth: {total_pixels}")

            # 对于 Indian Pines 数据集
            height, width = 145, 145
            print(f"Using Indian Pines dimensions: {height}x{width}")

            try:
                # 创建标签矩阵
                y_reshaped = np.zeros((height, width))
                valid_pixels = y_all != 0
                y_valid = y_all[valid_pixels]
                valid_positions = np.nonzero(valid_pixels)[0]

                # 检查标签范围
                print(f"Label range: [{y_valid.min()}, {y_valid.max()}]")
                if y_valid.max() > 16:
                    print("Warning: Labels exceed expected range (1-16)")

                # 计算每个有效像素的位置
                rows = valid_positions // width
                cols = valid_positions % width

                # 填充标签
                for idx, (i, j) in enumerate(zip(rows, cols)):
                    if idx < len(y_valid):
                        y_reshaped[i, j] = y_valid[idx]

                y_all = y_reshaped
                print("Successfully reshaped ground truth")
                print(f"Number of valid pixels: {len(y_valid)}")
                unique_labels = np.unique(y_all)
                print(f"Unique labels in ground truth: {unique_labels}")

            except Exception as e:
                print(f"Error during reshaping: {str(e)}")
                raise

        # 确保标签是正确的
    assert y_all.shape == (height, width), f"Unexpected shape: {y_all.shape}"
    assert np.all(y_all >= 0), "Negative labels found"
    assert len(np.unique(y_all)) <= 17, f"Too many unique labels: {np.unique(y_all)}"

    get_cls_map.get_cls_map(net, device, y_all)

if __name__ == "__main__":
    main()
