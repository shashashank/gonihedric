'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

import torch as T
import numpy as np
from typing import Any, Callable, List, Optional, Type, Union
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
# from segformer_pytorch import Segformer
from torchvision.models.segmentation import deeplabv3_resnet50


class PreActTinyBlock(T.nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActTinyBlock, self).__init__()
        self.bn1 = T.nn.BatchNorm2d(in_planes)
        self.conv1 = T.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = T.nn.Sequential(
        #         T.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        #     )
        T.nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = T.nn.functional.relu(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out += shortcut
        return out


class PreActBlock(T.nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = T.nn.BatchNorm2d(in_planes)
        self.conv1 = T.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = T.nn.BatchNorm2d(planes)
        self.conv2 = T.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = T.nn.Sequential(
                T.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        T.nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        out = T.nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(T.nn.functional.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(T.nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = T.nn.BatchNorm2d(in_planes)
        self.conv1 = T.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = T.nn.BatchNorm2d(planes)
        self.conv2 = T.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 =T.nn.BatchNorm2d(planes)
        self.conv3 = T.nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = T.nn.Sequential(
                T.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        T.nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out     = T.nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(T.nn.functional.relu(self.bn2(out)))
        out = self.conv3(T.nn.functional.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(T.nn.Module):
    def __init__(self,
                block,
                num_blocks,
                num_classes=1,
                zero_init_residual: bool = False,
                replace_stride_with_dilation:Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., T.nn.Module]] = None,
    ) -> None:
        super(PreActResNet, self).__init__()
        if norm_layer is None:
            norm_layer = T.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = T.nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = T.nn.ReLU(inplace=True)
        self.maxpool = T.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = T.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = T.nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
                T.nn.init.constant_(m.weight, 1)
                T.nn.init.constant_(m.bias, 0)
            elif isinstance(m, T.nn.Linear):
                T.nn.init.xavier_normal_(m.weight)
                T.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PreActBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride,
        dilate: bool = False,) -> T.nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return T.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = T.flatten(out, 1)
        out = self.fc(out)
        return out

class PreActResNetSmall(T.nn.Module):
    def __init__(self,
                block,
                num_blocks,
                num_classes=1,
                zero_init_residual: bool = False,
                replace_stride_with_dilation:Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., T.nn.Module]] = None,
        ) -> None:
        super(PreActResNetSmall, self).__init__()
        if norm_layer is None:
            norm_layer = T.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 4
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = T.nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = T.nn.ReLU(inplace=True)
        self.maxpool = T.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 4, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = T.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = T.nn.Linear(32*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
                T.nn.init.constant_(m.weight, 1)
                T.nn.init.constant_(m.bias, 0)
            elif isinstance(m, T.nn.Linear):
                T.nn.init.xavier_normal_(m.weight)
                T.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PreActBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride,
        dilate: bool = False,) -> T.nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return T.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = T.flatten(out, 1)
        out = self.linear(out)
        return out

class PreActResNet3B(T.nn.Module):
    def __init__(self,
                block,
                num_blocks,
                num_classes=1,
                zero_init_residual: bool = False,
                replace_stride_with_dilation:Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., T.nn.Module]] = None,
        ) -> None:
        super(PreActResNet3B, self).__init__()
        if norm_layer is None:
            norm_layer = T.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 4
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = T.nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = T.nn.ReLU(inplace=True)
        self.maxpool = T.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = T.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = T.nn.Linear(16*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
                T.nn.init.constant_(m.weight, 1)
                T.nn.init.constant_(m.bias, 0)
            elif isinstance(m, T.nn.Linear):
                T.nn.init.xavier_normal_(m.weight)
                T.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PreActBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride,
        dilate: bool = False,) -> T.nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return T.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avgpool(out)
        out = T.flatten(out, 1)
        out = self.linear(out)
        return out

class PreActResNetSmallest(T.nn.Module):
    def __init__(self,
                block,
                num_blocks,
                num_classes=1,
                zero_init_residual: bool = False,
                replace_stride_with_dilation:Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., T.nn.Module]] = None,
        ) -> None:
        super(PreActResNetSmallest, self).__init__()
        if norm_layer is None:
            norm_layer = T.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 2
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = T.nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = T.nn.ReLU(inplace=True)
        self.maxpool = T.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 2, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 4, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 6, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = T.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = T.nn.Linear(8*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
                T.nn.init.constant_(m.weight, 1)
                T.nn.init.constant_(m.bias, 0)
            elif isinstance(m, T.nn.Linear):
                T.nn.init.xavier_normal_(m.weight)
                T.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PreActBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride,
        dilate: bool = False,) -> T.nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return T.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = T.flatten(out, 1)
        out = self.linear(out)
        return out

class PreActResNetSmallOld(T.nn.Module):
    def __init__(self,
                block,
                num_blocks,
                num_classes=1,
                zero_init_residual: bool = False,
                replace_stride_with_dilation:Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., T.nn.Module]] = None,
        ) -> None:
        super(PreActResNetSmallOld, self).__init__()
        if norm_layer is None:
            norm_layer = T.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 4
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = T.nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.in_planes)
        # self.relu = T.nn.ReLU(inplace=True)
        self.maxpool = T.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = T.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = T.nn.Linear(32*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                T.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
                T.nn.init.constant_(m.weight, 1)
                T.nn.init.constant_(m.bias, 0)
            elif isinstance(m, T.nn.Linear):
                T.nn.init.xavier_normal_(m.weight)
                T.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PreActBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride,
        dilate: bool = False,) -> T.nn.Sequential:
        if dilate:
            self.dilation *= stride
            stride = 1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return T.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = T.flatten(out, 1)
        out = self.linear(out)
        return out

# def PreActResNet18(nTargets):
#     return PreActResNet(PreActBlock, [2,2,2,2], nTargets)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = T.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def getPositionalEmbeddings(tokens, dim):
    result = T.ones(tokens, dim)
    for i in range(tokens):
        for j in range(dim):
            result[i][j] = np.sin(i/10000**(j/dim)) if j%2==0 else np.cos(i/10000**((j-1)/dim))
    return result

class myMSA(T.nn.Module):
    def __init__(self, dim, heads):
        super(myMSA, self).__init__()
        self.d = dim
        self.heads = heads

        assert dim % heads == 0, "Dimension not divisible by number of heads"

        d_head = dim // heads
        self.qMapping = T.nn.ModuleList([T.nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.kMapping = T.nn.ModuleList([T.nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.vMapping = T.nn.ModuleList([T.nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.d_head = d_head
        self.softmax = T.nn.Softmax(dim=-1)

    def forward(self, sequences):
        # sequences: (batch, tokens, dim) or (N, seq_length, token_dim)
        # we go into shape (N, seq_length, heads, d_head)
        # and come back to (N, seq_length, item_dim)
        result = []
        for sequence in sequences:
            seqRes = []
            for head in range(self.heads):
                qMapping = self.qMapping[head]
                kMapping = self.kMapping[head]
                vMapping = self.vMapping[head]
                seq = sequence[:, head*self.d_head: (head+1)*self.d_head]
                q, k, v = qMapping(seq), kMapping(seq), vMapping(seq)
                attention = self.softmax(q @ k.T / np.sqrt(self.d_head))
                seqRes.append(attention @ v)
            result.append(T.hstack(seqRes))
        return T.cat([T.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(T.nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = T.nn.LayerNorm(hidden_d)
        self.mhsa = myMSA(hidden_d, n_heads)
        self.norm2 = T.nn.LayerNorm(hidden_d)
        self.ffn = T.nn.Sequential(
            T.nn.Linear(hidden_d, hidden_d * mlp_ratio),
            T.nn.GELU(),
            T.nn.Linear(hidden_d * mlp_ratio, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.ffn(self.norm2(out))
        return out

class PixelViT(T.nn.Module):#input_dim, num_classes,
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(PixelViT, self).__init__()

        self.chw = chw # (channel, height, width)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapping
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        print(self.input_d)
        self.linear_mapper = T.nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = T.nn.Parameter(T.rand(1,  self.hidden_d))

        # 3) Positional embeddings
        self.register_buffer('positional_embeddings', getPositionalEmbeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        # 4) Transformer encoder blocks
        self.blocks = T.nn.ModuleList([MyViTBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)])

        # 5) Classification FFN
        self.ffn = T.nn.Sequential(
            T.nn.Linear(self.hidden_d, out_d),
            T.nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Patchify input
        n, c, h, w = images.shape # batch, channel, height, width
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        print(patches.shape)
        # Running linear layer tokenization
        #  map vector corres. to each patch to hidden size dim
        tokens = self.linear_mapper(patches)

        # Add classification token
        tokens = T.cat([self.class_token.repeat(n, 1, 1), tokens], dim=1)

        # Add positional embeddings
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Run through transformer blocks
        for block in self.blocks:
            out = block(out)

        # getting the classification token
        out = out[:, 0]

        return self.ffn(out) # map to output dim, output category distribution



class NeuralNet(T.nn.Module):
  def __init__(self, input_size:int, hidden_size1:int, hidden_size2:int, output_size:int):
    super(NeuralNet, self).__init__()
    self.layer1 = T.nn.Sequential(
      T.nn.Linear(input_size, hidden_size1),
      T.nn.ReLU()
    )
    self.layer2 = T.nn.Sequential(
      T.nn.Linear(hidden_size1, hidden_size2),
      T.nn.ReLU()
    )
    self.layer3 = T.nn.Linear(hidden_size2, output_size)
    for param in self.parameters():
      if len(param.shape) > 1:
        T.nn.init.xavier_uniform_(param)
      else:
        T.nn.init.normal_(param, 0, 1)
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x

class ConvNet(T.nn.Module):
  def __init__(self, num_classes=10):
    super(ConvNet, self).__init__()
    self.layer1 = T.nn.Sequential(
      T.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
      T.nn.BatchNorm2d(16),
      T.nn.ReLU(),
      T.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = T.nn.Sequential(
      T.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
      T.nn.BatchNorm2d(32),
      T.nn.ReLU(),
      T.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc = T.nn.Linear(512, num_classes)
    for param in self.parameters():
      if len(param.shape) > 1:
        T.nn.init.xavier_uniform_(param)
      else:
        T.nn.init.normal_(param, 0, 1)
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out

class Autoencoder(T.nn.Module):
    def __init__(self, layer_widths, internal_activation=T.nn.ReLU(), output_activation=T.nn.Sigmoid(), bottleneck_activation=T.nn.Identity()):
        """
        Parameters:
        - layer_widths (list of int): Widths of layers from input to bottleneck.
        - output_activation (nn.Module): Activation function for the output layer.
        """
        super(Autoencoder, self).__init__()

        assert len(layer_widths) >= 2, "Need at least input and bottleneck layers"

        # Encoder
        encoder_layers = []
        for i in range(len(layer_widths) - 1):
            encoder_layers.append(T.nn.Linear(layer_widths[i], layer_widths[i + 1]))
            encoder_layers.append(internal_activation)
        encoder_layers.pop()  # Remove last ReLU (after bottleneck)
        self.encoder = T.nn.Sequential(*encoder_layers)
        self.bottleneck_activation = bottleneck_activation

        # Decoder (mirror of encoder, excluding bottleneck as input)
        decoder_layers = []
        decoder_widths = list(reversed(layer_widths))
        for i in range(len(decoder_widths) - 1):
            decoder_layers.append(T.nn.Linear(decoder_widths[i], decoder_widths[i + 1]))
            if i < len(decoder_widths) - 2:
                decoder_layers.append(internal_activation)
            else:
                if output_activation is not None:
                    decoder_layers.append(output_activation)
        self.decoder = T.nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.bottleneck_activation(encoded)
        decoded = self.decoder(encoded)
        return decoded

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, T.nn.Conv2d):
            T.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                T.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (T.nn.BatchNorm2d, T.nn.GroupNorm)):
            T.nn.init.constant_(m.weight, 1)
            T.nn.init.constant_(m.bias, 0)
        elif isinstance(m, T.nn.Linear):
            T.nn.init.kaiming_normal_(m.weight)
            T.nn.init.zeros_(m.bias)

def modelPicker(modelName:str, side:int, nTargets:int, data_dir:str):
    if modelName == "resnet18":
        transform = lambda x:x.reshape(-1, side, side)
        model = resnet18(weights=None)
        model.load_state_dict(T.load(data_dir+"checkpoints/resnet18DEFAULT.pt"))
        model.conv1 = T.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fcFeats = model.fc.in_features
        model.fc = T.nn.Linear(in_features=fcFeats, out_features=nTargets, bias=True)
    elif modelName == "resnet50":
        transform=lambda x:T.stack((x.reshape(-1, side, side),)*3, axis=1).squeeze()
        model = resnet50(weights=None, num_classes=nTargets)
        model.load_state_dict(T.load(data_dir+"checkpoints/resnet50DEFAULT.pt"))
    elif modelName == "deeplabv3_resnet50":
        transform=lambda x:T.stack((x.reshape(-1, side, side),)*3, axis=1).squeeze()
        model = deeplabv3_resnet50()
        model.classifier[4] = T.nn.Conv2d(256, nTargets, kernel_size=(1, 1), stride=(1, 1))
    elif modelName == "PARN1":
        transform = lambda x:x.reshape(-1, side, side)
        model = PreActResNetSmall(PreActTinyBlock, [1, 1, 1, 1], num_classes=nTargets)
    elif modelName == "PARN2":
        transform = lambda x:x.reshape(-1, side, side)
        model = PreActResNetSmall(PreActBlock, [1, 1, 1, 1], num_classes=nTargets)
    elif modelName == "PARN3":
        transform = lambda x:x.reshape(-1, side, side)
        model = PreActResNetSmall(PreActBlock, [2, 2, 2, 2], num_classes=nTargets)
    elif modelName == "PARN4":
        transform=lambda x:T.stack((x.reshape(-1, side, side),)*3, axis=1).squeeze()
        model = PreActResNet(PreActTinyBlock, [1, 1, 1, 1], num_classes=nTargets)
    elif modelName == "Linear1":
        transform = lambda x:x
        model = T.nn.Sequential(
            T.nn.Linear(side*side, 100),
            T.nn.ReLU(),
            T.nn.Linear(100, 100),
            T.nn.ReLU(),
            T.nn.Linear(100, nTargets)
        )
    elif modelName == "Linear2":
        transform = lambda x:x
        model = T.nn.Sequential(
            T.nn.Linear(side*side, 100),
            T.nn.ReLU(),
            T.nn.Linear(100, nTargets)
        )
    elif modelName == "PixelViT":
        transform = lambda x:x.reshape(-1, side, side)
        model = PixelViT(chw=(1, side, side), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=nTargets)
    elif modelName == "ConvNet":
        transform = lambda x:x.reshape(-1, side, side)
        model = ConvNet(num_classes=nTargets)
    elif modelName == "NeuralNet":
        transform = lambda x:x.reshape(-1, side*side)
        model = NeuralNet(input_size=side*side, hidden_size1=100, hidden_size2=100, output_size=nTargets)
    else:
        raise ValueError(f"Model {modelName} not recognized")
    initialize_weights(model)
    T.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model, transform


class ConvAutoencoder(T.nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()

        # Encoder: Convolutional layers to extract features
        self.encoder = T.nn.Sequential(
            T.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 14 x 14
            T.nn.ReLU(),
            T.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 7 x 7
            T.nn.ReLU(),
            T.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 4 x 4
            T.nn.ReLU(),
            T.nn.Flatten(),  # Flatten the output to (batch_size, 128*4*4)
            T.nn.Linear(128*4*4, latent_dim)  # Latent dimension
        )

        # Decoder: Fully connected + Transposed convolutional layers
        self.decoder = T.nn.Sequential(
            T.nn.Linear(latent_dim, 128*4*4),  # Latent to expanded shape
            T.nn.ReLU(),
            T.nn.Unflatten(1, (128, 4, 4)),  # Unflatten to 128 x 4 x 4
            T.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 7 x 7
            T.nn.ReLU(),
            T.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 14 x 14
            T.nn.ReLU(),
            T.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 28 x 28
            T.nn.Sigmoid()  # Sigmoid to get values between 0 and 1
        )

    def forward(self, x):
        # Forward pass through the encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
