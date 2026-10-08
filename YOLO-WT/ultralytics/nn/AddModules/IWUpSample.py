import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    # 创建正确形状的滤波器，确保与输入通道数匹配
    dec_filters_expanded = torch.zeros(4 * in_size, 1, dec_filters.shape[1], dec_filters.shape[2], dtype=type)
    for i in range(in_size):
        dec_filters_expanded[4*i:4*(i+1), 0] = dec_filters

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    # 同样为重建滤波器创建正确的形状
    rec_filters_expanded = torch.zeros(4 * out_size, 1, rec_filters.shape[1], rec_filters.shape[2], dtype=type)
    for i in range(out_size):
        rec_filters_expanded[4*i:4*(i+1), 0] = rec_filters

    return dec_filters_expanded, rec_filters_expanded

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)

    # 重新组织滤波器以匹配输入通道数
    if filters.shape[0] != 4 * c:
        # 如果滤波器数量不匹配，创建新的滤波器
        new_filters = torch.zeros(4 * c, 1, filters.shape[2], filters.shape[3], device=filters.device)
        for i in range(c):
            new_filters[4 * i:4 * (i + 1)] = filters[:4]
        filters = new_filters

    # 重新组织输入以进行转置卷积
    x_reshaped = x.reshape(b, c * 4, h_half, w_half)

    # 对每个通道单独进行转置卷积，然后合并结果
    result = []
    for i in range(c):
        # 提取当前通道的4个子带
        channel = x_reshaped[:, 4 * i:4 * (i + 1), :, :]
        # 使用对应的滤波器
        channel_filters = filters[4 * i:4 * (i + 1)]
        # 进行转置卷积
        conv_out = F.conv_transpose2d(channel, channel_filters, stride=2, padding=pad)
        # 添加到结果列表
        result.append(conv_out)

    # 合并所有通道的结果
    x_out = torch.cat(result, dim=1)
    return x_out

class IWUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, wave='db1'):
        """
        完全复用WTConv模块的小波上采样模块
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param scale: 上采样倍数（仅支持2的幂次：2/4/8）
        :param wave: 小波基（如db1/haar，兼容pywt）
        """
        super().__init__()
        self.scale = scale
        self.wave = wave
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 校验scale为2的幂次
        assert (scale & (scale - 1)) == 0 and scale >= 2, "scale必须是2的幂次（2/4/8...）"
        self.J = int(torch.log2(torch.tensor(scale)))  # 逆变换层数

        # 计算单尺度小波子带通道数：输入通道数需平均分配到4^J个小波子带（每层变换拆分为4个子带）
        self.subband_ch = in_ch // (4 ** self.J)  # 单尺度子带通道数
        dec_filters, rec_filters = create_wavelet_filter(
            wave=wave,
            in_size=self.subband_ch,
            out_size=self.subband_ch,
            type=torch.float32
        )
        # 注册为非训练参数（自动适配设备/精度）
        self.dec_filters = nn.Parameter(dec_filters, requires_grad=False)
        self.rec_filters = nn.Parameter(rec_filters, requires_grad=False)

        # 定义通道调整层（CBR：Conv+BN+ReLU），实现单尺度子带通道数到目标输出通道数的映射
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(self.subband_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )



    def _split_coeffs(self, x):
        """拆分输入为多尺度小波系数（适配复用的wavelet_transform输入格式）
        核心作用：将编码在通道维度的小波系数拆分为「低频分量+多尺度高频分量」，为逆小波变换做准备
        """
        B, C, H, W = x.shape
        split_size = self.subband_ch  # 单尺度子带通道数
        # 拆分低频分量（LL子带）：取输入的前split_size个通道，保留特征整体轮廓信息
        yL = x[:, :split_size, :, :]  # 低频分量（ll）
        # 初始化高频分量列表：存储各层的LH/HL/HH高频子带（每个元素对应一层的3个高频子带）
        yH = []  # 高频分量列表（lh/hl/hh）

        # 初始化通道拆分起始索引：从低频分量的下一个通道开始拆分高频分量
        curr_ch = split_size
        # 逐层拆分高频分量（共J层）
        for j in range(self.J):
            # 拆分当前层高频分量：取curr_ch到curr_ch+3*split_size的通道（3个高频子带，每个子带split_size个通道）
            yH_j = x[:, curr_ch:curr_ch + 3 * split_size, :, :]
            # 维度重塑：将[B, 3*split_size, H, W]重塑为[B, split_size, 3, H, W]
            # 适配inverse_wavelet_transform函数的输入格式（5维张量：批量×子带通道×子带类型×高度×宽度）
            yH_j = yH_j.view(B, split_size, 3, H, W)
            # 更新通道起始索引：移动到当前层高频分量的下一个通道，为下一层拆分做准备
            yH.append(yH_j)
            curr_ch += 3 * split_size
        # 返回拆分后的低频分量和多尺度高频分量列表
        return yL, yH

    def forward(self, x):
        """
        前向传播：拆分子带 → 复用inverse_wavelet_transform做多尺度逆变换 → 通道调整
        核心流程：输入特征 → 小波系数拆分 → 逐层逆小波变换（上采样） → 通道适配 → 输出高分辨率特征
        """
        B, C, H, W = x.shape
        dtype = x.dtype
        device = x.device

        # 校验输入通道数
        assert C == self.in_ch, f"输入通道数必须为{self.in_ch}，当前为{C}"
        assert C % (4 ** self.J) == 0, f"输入通道数{C}必须是4^{self.J}的倍数"

        # 1. 拆分小波系数（适配复用函数的输入格式）
        # 调用_split_coeffs方法，将输入拆分为低频分量yL和多尺度高频分量列表yH
        yL, yH = self._split_coeffs(x)
        yL = yL.to(dtype).to(device)
        yH = [h.to(dtype).to(device) for h in yH]

        # 2. 多尺度逆小波变换（从高层到低层，全程复用inverse_wavelet_transform）
        # 初始化重构的低频分量为拆分后的yL（最顶层低频分量）
        ll = yL
        # 逆序遍历高频分量列表（从最高层到最低层），逐层执行逆小波变换
        for h in yH[::-1]:
            # 拼接低频分量和当前层高频分量：ll.unsqueeze(2)将ll从[B, split_size, H, W]变为[B, split_size, 1, H, W]
            # 与h（[B, split_size, 3, H, W]）在dim=2维度拼接，得到[B, split_size, 4, H, W]（4个子带：LL+LH+HL+HH）
            curr_coeffs = torch.cat([ll.unsqueeze(2), h], dim=2)
            # 完全复用WDSConv中的inverse_wavelet_transform函数执行单尺度逆小波变换（scale=2）
            # rec_filters转换为输入的dtype和device，确保运算兼容
            ll = inverse_wavelet_transform(curr_coeffs, self.rec_filters.to(dtype).to(device))
            # 裁剪到目标尺寸：消除转置卷积的padding偏移，确保尺寸严格翻倍（H→2H，W→2W）
            ll = ll[:, :, :H * 2, :W * 2]
            # 更新空间尺寸：H和W翻倍，适配下一层逆变换的尺寸要求
            H, W = H * 2, W * 2  # 更新尺寸，适配下一层逆变换

        # 3. 通道调整（和原逻辑一致）
        x_out = self.conv_bn_relu(ll.to(dtype))
        return x_out
