import torch
from torch import nn as nn
from torch.nn import functional as F

from ..registry import NECKS
from ..utils import ConvModule


def use_ceil_mode(xs, xl):
    ws, hs = xs.size()[2:4]
    wl, hl = xl.size()[2:4]
    return (wl / ws < 2) or (hl / hs < 2)


def get_groups(channels, ref=8):
    if channels == 1:
        return 1
    xs = filter(lambda x: channels % x == 0, range(2, channels + 1))
    c = min(filter(lambda x: x >= ref, xs), key=lambda x: x - ref)
    return channels // c


def get_gn_cfg(channels):
    return {
        "type": "GN",
        "num_groups": get_groups(channels)
    }


def fast_normalize(w, eps=1e-4, dim=0):
    w = torch.relu(w)
    w = w / (torch.sum(w, dim=dim, keepdim=True) + eps)
    return w


def dwconv(in_channels, out_channels, kernel_size=3):
    conv1 = ConvModule(
        in_channels, in_channels, kernel_size,
        padding=kernel_size // 2, groups=in_channels,
        norm_cfg=get_gn_cfg(in_channels), activation='relu',
    )
    conv2 = ConvModule(
        in_channels, out_channels, 1,
        norm_cfg=get_gn_cfg(out_channels), activation=None,
    )
    return nn.Sequential(
        conv1,
        conv2,
    )


class BottomUpFusion2(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2,)), requires_grad=True)
        self.conv = dwconv(f_channels, f_channels, kernel_size=3)

    def forward(self, p, pp):
        pp = F.max_pool2d(pp, kernel_size=2, ceil_mode=use_ceil_mode(p, pp))
        w = fast_normalize(self.weight)
        p = w[0] * p + w[1] * pp
        p = self.conv(p)
        return p


class TopDownFusion2(nn.Module):

    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2,)), requires_grad=True)
        self.conv = dwconv(f_channels, f_channels, kernel_size=3)

    def forward(self, p, pp):
        h, w = p.size()[2:4]
        pp = F.interpolate(pp, (h, w), mode='bilinear', align_corners=False)
        w = fast_normalize(self.weight)
        p = w[0] * p + w[1] * pp
        p = self.conv(p)
        return p


class BottomUpFusion3(nn.Module):

    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((3,)), requires_grad=True)
        self.conv = dwconv(f_channels, f_channels, kernel_size=3)

    def forward(self, p1, p2, pp):
        pp = F.max_pool2d(pp, kernel_size=2, ceil_mode=use_ceil_mode(p1, pp))
        w = fast_normalize(self.weight)
        p = w[0] * p1 + w[1] * p2 + w[2] * pp
        p = self.conv(p)
        return p


class BiFPNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        n = len(in_channels)
        self.lats = nn.ModuleList([
            ConvModule(c, out_channels, kernel_size=1, norm_cfg=get_gn_cfg(out_channels))
            if c != out_channels else nn.Identity()
            for c in in_channels
        ])
        self.tds = nn.ModuleList([
            TopDownFusion2(out_channels)
            for _ in range(n - 1)
        ])
        self.bus = nn.ModuleList([
            BottomUpFusion3(out_channels)
            for _ in range(n - 2)
        ])
        self.bu = BottomUpFusion2(out_channels)

    def forward(self, ps):
        ps = [lat(p) for p, lat in zip(ps, self.lats)]

        ps2 = [ps[-1]]
        for p, td in zip(reversed(ps[:-1]), self.tds):
            ps2.append(td(p, ps2[-1]))
        ps3 = [ps2[-1]]
        ps2 = reversed(ps2[1:-1])

        for p1, p2, bu in zip(ps[1:-1], ps2, self.bus):
            ps3.append(bu(p1, p2, ps3[-1]))
        ps3.append(self.bu(ps[-1], ps3[-1]))

        return tuple(ps3)


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, num_layers):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels

        self.extras = nn.ModuleList()
        extra_levels = num_outs - len(in_channels)

        self.fpns = nn.ModuleList([
            BiFPNLayer(in_channels + [out_channels] * extra_levels, out_channels)
        ])
        for _ in range(num_layers - 1):
            self.fpns.append(BiFPNLayer([out_channels] * num_outs, out_channels))

        if extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    extra_fpn_conv = ConvModule(
                        self.in_channels[-1],
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        norm_cfg=get_gn_cfg(out_channels),
                        activation=None)
                else:
                    extra_fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        norm_cfg=get_gn_cfg(out_channels),
                        activation='relu',
                        inplace=False,
                        order=('act', 'conv', 'norm')
                    )
                self.extras.append(extra_fpn_conv)

    def forward(self, ps):
        assert isinstance(ps, (tuple, list))
        ps = list(ps)
        for extra in self.extras:
            ps.append(extra(ps[-1]))
        for fpn in self.fpns:
            ps = fpn(tuple(ps))
        return ps

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
