import torch
import torch.nn.functional as F
import math


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, t):
    return a + t * (b - a)


def generate_perlin_noise_2d_float(shape, res, generator):
    """
    shape: (H, W) 出力サイズ（整数）
    res: (scale_y, scale_x) 何ピクセルごとにグラディエントを置くか（float対応）
    """
    H, W = shape
    scale_y, scale_x = res

    # 必要なグリッド数を計算
    grid_y = math.ceil(H / scale_y)
    grid_x = math.ceil(W / scale_x)

    # ランダムグラディエントを生成し正規化
    gradients = torch.rand(grid_y + 1, grid_x + 1, 2, generator=generator) * 2 - 1
    gradients = F.normalize(gradients, dim=2)

    # ピクセルごとの座標グリッドを生成
    y = torch.arange(H).view(-1, 1).repeat(1, W).float()
    x = torch.arange(W).view(1, -1).repeat(H, 1).float()

    # 各ピクセルが属するグリッドセル
    gx = (x / scale_x).floor().long().clamp(0, grid_x)
    gy = (y / scale_y).floor().long().clamp(0, grid_y)

    # セル内の相対位置
    tx = (x / scale_x) - gx.float()
    ty = (y / scale_y) - gy.float()

    # フェード補間用
    u = fade(tx)
    v = fade(ty)

    # 勾配取得ヘルパ
    def get_grad(ix, iy):
        return gradients[(gy + iy).clamp(max=grid_y), (gx + ix).clamp(max=grid_x)]

    # 各頂点との距離ベクトルとのドット積
    def dot(ix, iy, dx, dy):
        grad = get_grad(ix, iy)
        disp = torch.stack([dx - ix, dy - iy], dim=2)
        return (grad * disp).sum(dim=2)

    n00 = dot(0, 0, tx, ty)
    n10 = dot(1, 0, tx, ty)
    n01 = dot(0, 1, tx, ty)
    n11 = dot(1, 1, tx, ty)

    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    nxy = lerp(nx0, nx1, v)

    return nxy
