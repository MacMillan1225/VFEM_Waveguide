import numpy as np
from matplotlib import path


# 以下为绘制模式图所需函数

def trifunction(x, y, e, barray):
    x1 = e.nodes[0].x
    x2 = e.nodes[1].x
    x3 = e.nodes[2].x
    y1 = e.nodes[0].y
    y2 = e.nodes[1].y
    y3 = e.nodes[2].y
    l1 = pow(pow((x2 - x3), 2) + pow((y2 - y3), 2), 0.5)
    l2 = pow(pow((x1 - x3), 2) + pow((y1 - y3), 2), 0.5)
    l3 = pow(pow((x1 - x2), 2) + pow((y1 - y2), 2), 0.5)
    L1 = ((x2 * y3 - x3 * y2) + x * (y2 - y3) + y * (x3 - x2)) / 2 / e.volume
    L2 = ((x3 * y1 - x1 * y3) + x * (y3 - y1) + y * (x1 - x3)) / 2 / e.volume
    L3 = ((x1 * y2 - x2 * y1) + x * (y1 - y2) + y * (x2 - x1)) / 2 / e.volume
    dL1x = (y2 - y3) / 2 / e.volume
    dL2x = (y3 - y1) / 2 / e.volume
    dL3x = (y1 - y2) / 2 / e.volume
    dL1y = (x3 - x2) / 2 / e.volume
    dL2y = (x1 - x3) / 2 / e.volume
    dL3y = (x2 - x1) / 2 / e.volume
    n1x = l1 * (L2 * dL3x - L3 * dL2x)
    n2x = l2 * (L3 * dL1x - L1 * dL3x)
    n3x = l3 * (L1 * dL2x - L2 * dL1x)
    n1y = l1 * (L2 * dL3y - L3 * dL2y)
    n2y = l2 * (L3 * dL1y - L1 * dL3y)
    n3y = l3 * (L1 * dL2y - L2 * dL1y)
    for i in e.list:
        barray[i] = -barray[i]
    nx = n1x * barray[0] + n2x * barray[1] + n3x * barray[2]
    ny = n1y * barray[0] + n2y * barray[1] + n3y * barray[2]
    return nx, ny


def f(x, y, liste, x0):
    zx = np.zeros(x.shape)
    zy = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            for k in range(0, len(liste)):
                x1 = liste[k].nodes[0].x
                x2 = liste[k].nodes[1].x
                x3 = liste[k].nodes[2].x
                y1 = liste[k].nodes[0].y
                y2 = liste[k].nodes[1].y
                y3 = liste[k].nodes[2].y
                p = path.Path([(x1, y1), (x2, y2), (x3, y3)])
                if p.contains_points([(x[i][j], y[i][j])]) or ((x[i][j] * y2 - x2 * y[i][j]) + (x2 * y3 - x3 * y2) + (
                        x3 * y[i][j] - y3 * x[i][j]) == 0 and min(x2, x3) <= x[i][j] <= max(x2, x3)) \
                        or ((x1 * y[i][j] - x[i][j] * y1) + (x[i][j] * y3 - x3 * y[i][j]) + (
                        x3 * y1 - y3 * x1) == 0 and min(x1, x3) <= x[i][j] <= max(x1, x3)) \
                        or ((x1 * y2 - x2 * y1) + (x2 * y[i][j] - x[i][j] * y2) + (
                        x[i][j] * y1 - y[i][j] * x1) == 0 and min(x1, x2) <= x[i][j] <= max(x1, x2)):
                    list8 = []
                    list8.append(x0[liste[k].nodes[3].ID])
                    list8.append(x0[liste[k].nodes[4].ID])
                    list8.append(x0[liste[k].nodes[5].ID])
                    zx[i][j], zy[i][j] = trifunction(x[i][j], y[i][j], liste[k], list8)
                    break
    return zx, zy
