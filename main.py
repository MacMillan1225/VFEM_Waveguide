import matplotlib.colors
import matplotlib.pyplot as plt
import Mesher
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

from DrawFun import f
from assemble import assemble
import solve

# Setting
# figure font
number_font = fm.FontProperties(family='Microsoft YaHei',
                                weight='bold')
# figure parameter
plt.rcParams['figure.dpi'] = 300

arrow_scale = 2
alpha = 0.8
# Electric field division
x_div = 50
y_div = 25
# working wave number
working_k = 270
# The threshold of beta
beta_thre = 1
# data_mode
data_mode = 1


if data_mode:
    # creat mesh
    nodes, facet, edge, max1 = Mesher.create_mesh(xmin=-0.0254, xmax=0.0254,
                                                  ymin=-0.0127, ymax=0.0127, h_radius=0.00635)
    # np.savez('mesh_data', nodes=nodes, facet=facet, edge=edge, max1=max1)

else:
    mesh_data = np.load('mesh_data.npz', allow_pickle=True)
    nodes = mesh_data['nodes']
    facet = mesh_data['facet']
    edge = mesh_data['edge']
    max1 = mesh_data['max1']

# draw fig
fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")
x, y = nodes[:, 0], nodes[:, 1]
patches = []
facet1 = facet[:, [0, 2, 4]]
for i in range(0, facet1.shape[0]):
    L = [list(x[facet1[i] - 1]), list(y[facet1[i] - 1])]
    polygon = Polygon(list(map(list, zip(*L))), True)
    patches.append(polygon)
pc = PatchCollection(patches, color="w", edgecolor="k")
ax.add_collection(pc)
ax.set_xlim(-0.027, 0.027)
ax.set_ylim(-0.0135, 0.0135)
plt.show()

if data_mode:
    # boundary nodes
    b_n1 = np.where((abs(nodes[:, 0]) == 0.0254) | (abs(nodes[:, 1]) == 0.0127) |
                    (nodes[:, 0] ** 2 + nodes[:, 1] ** 2 < 0.0000403525))
    b_n = list(b_n1[0]+896)
    b_e = list(np.where(edge[:, 1] == 0)[0])
    delete_flag = b_e+b_n
    # np.save('delete_flag',delete_flag)
    # boundary_nodes = nodes[b_n]
    # boundary_edges = edge[b_e]
else:
    delete_flag = np.load('delete_flag.npy', allow_pickle=True)

# draw edge
# plt.scatter(boundary_nodes[:,0],boundary_nodes[:,1])
# tri = edge[b_e][:,0]-1
# facet2 = facet[list(tri)]
# facet2 = facet2[:, [0, 2, 4]]
# patches = []
# for i in range(0, facet2.shape[0]):
#     L = [list(x[facet2[i] - 1]), list(y[facet2[i] - 1])]
#     polygon = Polygon(list(map(list, zip(*L))), True)
#     patches.append(polygon)
# pc = PatchCollection(patches, color="w", edgecolor="r")
# ax.add_collection(pc)
# plt.show()


if data_mode:
    # Using assemble to get A/B
    A, B, liste = assemble(k=working_k, nodes=nodes, facet=facet, edge=edge, max1=max1)
    # 求解AB
    A1 = solve.remove_line(A, delete_flag)
    B1 = solve.remove_line(B, delete_flag)

    np.savez("AB_data", A=A, B=B, liste=liste, A1=A1, B1=B1)
else:
    AB_data = np.load('AB_data.npz', allow_pickle=True)
    A = AB_data['A']
    B = AB_data['B']
    liste = AB_data['liste']
    A1 = AB_data['A1']
    B1 = AB_data['B1']

if data_mode:
    # 解、滤除
    n_beta_square, x_0 = solve.solve1(A1, B1)
    beta_square = np.real(-n_beta_square)
    avail_index = np.where((beta_square > beta_thre) & (beta_square <= working_k ** 2))[0]
    beta_square = beta_square[avail_index]

    # 计算kc
    kc = np.sqrt(working_k**2-beta_square)

    # 计算电场
    x = np.tile(np.linspace(-0.0254, 0.0254, x_div), (y_div, 1))
    y = np.tile(np.linspace(-0.0127, 0.0127, y_div).reshape(y_div, 1), (1, x_div))

    x_0 = x_0[:, avail_index]

    add_flag = np.array(delete_flag) - np.array(range(len(delete_flag)))
    x_0 = np.insert(x_0, add_flag, 0, axis=0)

    np.savez('eig', n_beta_square=n_beta_square, x_0=x_0, x=x, y=y, kc=kc)

else:
    eig = np.load('eig.npz', allow_pickle=True)
    n_beta_square = eig['n_beta_square']
    kc = eig['kc']
    x_0 = eig['x_0']
    x = eig['x']
    y = eig['y']

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(label='Wave number for different models')
text = ['TE10', 'TE01', 'TE20', 'TE11', 'TE21', 'TE30', 'TM11', 'TM21', 'TM31', 'TM40', 'TM12', 'TM02']
viridis = cm.viridis
color = viridis(kc/np.max(kc))
p = ax.barh(text, kc, color=color)
ax.bar_label(p, padding=-45, fmt='%.3f', color='w', fontproperties=number_font)
norm1 = matplotlib.colors.Normalize(vmin=np.min(kc), vmax=np.max(kc))
im1 = cm.ScalarMappable(norm=norm1, cmap=viridis)
plt.colorbar(im1, orientation='horizontal', ticks=np.linspace(50, 250, 9), pad=0.1)
plt.savefig("picture\\Wave_number.png")
plt.show()

for i in range(12):
    zx, zy = f(x, y, liste, x_0[:, i])

    fig = plt.figure()

    ax = fig.add_subplot(111, aspect="equal")
    ax.set_title(label='Electric Field Distribution for %s' % (text[i]))
    left, bottom, width, height = (-0.0254, -0.0127, 0.0508, 0.0254)
    rect = mpatches.Rectangle((left, bottom), width, height, fill=False, color="black", linewidth=1)
    plt.gca().add_patch(rect)
    circle = mpatches.Circle((0, 0), radius=0.00635, fill=False, color='black', linewidth=1)
    plt.gca().add_patch(circle)
    plt.quiver(x, y, zx, zy, scale=arrow_scale, alpha=alpha)
    ax.set_xlim(-0.027, 0.027)
    ax.set_ylim(-0.0135, 0.0135)
    plt.savefig("picture/%s.png" % (text[i]))
    plt.show()
