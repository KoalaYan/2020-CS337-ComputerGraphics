import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100
color_list = [[108, 120, 96], [40, 44, 44], [68, 92, 124], [136, 88, 72], [124, 140, 112], [192, 252, 252], [120, 136, 116], [100, 152, 144], [132, 128, 104], [104, 148, 152], [120, 128, 116], [112, 160, 168], [232, 252, 160], [76, 132, 124], [140, 152, 136], [92, 144, 144], [140, 144, 164], [252, 252, 252], [124, 124, 100], [84, 144, 132], [104, 156, 152], [104, 144, 136], [124, 132, 148], [96, 108, 112]]

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for poi in color_list:
    xs = poi[0]
    ys = poi[1]
    zs = poi[2]
    ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()