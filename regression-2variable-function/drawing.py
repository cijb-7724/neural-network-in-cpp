import numpy as np
import matplotlib.pyplot as plt
import csv

# CSVファイルからデータを読み込んで2次元配列に格納する関数
def read_csv_to_2d_array(file_name):
    data = []
    with open(file_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            # 各行のデータをリストとして追加
            data.append([float(val) for val in row])
    return data

# ファイル名
file_name = 'point_.csv'

points = read_csv_to_2d_array(file_name)


def func(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) - 4*np.cos(y/3) + np.log10(1 + (x + y)**4)
    return 3*np.sin(np.sqrt(x**2+y**2)) + (x-5)**2/30 + (y+5)**2/30
    return 3*np.sin(np.sqrt(x**2 + y**2)/2)

# グリッドの作成
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)

# 関数値の計算
z = func(x, y)

# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# 軸のアスペクト比を等しくする
ax.set_box_aspect([np.ptp(arr) for arr in [x, y, z]])


points = np.array(points)
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='.', s=10)


plt.show()
