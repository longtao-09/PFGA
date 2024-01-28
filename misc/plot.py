import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


# 假设 data 是你的 torch_geometric.data.Data 对象
# data = Data(...)
def plot_visual(data, feature, save_path='plot_avg.pdf'):
    # 使用 t-SNE 对特征进行降维
    defalt_path = '/home/longtao/FGNN/sucai/'
    save_path = defalt_path + save_path
    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(feature.cpu().detach().numpy())
    #
    # # 绘制 t-SNE 结果
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    # plt.title('t-SNE Visualization of Node Features')
    # plt.xlabel('t-SNE feature 1')
    # plt.ylabel('t-SNE feature 2')
    # plt.show()
    # 获取类别标签
    node_labels = data.y.cpu().detach().numpy()

    # 绘制 t-SNE 结果，根据类别着色
    plt.figure(figsize=(10, 6), dpi=400)
    for label in set(node_labels):
        indices = [i for i, l in enumerate(node_labels) if l == label]
        plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1])

    plt.title('t-SNE Visualization of Node Features by Category')
    # plt.xlabel('t-SNE feature 1')
    # plt.ylabel('t-SNE feature 2')
    plt.legend()
    # plt.xticks([], [])
    # plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()

    # # 绘制特征热图
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(feature.detach().numpy(), cmap='viridis')
    # plt.title('Heatmap of Node Features')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Node Index')
    # plt.show()
