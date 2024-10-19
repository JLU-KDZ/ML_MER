import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d

# 混淆矩阵
conf_matrix = np.array([
    [116, 9, 79, 1, 3, 16],
    [12, 180, 45, 5, 5, 53],
    [33, 30, 462, 1, 8, 49],
    [0, 0, 0, 247, 0, 0],
    [0, 0, 3, 0, 238, 0],
    [12, 43, 154, 6, 7, 143]
])

# 类别数
num_classes = conf_matrix.shape[0]

# 生成真实标签和预测标签
y_true = []
y_scores = []

for i in range(num_classes):
    y_true.extend([i] * sum(conf_matrix[i]))
    for j in range(num_classes):
        y_scores.extend([j] * conf_matrix[i, j])

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# 将标签二值化
y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))

# 计算每个类别的ROC曲线和AUC
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], (y_scores == i).astype(int))
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算宏平均ROC
all_fpr = np.linspace(0, 1, 100)  # 使用100个点进行插值
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):
    f = interp1d(fpr[i], tpr[i], kind='linear', fill_value='extrapolate')
    mean_tpr += f(all_fpr)

mean_tpr /= num_classes
mean_auc = auc(all_fpr, mean_tpr)

# 绘制宏平均ROC曲线
plt.plot(all_fpr, mean_tpr, label='Macro-average ROC curve (AUC = {0:0.2f})'.format(mean_auc), color='blue')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()
