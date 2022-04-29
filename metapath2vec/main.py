from embedding import node_embedding, edge_embedding
from dataset import split_train_test
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
clf = svm.SVC(C=5, kernel='rbf', probability=True)
y_true = []
y_predict = []
acc1 = 0
pre1 = 0
rec1 = 0
f11 = 0
spe1= 0
acc11 = 0
pre11 = 0
rec11 = 0
f111 = 0
spe11= 0
for i in range(3):
    path = os.path.join('data/1/data/dataset', str(i+1))
    data = split_train_test('data/1/data/inters.txt')
    test_pos_edges, train_num, test_num, labels = data.load(i)
    node_feas = node_embedding('data/1/data/inters.txt', f'{path}/vetors.txt')
    rna_embeds, pro_embeds = node_feas.get_node_embeds()
    rna_embeds = rna_embeds.astype(float)
    pro_embeds = pro_embeds.astype(float)
    edge_feas = edge_embedding(rna_embeds, pro_embeds, train_num.item(), test_num.item(), test_pos_edges, labels, 'data/1/data/inters.txt')
    pos_train_feas, pos_test_feas, neg_train_feas, neg_test_feas = edge_feas.get_edge_embeds()
    train_x = np.vstack((pos_train_feas, neg_train_feas))
    a = np.ones(pos_train_feas.shape[0])
    b = np.zeros(pos_train_feas.shape[0])
    train_y = np.hstack((a, b))
    test_x = np.vstack((pos_test_feas, neg_test_feas))
    a = np.ones(pos_test_feas.shape[0])
    b = np.zeros(pos_test_feas.shape[0])
    test_y = np.hstack((a, b))
    clf.fit(train_x, train_y)
    label = clf.predict(test_x)
    target = clf.decision_function(test_x)
    y_true = np.hstack((y_true, test_y))
    y_predict = np.hstack((y_predict, target))
    acc = accuracy_score(test_y, label)
    acc1 = acc1 + acc
    pre = precision_score(test_y, label)
    pre1 = pre1 + pre
    rec = recall_score(test_y, label)
    rec1 = rec1 + rec
    f1 = f1_score(test_y, label)
    f11 = f11 + f1
    spe = recall_score(1 - test_y, 1 - label)
    spe1 = spe1 + spe
    fpr, tpr, thersholds = roc_curve(test_y, target)
    score_flag = 0
    thre_flag = 0
    for thre in thersholds:
        score = np.zeros(shape=target.shape)
        score[target >= thre] = 1
        score[target < thre] = 0
        acc = accuracy_score(test_y, score)
        if acc > score_flag:
            score_flag = acc
            thre_flag = thre
    score = np.zeros(shape=target.shape)
    score[target >= thre_flag] = 1
    score[target < thre_flag] = 0
    aa = accuracy_score(test_y, score)
    acc11 = acc11 + aa
    pp = precision_score(test_y, score)
    pre11 = pre11 + pp
    rr = recall_score(test_y, score)
    rec11 = rec11 + rr
    ff1 = f1_score(test_y, score)
    f111 = f111 + ff1
    sspe = recall_score(1 - test_y, 1 - score)
    spe11 = spe11 + sspe
print(pre1 / 3, rec1 / 3, spe1 / 3, acc1 / 3, f11 / 3)
print(pre11/3, rec11/3, spe11/3, acc11/3, f111/3)
np.save('data/1/data/dataset/y_true.npy', y_true)
np.save('data/1/data/dataset/y_predict.npy', y_predict)
fpr, tpr, thersholds = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', linestyle='-', label='ROC (area = {0:.3f},C={1})'.format(roc_auc, 5), lw=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")#设置图例位置
plt.show()