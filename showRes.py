import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sem
from PIL import Image
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def create_image(img, save_path, img_name):
    im_shape = img.shape
    image = np.zeros([670, 1000], dtype='uint8')
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            image[i*10:i*10+10, j*10:j*10+10] = int(img[i, j]*255)
    im = Image.fromarray(image)
    im.save(save_path + img_name + '.png')

    print(save_path + img_name + '.png !')

# # here we create the roc results
a = np.load('./mid_data/scores1.npy')
a = a/max(a)
b = np.load('./mid_data/y_test.npy')
b = 1 - b
res_file_path = './res/sem/sem_res.npz'
save_roc_pr_curve_data(a, b, res_file_path)

# b = np.load('./mid_data/test_label.npy')

# # here we print the results.
# c = np.resize(a, [67, 100, 40])
# c1 = c[:, :, 1]
# for i in range(40):
#     create_image(c[:, :, i], './slide_res/', str(i))

# # here we have a look of the results
res = np.load(res_file_path)
print(res['fpr'][2])
print(res['tpr'][2])
# print(res['roc_auc'][2])

plt.figure()
lw = 2
plt.plot(res['fpr'], res['tpr'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % res['roc_auc'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


print(res)

# print(a)
# print(b)
# print(c)
