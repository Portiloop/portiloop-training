from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
import matplotlib.pyplot as plt

# size (sequence, batch, features)
# test = torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])

# test = test.flatten(0, 1)

tn = 1
tp = 2
fn = 3
fp = 4

cm = np.array([[1, 2], [3, 4]])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=['non-spindle', 'spindle'])

disp.plot()
plt.savefig('temp/test1029034.png')

# print(test.shape)
