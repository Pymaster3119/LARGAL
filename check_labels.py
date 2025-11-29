import pickle
import numpy as np

with open('train.pkl','rb') as f:
    data = pickle.load(f)

labels = np.array(data[1])
print('labels dtype:', labels.dtype)
print('labels min, max:', labels.min(), labels.max())
print('unique labels (up to 20):', np.unique(labels)[:20])
print('sample labels:', labels[:50])
