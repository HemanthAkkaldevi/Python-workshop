#print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
#from time import 
n_colors = 64
china = load_sample_image("china.jpg")
print(type(china))
china = np.array.(china, dtype=np.float64)
w, h, d = original_shape = tuple(china.shape)
assert d ==3
image_array = np.reshape(china)
