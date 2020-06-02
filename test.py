import numpy as np
import pandas as pd

from feat_extraction import feature_extraction, feature_combination_train
df = pd.read_csv('dataset.csv')
coarse_class = df['coarse'].values
fine_class = df['coarse'].values

data = df['indo'].values
print(data)
fe = feature_extraction(data)
print(fe)
vector, label = feature_combination_train(fe)
print(label)
vectorized = pd.DataFrame(data=vector, columns=label)
print(vectorized)
vectorized.to_csv('vector.csv', index=False)

