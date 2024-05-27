# -*- coding: utf-8 -*-
# lr_12feats_2decs

make_regression 12 features 2 decimal places
"""

import pandas as pd
import numpy as np
import random
import re
from sklearn.datasets import make_regression

def make_null(r, w):
    rtn = random.choices([np.nan, r], weights=[w, 100-w])
    return re.sub(r"[\[\]]",'', str(rtn))

X, y = make_regression(n_samples=200, n_features=3, n_informative=2)

cols = ['Feature_01', 'Feature_02', 'Feature_03', 'Feature_04',
        'Feature_05', 'Feature_06', 'Feature_07', 'Feature_08',
        'Feature_09', 'Feature_10', 'Feature_11', 'Feature_12']

random.shuffle(cols)

df = pd.DataFrame(data=X, columns=cols[:3])

# introduce multicollinearity
df[cols[3]] = [i + random.random() for i in df.iloc[:, 0]]

# duplicate some columns
df[cols[4]] = [float(random.randint(1,9)) for _ in range(200)]
df[cols[5]] = df[cols[4]]

# make some constant / quasi constants
df[cols[6]] = 0.03
df[cols[7]] = random.choices([0.07, 0.13], weights=[.95, .05], k=200)

# skew some variables
df[cols[8]] = np.random.normal(0, 1, 200)
df[cols[9]] = np.random.normal(0, 1, 200)
df[cols[8]] = df[cols[8]].apply(lambda r: abs(r) if (r < -0.02) else r)
df[cols[9]] = df[cols[9]].apply(lambda r: abs(r)*-1 if (r > 0.01) else r)

# variables that need scaling
df[cols[10]] = random.sample(range(10000, 30000), 200)
df[cols[11]] = random.sample(range(1000, 3000), 200)

# create missing values
df[cols[8]] = df[cols[8]].apply(make_null, args=(2,))
df[cols[9]] = df[cols[9]].apply(make_null, args=(3,))
df[cols[10]] = df[cols[10]].apply(make_null, args=(4,))
df[cols[11]] = df[cols[11]].apply(make_null, args=(5,))

# create some outliers
for i in random.sample(range(len(df)), 20):
  df.loc[i, cols[8]] = (random.random() + random.choice([3, 4])) * random.choice([-1, 1])

df['target'] = y

# duplicates some rows
dupes = df.loc[0:7]
df = pd.concat([df, dupes], ignore_index=True)

# shuffle rows and reindex columns
df = df.sample(frac=1).reset_index(drop=True)
# df = df.reindex(sorted(df.columns), axis=1)
df = df.astype(float)
df = df.round(2)

df.to_csv('lr_12feats_2decs.csv') # 12 features rounded to 2 decimals for linear regression