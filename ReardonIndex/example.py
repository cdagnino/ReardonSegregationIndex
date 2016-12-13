import ReardonIndex as rdon
import os
import pandas as pd

__author__ = 'cd'

df = pd.read_csv(os.path.join("tests", "distr1.csv"))

indices = rdon.ordinal_seg(df, unit_var='rbd', ord_var='cat')
print(indices)
