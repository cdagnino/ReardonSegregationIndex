__author__ = 'cd'

import pandas as pd
import numpy as np
import ReardonIndex as rdon
import os

#### Calculated by hand
expected = {'info1': 0.123308074,
          'variation1': 0.154655612,
          'info2': 0.029847988,
          'variation2': 0.038903061}


def test_no_subgroups():
    df1 = pd.read_csv(os.path.join("tests", "distr1.csv"))
    df2 = pd.read_csv(os.path.join("tests", "distr2.csv"))
    df1_indices = rdon.ordinal_seg(df1, unit_var='rbd', ord_var='cat')
    assert np.allclose(df1_indices['variation'], expected['variation1'])
    assert np.allclose(df1_indices['information'], expected['info1'])

    df2_indices = rdon.ordinal_seg(df2, unit_var='rbd', ord_var='cat')
    assert np.allclose(df2_indices['variation'], expected['variation2'])
    assert np.allclose(df2_indices['information'], expected['info2'])

def test_subgroups():
    df1 = pd.read_csv(os.path.join("tests", "distr1.csv"))
    df2 = pd.read_csv(os.path.join("tests", "distr2.csv"))
    df = pd.concat([df1, df2])

    result = rdon.ordinal_seg_per_group(df, unit_var='rbd', ord_var='cat', group_var='group')

    assert np.allclose(result['information'], [expected['info1'], expected['info2']])
    assert np.allclose(result['variation'], [expected['variation1'], expected['variation2']])
