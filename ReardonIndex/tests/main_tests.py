__author__ = 'cd'

import pandas as pd
import numpy as np
import ReardonIndex as rdon
import os

#### Calculated by hand in Reardon test.xlsx
expected = {'info1': 0.123308074,
          'variation1': 0.154655612,
          'info2': 0.029847988,
          'variation2': 0.038903061,
          'variation1B': 0.194979716}

index_names = ['information', 'variation', 'sq_root']


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

def test_subgroupbsb():
    """
    Test with unequal number of groups
    """
    df1B = pd.read_csv(os.path.join("tests", "distr1B.csv"))
    df2 = pd.read_csv(os.path.join("tests", "distr2.csv"))
    df_B = pd.concat([df1B, df2])

    result = rdon.ordinal_seg_per_group(df_B, unit_var='rbd', ord_var='cat', group_var='group')

    assert np.allclose(result['variation'], [expected['variation1B'], expected['variation2']])

def test_decomposition():
    """
    For any dataset, the within variation + between variation should equal total variation
    """
    df1B = pd.read_csv(os.path.join("tests", "distr1B.csv"))
    df2 = pd.read_csv(os.path.join("tests", "distr2.csv"))
    df_B = pd.concat([df1B, df2])

    expected_dec = {'within_variation': 0.121517339,
            'between_variation': 0.004138663,
                    'totalvariation': 0.125656002}

    decomp_d = rdon.decomposition(df_B, unit_var='rbd', ord_var="cat", group_var="group")

    for key in ['between_variation', 'totalvariation', 'within_variation']:
        actual_value, expected_value = decomp_d[key], expected_dec[key]
        assert np.allclose(actual_value, expected_value)


def test_decomposition_random_df():
    """
    Tests the decomposition for a random df
    """
    # df = get_random_df()
    pass


    # for ind in index_names:
    #     total = 0.
    #     between = 0.
    #     within = 0.
    #     assert np.allclose(between + within, total)

