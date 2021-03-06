from __future__ import division
__author__ = 'cd'

# -*- coding: utf-8 -*-
#
# Computes family of segregation indices found in Reardon
#  "Measures of ordinal segregation"
#
#  Author: cdagnino     Dec 2016
#
###############################

import pandas as pd
import numpy as np


def h_info(mat):
    result = -(mat * np.log2(mat) + (1 - mat) * np.log2(1 - mat))
    return np.where(np.isnan(result), 0., result)


def r_variation(mat):
    return 4 * mat * (1 - mat)


def s_sq_root(mat):
    return 2 * np.sqrt(mat * (1 - mat))

index_dict = {'information': h_info, 'variation': r_variation,
                  'sq_root': s_sq_root}


def ordinal_seg(df: pd.DataFrame, unit_var: str, ord_var: str, sort=True) -> pd.Series:
    """
    Calculates the three types of Reardon's Ordinal Segregation Index

    The input dataset at the minimum should contain a administrative or unit variable (such as schools
    or firms) and an ordered variable/category (such as income levels for students or education levels for workers)

    Parameters
    ----------
    df : pd.DataFrame
        needs to contain both a unit variable and an ordinal variable
    unit_var : str
    ord_var : str

    Returns
    -------
    dict
        Returns a dictionary with the three types of the segregation Index: information, variation and square root

    Example:
    --------
    #reardonOrdSeg(df, unit_var='school', ord_var='education_level')
    #
    """

    df = df[[unit_var, ord_var]].dropna(how='any')
    if sort:
        df = df.sort_values(by=[unit_var, ord_var])

    num_Cats = df[ord_var].unique().size
    num_per_unit = df.groupby(unit_var).count().values.ravel()
    num_total = num_per_unit.sum()

    df_count = df.groupby([unit_var, ord_var]).size()

    #1. Overall Cumulative Distribution
    pop_df = df_count.groupby(level=ord_var).sum()
    pop_cum_dist = pop_df.values.cumsum() / pop_df.values.sum()

    #2. Distribution per unit variable
    cumsum_count = df_count.groupby(level=unit_var).cumsum()
    unstacked = cumsum_count.unstack(level=ord_var)
    totals = df_count.groupby(level=0).sum()
    cum_dist = unstacked.div(totals, axis=0)
    # Delete last category (it's always 1)
    cum_dist = cum_dist.iloc[:, :(num_Cats - 1)]

    #3. Iterate over each index function
    reardon_index = {}

    for key, index_func in index_dict.items():
        # Variation of population
        v_pop = (index_func(pop_cum_dist).sum()) / (num_Cats - 1)

        assert isinstance(v_pop, float)
        assert np.isfinite(v_pop)
        assert (v_pop >= 0. and v_pop <= 1.)

        # Variation over units
        v_units = ((index_func(cum_dist).sum(axis=1)) / (num_Cats - 1))

        assert np.isfinite(v_units).all()
        assert (v_units >= 0.).all()
        assert (v_units <= 1.).all()

        segregation_index = (num_per_unit * (v_pop - v_units)).sum()
        segregation_index /= (num_total * v_pop)

        assert (segregation_index >= 0. and segregation_index <= 1.)

        reardon_index[key] = segregation_index
        reardon_index["Vg_" + key] = v_pop

    reardon_index["N_g"] = num_total

    return pd.Series(reardon_index)


def ordinal_seg_per_group(df: pd.DataFrame, unit_var: str, ord_var: str, group_var: str):
    """
    Calculates the three types of Reardon's Ordinal Segregation Index per "super" group

    The input dataset at the minimum should contain:
    + an administrative or unit variable (such as schools or firms). Unit ID should be unique across groups!
    + an ordered variable/category (such as income levels for students or education levels for workers)
    + a group variable that contains several administrative units. Example: school districs or cities

    Parameters
    ----------
    df : pd.DataFrame
        needs to contain a unit variable, a ordinal variable and a group variable
    unit_var : str
    ord_var : str
    group_var : str

    Returns
    -------
    pd.DataFrame
        dataframe: for each group, the three types of the segregation Index:
        information, variation and square root

    Example:
    --------
    #reardonOrdSeg(df, unit_var='school', ord_var='education_level')
    #
    """
    df = (df[[unit_var, ord_var, group_var]].dropna(how='any')
          .sort_values(by=[group_var, unit_var, ord_var]))

    return df.groupby(group_var).apply(ordinal_seg, unit_var=unit_var, ord_var=ord_var, sort=False)


def decomposition(df: pd.DataFrame, unit_var: str, ord_var: str, group_var: str):
    """

    Decomposes the segregation indices into between and within

    Parameters
    ----------
    df
    unit_var
    ord_var
    group_var

    Returns
    -------
    dict with between, within and total segregation for the three segregation indices
    """
    bigN = len(df)
    decomp = {}

    overall_Series = ordinal_seg(df, unit_var=unit_var, ord_var=ord_var)
    between_Series = ordinal_seg(df, unit_var=group_var, ord_var=ord_var)
    within_df = ordinal_seg_per_group(df, unit_var=unit_var, ord_var=ord_var, group_var=group_var)

    for key in index_dict:
        decomp['between_' + key] = between_Series[key]

        decomp['total' + key] = overall_Series[key]

        # Within part
        t_times_V = bigN * overall_Series["Vg_" + key]
        sum_over_groups = (within_df.N_g * within_df['Vg_' + key] * within_df[key]).sum()
        decomp['within_' + key] = sum_over_groups / t_times_V

    return decomp


#### Utilities

def random_multilevel_df(max_n=80, max_n_groups=5, max_n_cats=4):
    """
    Simulates a random dataframe with individuals.
    The (administrative) units belong to a single (super)group.
    The individuals belong to both a category and an administrative unit

    Returns
    ------

    dataframe with 'unit', 'cat' and 'group' columns
    """
    n = np.random.randint(60, high=max_n)
    n_groups = np.random.randint(2, high=max_n_groups)
    n_cats = np.random.randint(2, high=max_n_cats)
    n_units = np.random.randint(n_groups, high=n_groups * 3)

    # Assign units to groups
    unit_to_group = (pd.Series(np.random.randint(0, high=n_groups, size=n_units))
                     .to_frame(name="group"))
    # Assign individuals to units and cats
    d = {}
    d['unit'] = np.random.randint(0, high=n_units, size=n)
    d['cat'] = np.random.randint(0, high=n_cats, size=n)
    df = pd.DataFrame(d)
    # Merge
    df = pd.merge(df, unit_to_group,
                  left_on='unit', right_index=True)
    # Merge
    return df