#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:05:26 2023

@author: Martin Colledge

B-positive implementation in Python. All equations from :

van der Elst, N. J. (2021). B-positive: A robust
estimator of aftershock magnitude distribution in transiently incomplete
catalogs. Journal of Geophysical Research: Solid Earth, 126, e2020JB021027.
https://doi. org/10.1029/2020JB021027
"""

# Import usual modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mle_b_value
import scipy.stats


def arcoth(value):
    """
    Arc hyperbolic cotangent.

    Calculated in terms of logarithms.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    float
        Arc hyperbolic cotangent of the input.

    """
    return 0.5 * np.log((value + 1) / (value - 1))


def discrete_b_positive(
    catalog, difference_cutoff, delta_magnitude, bootstrap=True
):
    """
    Discrete b-positive Laplace mle.

    As defined in equation A3 modified to the terms of equation A17 (relative
    to equation A2) of van der Elst, N. J. (2021). B-positive: A robust
    estimator of aftershock magnitude distribution in transiently incomplete
    catalogs. Journal of Geophysical Research: Solid Earth, 126, e2020JB021027.
    https://doi. org/10.1029/2020JB021027

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff : float
        Magnitude difference cutoff.
    delta_magnitude : float
        Magnitude discretizaton step.

    Returns
    -------
    bpositive : float
        b-positive b-value estimation of given catalog.

    """

    rounded_magnitudes = (
        10
        * delta_magnitude
        * np.round(catalog["Magnitude"] / (10 * delta_magnitude), 1)
        + (delta_magnitude / 2) * (delta_magnitude * 10 - 1)
    ).copy()

    bpositive = discrete_b_positive_routine(
        rounded_magnitudes, difference_cutoff, delta_magnitude
    )

    if bootstrap:

        def discrete_b_positive_routine_bootstrap(magnitudes):
            return discrete_b_positive_routine(
                magnitudes, difference_cutoff, delta_magnitude
            )

        res = scipy.stats.bootstrap(
            (rounded_magnitudes,),
            statistic=discrete_b_positive_routine_bootstrap,
            batch=20,
        )
        bpositive_std = res.standard_error
        return bpositive, bpositive_std
    else:
        return bpositive


def discrete_b_positive_routine(
    magnitudes, difference_cutoff, delta_magnitude
):
    return (
        (1 / (delta_magnitude / 2))
        * (
            arcoth(
                (1 / (delta_magnitude / 2))
                * (
                    np.mean(
                        np.diff(magnitudes)[
                            np.diff(magnitudes) >= difference_cutoff
                        ]
                    )
                    - difference_cutoff
                    + (delta_magnitude / 2)
                )
            )
        )
        / np.log(10)
    )


def discrete_b_negative(
    catalog, difference_cutoff, delta_magnitude, bootstrap=True
):
    """
    Discrete b-negative Laplace mle.

    Adapted from the discrete_b_positive_function with modification following
    equation 6 of van der Elst (2021)


    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff : float
        Magnitude difference cutoff.
    delta_magnitude : float
        Magnitude discretizaton step.

    Returns
    -------
    bnegative : float
        b-negative b-value estimation of given catalog.

    """
    rounded_magnitudes = (
        10
        * delta_magnitude
        * np.round(catalog["Magnitude"] / (10 * delta_magnitude), 1)
        + (delta_magnitude / 2) * (delta_magnitude * 10 - 1)
    ).copy()

    bnegative = discrete_b_negative_routine(
        rounded_magnitudes, difference_cutoff, delta_magnitude
    )

    if bootstrap:

        def discrete_b_negative_routine_bootstrap(magnitudes):
            return discrete_b_negative_routine(
                magnitudes, difference_cutoff, delta_magnitude
            )

        res = scipy.stats.bootstrap(
            (rounded_magnitudes,),
            statistic=discrete_b_negative_routine_bootstrap,
            batch=30,
        )
        bnegative_std = res.standard_error
        return bnegative, bnegative_std
    else:
        return bnegative


def discrete_b_negative_routine(
    magnitudes, difference_cutoff, delta_magnitude
):
    return (
        (1 / (delta_magnitude / 2))
        * (
            arcoth(
                (1 / (delta_magnitude / 2))
                * (
                    np.mean(
                        -np.diff(magnitudes)[
                            np.diff(magnitudes) <= -difference_cutoff
                        ]
                    )
                    - difference_cutoff
                    + (delta_magnitude / 2)
                )
            )
        )
        / np.log(10)
    )


# %% Continuous formulas
def continuous_b_positive(catalog, difference_cutoff):
    """
    Continuous b-positive Laplace mle.

    Warning : Overestimates b-value for tested catalog with delta_magnitude.
    Prefer using the discrete formula.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff : float
        Magnitude difference cutoff.

    Returns
    -------
    b_positive : TYPE
        b-positive b-value estimation of given catalog.

    """
    cat_diff = np.round(catalog["Magnitude"].diff(), 2)
    cat_diff_positive = cat_diff[cat_diff >= np.round(difference_cutoff, 2)]
    bpositive = (
        1 / (np.mean(cat_diff_positive) - np.round(difference_cutoff, 2))
    ) / np.log(10)
    return bpositive


def continuous_b_negative(catalog, difference_cutoff):
    """
    Continuous b-negative Laplace mle.

    Warning : Overestimates b-value for tested catalog with delta_magnitude.
    Prefer using the discrete formula.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff : float
        Magnitude difference cutoff.

    Returns
    -------
    b_negative : TYPE
        b_negative b-value estimation of given catalog.

    """
    cat_diff = np.round(catalog["Magnitude"].diff(), 2)
    cat_diff_negative = -cat_diff[cat_diff <= -np.round(difference_cutoff, 2)]
    bnegative = (
        1
        / (np.mean(cat_diff_negative) - np.round(difference_cutoff, 2))
        / np.log(10)
    )
    return bnegative


# %% Plots


def plot_magnitude_difference(catalog):
    """
    Quick plot of positive magnitude difference distribution.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.

    Returns
    -------
    None.

    """
    plt.figure()
    sns.histplot(
        catalog["Magnitude"].diff()[catalog["Magnitude"].diff() > 0],
        binwidth=0.1,
        element="poly",
    )
    plt.yscale("log")
    plt.show()


def test_continuous_b_pos_neg_with_difference_cutoff(
    catalog, difference_cutoff_range
):
    """
    Test continuous estimators in relation to magnitude difference cutoff.

    Go through a range of magnitude difference cutoffs and plot both b-positive
    and b-negative to illustrate the overestimation of b-value with b-negative.

    Warning : Continuous estimators overestimate b-value for tested catalog
    with delta_magnitude.
    Prefer using the discrete formula.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff_range : list
        Magnitude difference cutoff list or 1D array.


    Returns
    -------
    None.

    """
    b_pos = []
    b_neg = []
    for difference_cutoff in difference_cutoff_range:
        b_pos.append(continuous_b_positive(catalog, difference_cutoff))
        b_neg.append(continuous_b_negative(catalog, difference_cutoff))
    plt.figure(figsize=(10, 3))
    plt.plot(difference_cutoff_range, b_pos, label="b-positive")
    plt.plot(difference_cutoff_range, b_neg, label="b-negative")
    plt.ylabel("b-value")
    plt.xlabel("Cutoff magnitude difference")
    plt.legend()
    plt.title("Continuous magnitude distribution")
    plt.grid()
    plt.show()


def test_discrete_b_pos_neg_with_difference_cutoff(
    catalog, difference_cutoff_range, delta_magnitude
):
    """
    Test discrete estimators in relation to magnitude difference cutoff.

    Go through a range of magnitude difference cutoffs and plot both b-positive
    and b-negative to illustrate the overestimation of b-value with b-negative.

    Warning : Continuous estimators overestimate b-value for tested catalog
    with delta_magnitude.
    Prefer using the discrete formula.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff_range : list
        Magnitude difference cutoff list or 1D array.
    delta_magnitude : float
        Magnitude discretizaton step.

    Returns
    -------
    None.

    """
    b_pos = []
    b_pos_std = []
    b_neg = []
    b_neg_std = []

    for difference_cutoff in difference_cutoff_range:
        b_pos_tmp, b_pos_std_tmp = discrete_b_positive(
            catalog, difference_cutoff, delta_magnitude
        )
        b_pos.append(b_pos_tmp)
        b_pos_std.append(b_pos_std_tmp)

        b_neg.append(
            discrete_b_negative(catalog, difference_cutoff, delta_magnitude)
        )
    plt.figure(figsize=(10, 3))

    plt.plot(difference_cutoff_range, b_pos, label="b-positive")
    plt.plot(difference_cutoff_range, b_neg, label="b-negative")
    plt.fill_between(
        difference_cutoff_range,
        [b - bstd for b, bstd in zip(b_pos, b_pos_std)],
        [b + bstd for b, bstd in zip(b_pos, b_pos_std)],
    )
    plt.ylabel("b-value")
    plt.xlabel("Cutoff magnitude difference")
    plt.legend()
    plt.title("Discrete magnitude distribution")
    plt.grid()
    plt.show()


def temporal_variation_of_b_value(
    catalog,
    difference_cutoff,
    completeness_magnitude,
    delta_magnitude,
    window_size,
):
    """
    Temporal evolution of b-value with constant number of events.

    Parameters
    ----------
    catalog : pandas dataframe
        Seismicity catalog with 'Magnitude' column.
    difference_cutoff : float
        Magnitude difference cutoff.
    completeness_magnitude : float or int
        Completeness magnitude estimate of the catalog.
    delta_magnitude : float
        Magnitude discretizaton step.
    window_size : int
        Number of events to consider in rolling window.

    Returns
    -------
    None.

    """
    # Recreate figure 8 d of Van der Elst 2021
    b_pos = []
    b_mle = []
    for event_number in range(len(catalog) - window_size):
        rolling_catalog = catalog[event_number : event_number + window_size]
        b_pos.append(
            discrete_b_positive(
                rolling_catalog, difference_cutoff, delta_magnitude
            )
        )
        b_mle.append(
            mle_b_value.mle_b_value_continuous(
                rolling_catalog, completeness_magnitude
            )[0]
        )

    plt.figure(figsize=(15, 9))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(
        np.arange(len(catalog) - window_size), b_mle, label="Aki-Utsu mle"
    )
    plt.legend()
    plt.ylabel("b-value")
    plt.xlim(0, len(catalog))
    plt.ylim(0, 2)

    plt.grid()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(
        np.arange(len(catalog) - window_size), b_pos, label="b-positive", c="r"
    )
    plt.ylabel("b-value")
    plt.legend()
    plt.grid()
    plt.ylim(0, 2)
    plt.xlim(0, len(catalog))

    ax3 = plt.subplot(3, 1, 3)
    ax3.scatter(
        catalog["EqID"],
        catalog["Magnitude"],
        marker="o",
        color=[0, 0, 0, 0],
        s=5,
        edgecolors="grey",
    )
    plt.ylabel("Magnitude")
    plt.xlabel("Earthquake Number")
    plt.xlim(0, len(catalog))
    plt.show()
