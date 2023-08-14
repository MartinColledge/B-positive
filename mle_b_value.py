#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:06:55 2023

@author: martin
"""

# Import usual modules

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)


def mle_b_value_continuous(catalog, threshold_magnitude):
    """
    Mle b-value estimation with uncertainty from continuous distribution.

    Estimate b-value from the Magnitudes using a continuous magnitude
    distribution as first described in Utsu (1966) and Aki (1965).

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'
    Nava et al (2017),
    'Gutenberg-Richter b-value maximum likelihood estimation and sample size.'
    doi : 10.1007/s10950-016-9589-1

    Parameters
    ----------
    catalog : pandas DataFrame
        Earthquake catalo.
    threshold_magnitude : float
        Magnitude above which to cut the catalog and under which the catalog
        is considered incomplete.
    delta_magnitude : float, optional
        Precision of magnitudes to which the values are rounded.
        The default is False.

    Returns
    -------
    b_value : float
        GR b value.
    b_value_uncertainty : float
        GR b value uncertainty.

    """
    catalog_above_threshold = catalog[
        catalog["Magnitude"] >= threshold_magnitude
    ]

    b_value = (np.log10(np.e)) / (
        (catalog_above_threshold["Magnitude"].mean()) - threshold_magnitude
    )
    b_value_uncertainty = b_value / np.sqrt(len(catalog_above_threshold))

    return b_value, b_value_uncertainty


def mle_b_value_binned_utsu(catalog, threshold_magnitude, delta_magnitude):
    """
    Mle b-value estimation from binned distribution with utsu formula.

    Estimate b-value from the binned distribution while explicitely taking
    into account the theta2 bias that comes from the magnitude binning.
    That is, when data is binned, the lowest bin magnitude
    M_min =M_threshold -delta_magnitude/2

    Similar results to the Tinti version but Tinti version should be prefered.

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'

    Parameters
    ----------
    catalog : pandas DataFrame
        Earthquake catalo.
    threshold_magnitude : float
        Magnitude above which to cut the catalog and under which the catalog
        is considered incomplete.
    delta_magnitude : float, optional
        Precision of magnitudes to which the values are rounded.
        The default is False.

    Returns
    -------
    b_value : float
        GR b value.
    b_value_uncertainty : float
        GR b value uncertainty.

    """
    catalog_above_threshold = catalog[
        catalog["Magnitude"] >= threshold_magnitude
    ]

    b_value = np.log10(np.e) / (
        catalog_above_threshold["Magnitude"].mean()
        - (threshold_magnitude - delta_magnitude / 2)
    )
    b_value_uncertainty = b_value / np.sqrt(len(catalog_above_threshold))

    return b_value, b_value_uncertainty


def mle_b_value_binned_tinti(catalog, threshold_magnitude, delta_magnitude):
    """
    Mle b-value estimation with uncertainty from binned distribution.

    Estimate b-value from the Magnitudes using a binned magnitude
    distribution as described in Tinti and Mulargia (1987).

    To be prefered over the binned Utsu formula.

    Sources :
    Marzocchi and Sandri (2003),
    'A review and new insight on the estimation of the b-value and its
    uncertainty.'

    Parameters
    ----------
    catalog : pandas DataFrame
        Earthquake catalo.
    threshold_magnitude : float
        Magnitude above which to cut the catalog and under which the catalog
        is considered incomplete.
    delta_magnitude : float, optional
        Precision of magnitudes to which the values are rounded.
        The default is False.

    Returns
    -------
    b_value : float
        GR b value.
    b_value_uncertainty : float
        GR b value uncertainty.

    """
    catalog_above_threshold = catalog[
        catalog["Magnitude"] >= threshold_magnitude
    ]

    p_tinti = 1 + (delta_magnitude) / (
        catalog_above_threshold["Magnitude"].mean() - threshold_magnitude
    )
    b_value = np.log(p_tinti) / (delta_magnitude * np.log(10))

    b_value_uncertainty = np.abs(
        (1 - p_tinti)
        / (
            np.log(10)
            * delta_magnitude
            * np.sqrt(len(catalog_above_threshold) * p_tinti)
        )
    )
    return b_value, b_value_uncertainty
