#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:18:18 2023

@author: Martin Colledge

Partial recreation of following paper for Amatrice-Norcia sequence :

van der Elst, N. J. (2021). B-positive: A robust
estimator of aftershock magnitude distribution in transiently incomplete
catalogs. Journal of Geophysical Research: Solid Earth, 126, e2020JB021027.
https://doi. org/10.1029/2020JB021027
"""

# Import usual modules
import os
import datetime
import configparser
import numpy as np
import pandas as pd

import b_positive
import mle_b_value
from manage_paths import get_file_names

config = configparser.ConfigParser()
config.read("constants.ini")

# Get path and extension of data input
PATH_TO_DATA = [config["PATHS"]["PATH_TO_CATALOG"]]
FILE_EXTENSION = [config["PATHS"]["CATALOG_EXTENSION"]]

COMPLETENESS_MAGNITUDE = float(config["BVALUE"]["COMPLETENESS_MAGNITUDE"])
DELTA_MAGNITUDE = float(config["BVALUE"]["DELTA_MAGNITUDE"])
DIFFERENCE_CUTOFF_MIN = float(config["BVALUE"]["DIFFERENCE_CUTOFF_MIN"])
DIFFERENCE_CUTOFF_MAX = float(config["BVALUE"]["DIFFERENCE_CUTOFF_MAX"])
DIFFERENCE_CUTOFF_STEP = float(config["BVALUE"]["DIFFERENCE_CUTOFF_STEP"])
DIFFERENCE_CUTOFF = float(config["BVALUE"]["DIFFERENCE_CUTOFF"])
WINDOW_SIZE = int(config["BVALUE"]["WINDOW_SIZE"])

difference_cutoff_range = np.arange(
    DIFFERENCE_CUTOFF_MIN, DIFFERENCE_CUTOFF_MAX, DIFFERENCE_CUTOFF_STEP
)
# Get Script Name
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Get the file names
path_to_data, file_extension, file_names = get_file_names(
    PATH_TO_DATA, FILE_EXTENSION
)


def to_datetime(cat):
    """
    Convert date and time element columns into one DateTime object column.

    Parameters
    ----------
    cat : pandas dataframe
        Earthquake catalog.

    Returns
    -------
    None.

    """
    # Convert times to strings and concatenate them
    cat[["Year", "Month", "Day", "Hour", "Minute"]] = cat[
        ["Year", "Month", "Day", "Hour", "Minute"]
    ].astype(int)
    cat[["Year", "Month", "Day", "Hour", "Minute", "Second"]] = cat[
        ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    ].astype(str)
    # Convert date and time to datetime object, the format is inferred but be
    # careful to give date in logical way (prior problem with dd/MM/yyyy-...)
    cat["DateTime"] = pd.to_datetime(
        cat["Year"]
        + "/"
        + cat["Month"]
        + "/"
        + cat["Day"]
        + "-"
        + cat["Hour"]
        + ":"
        + cat["Minute"]
        + ":"
        + cat["Second"],
    )

    # Drop date and time columns as they are obsolete
    cat.drop(
        ["Year", "Month", "Day", "Hour", "Minute", "Second"],
        axis=1,
        inplace=True,
    )


# Get the file
for i, file_name in enumerate(sorted(file_names)):
    catalog_tmp = pd.read_csv(
        os.path.join(path_to_data, file_name),
        names=[
            "Longitude",
            "Latitude",
            "DecimalYear",
            "Month",
            "Day",
            "Magnitude",
            "Depth",
            "Hour",
            "Minute",
            "Second",
            "DistanceFromGrid",
        ],
        sep=r"\s+",
    )
    short_name = file_name[: -len(file_extension)]
    if i == 0:
        catalog = catalog_tmp
    else:
        catalog = pd.concat((catalog, catalog_tmp))

# catalog["Magnitude"] = np.round(catalog["Magnitude"], 1)
catalog["Year"] = np.floor(catalog["DecimalYear"])

# Convert the date and time into a DateTime element
to_datetime(catalog)
# Remove earthquakes one second apart to remove potential duplicates
catalog = catalog[catalog["DateTime"].diff() > datetime.timedelta(seconds=1)]
catalog = catalog.sort_values(by="DateTime")
catalog = catalog.reset_index(drop=True)
catalog = catalog.reset_index().rename(columns={"index": "EqID"})
# Use DateTime as index
catalog.set_index("DateTime", inplace=True)

# %% Figure 5

# Recreate figure 5 but for Amatrice and Norcia
# b_positive.test_discrete_b_pos_neg_with_difference_cutoff(
#     catalog, difference_cutoff_range, delta_magnitude=0.1
# )
# b_positive.test_continuous_b_pos_neg_with_difference_cutoff(
#     catalog, difference_cutoff_range
# )

# # %% Figure 8 a,b and d

# # mle is different due to constant Mc
# # b-positive is given roughly every 40 days in article, every day here.
# # Main conclustion of lack of b value drop after main shocks clearly visible
# b_positive.temporal_variation_of_b_value(
#     catalog,
#     DIFFERENCE_CUTOFF,
#     COMPLETENESS_MAGNITUDE,
#     DELTA_MAGNITUDE,
#     WINDOW_SIZE,
# )

# %% Figure 9

# Figure 9 b - pre foreshock
foreshock_datetime = catalog[catalog["Magnitude"] == 6.2].index.values[0]
# Select event prior to foreshock and exclude foreshock
catalog_pre_foreshock = catalog.loc[: foreshock_datetime - 1]
# Chose a completeness magnitude of 1.75
catalog_pre_foreshock = catalog_pre_foreshock[
    catalog_pre_foreshock["Magnitude"] >= 1.75
]
bvalue_preforeshock = mle_b_value.mle_b_value_binned_tinti(
    catalog_pre_foreshock, 1.75, DELTA_MAGNITUDE
)
print(bvalue_preforeshock)
# 1.10+-0.04, just as in the paper


# Figure 9 b - post main shock + 1 day
mainshock_datetime = catalog[catalog["Magnitude"] == 6.6].index.values[0]
catalog_post_mainshock = catalog.loc[
    mainshock_datetime + np.timedelta64(1, "D") :
]
catalog_post_mainshock = catalog_post_mainshock[
    catalog_post_mainshock["Magnitude"] >= 1.75
]
bvalue_post_mainshock = mle_b_value.mle_b_value_binned_tinti(
    catalog_post_mainshock, 1.75, DELTA_MAGNITUDE
)
print(bvalue_post_mainshock)
# 1.06+-0.01, just as in the paper except for 0.01 in the uncertainty.


# Figure 9 c - pre foreshock
foreshock_datetime = catalog[catalog["Magnitude"] == 6.2].index.values[0]
catalog_pre_foreshock = catalog.loc[: foreshock_datetime - 1]
pre_foreshock_b_positive = b_positive.discrete_b_positive(
    catalog_pre_foreshock, DIFFERENCE_CUTOFF, DELTA_MAGNITUDE, bootstrap=True
)
print(pre_foreshock_b_positive)
# 1.26, same as paper.

# Figure 9 c - post mainshock
mainshock_datetime = catalog[catalog["Magnitude"] == 6.6].index.values[0]
catalog_post_mainshock = catalog.loc[mainshock_datetime + 1 :]
post_mainshock_b_positive = b_positive.discrete_b_positive(
    catalog_post_mainshock, DIFFERENCE_CUTOFF, DELTA_MAGNITUDE, bootstrap=True
)
print(post_mainshock_b_positive)
# 1.12, same as paper.
