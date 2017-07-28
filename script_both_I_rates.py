#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# SCRIPT: COMPUTE WEAK AND STRONG ERROR OF APPROX. OF I TO I AND MAKE PLOTS
# Note: Please edit as deemed necessary.
#
# Paper: A regularity structure for rough volatility.
# Authors: C. Bayer, P. Friz, P. Gassiat, J. Martin, B. Stemper (2017).
# Maintainer: B. Stemper (stemper@math.tu-berlin.de)

# ------------------------------------------------------------------------------
# IMPORTS

# Import standard library packages and helper functions.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress as ls
from scipy.integrate import quad
from math import exp
import time

# Import some custom modules written for this script.
from logger import logger
from roughvol import IV

# ------------------------------------------------------------------------------
# INITIALISE VARIABLES AND WRITE LOGFILE

# Declare simulation ID.
sim_id = 80

# Declare model- and approximation-specific parameters.
parameter = {
    "hurst_index": 0.5,
    "time_steps": 2**17,
    "mc_runs": 10**3,
    "max_haar_level": 4,
    "nb_batches": 40,
    "f0": "lambda x: np.exp(x)",
    "f1": "lambda x: np.exp(x)"
    }

# Write parameter values to log file.
logger(sim_id, parameter)

# Translate key/value pairs from dictionary to variables.
for key, value in parameter.items():
    exec(key + '=' + str(value))

# Define some more convenience variables.
batch_size = int(mc_runs/nb_batches)
max_haar_terms = 2**max_haar_level

# Create Pandas Data-Frame to store level information
df = pd.DataFrame(np.zeros((4, max_haar_level)),
                  columns=np.arange(0, max_haar_level),
                  index=['M2_diff_strong', 'M4_diff_strong', 'M1weak', 'M2weak']
                  )

# Initialize running totals for quantities of interest.
weak_sums1 = np.zeros(max_haar_level)
weak_sums2 = np.zeros(max_haar_level)

strong_diff_sums2 = np.zeros(max_haar_level)
strong_diff_sums4 = np.zeros(max_haar_level)

# Initialise final level object (needed for computation of object I).
final_IV = IV(max_haar_level, time_steps, hurst_index, f0, f1)

print('Finished initialisation of simulation %i, proceed with simulation.'
      % sim_id)

# ------------------------------------------------------------------------------
# RUN MONTE CARLO SIMULATIONS OF OBJECT OF INTEREST AND COMPUTE MOMENTS

# Split up total number of simulations in chunks of prespecified size.
for _ in range(nb_batches):

    # Start batch timer.
    tic = time.time()

    # Compute normals for the finest Haar grid needed.
    normals = np.random.randn(batch_size, max_haar_terms)

    # Compute strong reference result.
    # I_vals_strong = final_IV.compute_I(normals)

    # Iterate through Haar levels.
    for level in range(max_haar_level):

        # Initialise level object.
        LevelObject = IV(level, time_steps, hurst_index, f0, f1)

        # Construct level-specific normals out of global normals array.
        level_diff = max_haar_level - level
        batch_normals = 2**(-level_diff/2) * np.sum(normals.reshape(batch_size,
                                                    2**level, 2**level_diff),
                                                    axis=2)

        # Compute value of I_eps on specific level.
        I_eps_vals = LevelObject.compute_I(batch_normals)

        # Compute strong differences and weak results.
        # strong_diff = I_eps_vals - I_vals_strong
        weak = I_eps_vals**2

        # Add to respective running totals.
        weak_sums1[level] += np.sum(weak)
        weak_sums2[level] += np.sum(weak**2)

        # strong_diff_sums2[level] += np.sum(strong_diff**2)
        # strong_diff_sums4[level] += np.sum(strong_diff**4)

    # Close timer and print out batch details.
    elapsed = time.time() - tic
    remaining = (nb_batches - (_ + 1)) * elapsed

    print('Batch %i/%i: Time %.1f s. Remaining: %.1f s' % (_ + 1, nb_batches, 
                                                           elapsed, remaining))

# Convert running totals to moments and store in dataframe object.
df.ix['M1weak', :] = weak_sums1/mc_runs
df.ix['M2weak', :] = weak_sums2/mc_runs
# df.ix['M2_diff_strong', :] = strong_diff_sums2/mc_runs
# df.ix['M4_diff_strong', :] = strong_diff_sums4/mc_runs

# Save computed data to pickle.
df.to_pickle('%s_data.pkl' % sim_id)

print(df)

print('Finished simulation, now generate graphics.')

# ------------------------------------------------------------------------------
# READ IN PICKLED DATA TO DATAFRAME AND GENERATE DESIRED GRAPHICS

# Pick data to plot by their ID and supplement with additional info.
IDs = [sim_id]
colors = ['g']
hursts = [hurst_index]

# Set the graphics environment through seaborn methods.
sns.set_context("paper")
sns.set(font='serif')
sns.set_style({"font.family": "serif",
               "font.serif": ["Times", "Palatino", "serif"]})
sns.set_style('whitegrid')
sns.set_palette("husl")

# Create matplotlib figure & axes objects with specified information etc.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Strong and weak error: non-constant renormalization')
ax.set_xlabel('$\epsilon = 2^{-N}$')
ax.set_ylabel('Error')
ax.set_xlim([10**(-3), 10])
ax.set_ylim([10**(-4), 10])
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')

# Generate data for x-axis.
eps = np.array([2**(-N) for N in range(max_haar_level)])

# Step through list of IDs, import data, process data and plot relevant info.
for nb, idx in enumerate(IDs):

    # Import data from dataframe and initialise variables.
    df = pd.read_pickle('%s_data.pkl' % idx)
    # M2_diff_strong = df.loc['M2_diff_strong', :].values
    # M4_diff_strong = df.loc['M4_diff_strong', :].values
    M1weak = df.loc['M1weak', :].values
    M2weak = df.loc['M2weak', :].values

    # Compute weak reference value and weak absolute difference.
    I_sq = quad(lambda t: exp(2*t**(2*hursts[nb])), 0, 1)[0]
    # print(I_sq)
    weak_abs_diff = np.absolute(M1weak - I_sq)
    # print(weak_abs_diff)

    # Construct CLT based normal 95% confidence intervals for weak error.
    var_weak = (M2weak - M1weak**2)/mc_runs
    std_weak = np.sqrt(var_weak)
    offset_weak = 1.96 * std_weak

    # Construct CLT based normal 95% confidence intervals for strong error.
    # var_strong = (M4_diff_strong - M2_diff_strong**2)/mc_runs
    # std_strong = np.sqrt(var_strong)
    # offset_strong = 1.96 * std_strong

    # Define how many levels to include for LS rate estimation of str error.
    cut = 5

    # Estimate through Least squares the strong rate.
    # str_sl, str_intcpt = ls(np.log(eps)[:cut],
                            # np.log(np.sqrt(M2_diff_strong))[:cut])[0:2]

    # Estimate through Least squares the weak rate.
    w_sl, w_intcpt = ls(np.log(eps), np.log(weak_abs_diff))[0:2]

    # Construct LS regression line.
    # fitted_line_str = np.exp(str_intcpt + str_sl * np.log(eps))
    fitted_line_weak = np.exp(w_intcpt + w_sl * np.log(eps))

    # Plot the weak error estimates and the confidence band.
    ax.plot(eps, weak_abs_diff, color=colors[nb], linestyle='', marker='+',
            label='H = %.1f, weak error, rate $\\approx$ %.2f ' % (hursts[nb],
            w_sl), markersize=6, mew=1)

    ax.fill_between(eps, weak_abs_diff-offset_weak,
                    weak_abs_diff + offset_weak, alpha=0.1,
                    facecolor=colors[nb], antialiased=True)

    ax.plot(eps, fitted_line_weak, color=colors[nb], linestyle='--',
            linewidth=0.8)

    # Plot the strong error estimates and the confidence band.
    # ax.plot(eps, np.sqrt(M2_diff_strong), color=colors[nb], linestyle='',
    #         mew=1, marker='x', markersize=6,
    #         label='H = %.1f, strong error, rate $\\approx$ %.2f' % (hursts[nb],
    #         str_sl))

    # ax.fill_between(eps, np.sqrt(M2_diff_strong) - np.sqrt(offset_strong),
    #                 np.sqrt(M2_diff_strong) + np.sqrt(offset_strong),
    #                 alpha=0.3, facecolor=colors[nb], antialiased=True)

    # ax.plot(eps, fitted_line_str, colors[nb], linewidth=0.8, linestyle='--')

ax.legend(loc='lower right', frameon=True)

# Exporting image to PDF.
fig.savefig("%i_I_bothrates.pdf" % idx, bbox_inches='tight', dpi=500)
# fig.savefig("test.pdf", bbox_inches='tight', dpi=500)

print("Image saved. Complete.")


