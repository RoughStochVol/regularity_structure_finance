#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# SCRIPT
#
# COMPUTE WEAK ERROR OF OPTION PRICE
#
# Paper: A regularity structure for rough volatility.
# Authors: C. Bayer, P. Friz, P. Gassiat, J. Martin, B. Stemper (2017).
# Maintainer: B. Stemper (stemper@math.tu-berlin.de)

# ------------------------------------------------------------------------------
# IMPORTS

# Standard library imports.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress as ls
import time

# Custom imports.
from logger import logger
from roughvol import pricer
    
# ------------------------------------------------------------------------------
# INITIALISE VARIABLES AND WRITE LOGFILE

# Declare simulation ID.
sim_id = 82

# Declare model- and approximation-specific parameters.
parameter = {
    "spot_price": 1,
    "strike": 1,
    "hurst_index": 0.4 ,
    "spot_vol": 0.2,
    "vvol": 2,
    "correlation": -0.8,
    "time_steps": 2**17,
    "mc_runs": 10**2,
    "max_haar_level": 8,
    "nb_batches": 10 
    }

# Write parameter values to log file.            
logger(sim_id, parameter)

# Translate key/value pairs from dictionary to variables.
for key, value in parameter.items():
    exec(key + '=' + str(value))

# Define some more convenience variables.
batch_size = int(mc_runs/nb_batches)
max_haar_terms = 2**max_haar_level
   
# Create Pandas Data-Frame to store level information.
df = pd.DataFrame(np.zeros((2, max_haar_level + 1)),
                  columns = np.arange(0, max_haar_level + 1), 
                  index = ['M1', 'M2']
                  )

# Initialize running totals for moments.
sums1 = np.zeros(max_haar_level + 1)
sums2 = np.zeros(max_haar_level + 1)

print('Finished initialisation of simulation %i, proceed with simulation.'
      % sim_id)

# ------------------------------------------------------------------------------
# RUN MONTE CARLO SIMULATIONS OF OBJECT OF INTEREST AND COMPUTE MOMENTS

# Split up the total number of simulations in chunks of prespecified size.
for _ in range(nb_batches):

    # Start batch timer.
    tic = time.time()
    
    # Compute normals for the finest Haar grid needed.
    normals = np.random.randn(batch_size, max_haar_terms)

    # Iterating through Haar levels.
    for level in range(max_haar_level + 1):
    
        # Initialise Pricing object.
        LevelObject = pricer(spot_price, strike, hurst_index, spot_vol, vvol, 
                             correlation, level, time_steps)
    
        # Construct level-specific normals out of global normals array.
        level_diff = max_haar_level - level
        batch_normals = 2**(-level_diff/2) * \
                        normals.reshape(batch_size, 2**level, 
                                        2**level_diff).sum(axis=2) 
        
        results = LevelObject.compute(batch_size, batch_normals)[0]
        
        # Add to running totals for specific level.
        sums1[level] += np.sum(results)
        sums2[level] += np.sum(results**2)
        
    # Close timer and print out batch details.
    elapsed = time.time() - tic
    remaining = (nb_batches - (_ + 1)) * elapsed

    print('Batch %i/%i: Time %.1f s. Remaining: %.1f s' % (_ + 1, nb_batches, 
                                                           elapsed, remaining))
    
# Convert running totals to moments and store in dataframe object.
df.ix['M1', :] = sums1/mc_runs
df.ix['M2', :] = sums2/mc_runs

# Save computed data to pickle.
df.to_pickle('%s_data.pkl' %sim_id)
    
print(df)

print('Finished simulation, now generate graphics.')

# ------------------------------------------------------------------------------
# READ IN PICKLED DATA TO DATAFRAME AND GENERATE DESIRED GRAPHICS

# Define how many levels to display (< max_haar_level)
cut = 7

# Select IDs to print
IDs = [sim_id]
colors = ['m']
hursts = [hurst_index]

# Setting the graphics environment through Seaborn methods.
sns.set_context("paper")
sns.set(font='serif')
sns.set_style({"font.family": "serif",
               "font.serif": ["Times", "Palatino", "serif"]})
sns.set_style('whitegrid')
sns.set_palette("husl")

# Create matplotlib figure & axes objects with specified information etc.
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Option price: Weak error')
ax.set_xlabel('$\epsilon = 2^{-N}$')
ax.set_ylabel('Error')
ax.set_xlim([10**(-3), 10])
#ax.set_ylim([0, 1.5])
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')

# Generate data for x-axis.
eps = np.array([2**(-N) for N in range(cut)])

# Step through list of IDs, import data, process data and plot relevant info.
for nb, idx in enumerate(IDs):
    
    # Import data from dataframe and initialise variables.
    df = pd.read_pickle('%s_data.pkl' % idx)
    
    M1 = df.loc['M1',:].values[:cut]
    M2 = df.loc['M2',:].values[:cut]
    
    # Construct normal confidence intervals based on CLT.
    var_mean = (M2 - M1**2)/mc_runs
    std_mean = np.sqrt(var_mean)
    offset = 1.96 * std_mean

    # Create absolute differences.
    absdiffs = np.abs(M1 - df.loc['M1',max_haar_level])

    # Estimate through Least squares the weak rate.
    slope, intercept = ls(np.log(eps), np.log(absdiffs))[0:2]

    # Construct LS regression line.
    fitted_line = np.exp(intercept + slope * np.log(eps))

    # Plot the weak error estimates and the confidence band.

    ax.plot(eps, absdiffs, color=colors[nb], linestyle='', marker='+',
            label='H = %.1f, rate $\\approx$ %.2f ' % (hursts[nb],
            slope), markersize=6, mew=1)

    ax.plot(eps, fitted_line, color=colors[nb], linestyle='--', linewidth=0.8)
 
    ax.fill_between(eps, absdiffs-offset,
                    absdiffs + offset, alpha=0.1,
                    facecolor=colors[nb], antialiased=True)

ax.legend(loc='lower right', frameon=True)

# Saving
fig.savefig("test2.pdf", bbox_inches='tight', dpi=500)
#fig.savefig("figures/%s_option_weak.pdf" %sim_id, bbox_inches='tight', dpi=500)

