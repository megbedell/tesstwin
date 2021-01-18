get_ipython().magic("matplotlib inline")
get_ipython().magic('config InlineBackend.figure_format = "retina"')

# Hide deprecation warnings from Theano
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Hide Theano compilelock warnings
import logging

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import astropy.units as u

import theano
import theano.tensor as tt

print("theano version: {0}".format(theano.__version__))

import pymc3 as pm

print("pymc3 version: {0}".format(pm.__version__))

import exoplanet as xo

print("exoplanet version: {0}".format(xo.__version__))

import scipy

print("scipy version: {0}".format(scipy.__version__))

import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
#plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1

__version__ = "0.0.1"
tic_id = 320004517
base_dir = "output/{0}/tic/{1}/".format(__version__, tic_id)

plot_dir = '../../paper/figures/'
