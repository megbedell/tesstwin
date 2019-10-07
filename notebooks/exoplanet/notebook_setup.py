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

import theano

print("theano version: {0}".format(theano.__version__))

import pymc3

print("pymc3 version: {0}".format(pymc3.__version__))

import exoplanet

print("exoplanet version: {0}".format(exoplanet.__version__))

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

__version__ = "0.0.1"
tic_id = 320004517
base_dir = "output/{0}/tic/{1}/".format(__version__, tic_id)

plot_dir = '../../paper/figures/'
