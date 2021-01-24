# High level module (mostly for importing the other modules)

# Python modules
import os

# Bioverse modules
import analysis
import classes
from hypothesis import h_HZ, h_age_oxygen
import util
import pdfs
import plots
import priors

def quickrun(N_proc=1,timed=False):
    g = classes.Generator()
    if N_proc == 1:
        d = g.generate()
    else:
        d = g.generate_multi(N_proc=N_proc)
    return d


if __name__ == "__main__":
    d = quickrun()