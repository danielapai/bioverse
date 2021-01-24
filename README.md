# Bioverse
Bioverse uses simulations of nearby planetary systems to assess the statistical power of next-generation space observatories for testing hypothesis about the properties and evolution of terrestrial planets. For a full code description, see the paper below.

If you'd like advice on using this code in your research, contact abixel@email.arizona.edu.

**Note**: users should downgrade to `scipy` v1.3.3 or earlier for optimal performance, because its `stats.truncnorm.rvs` method is 10-100X slower in v1.4 and newer. A fix is in progress.

# Requirements
`Bioverse` should work with Python 3.6+ (it has been tested in Python 3.7). It has the following dependencies, all of which can be installed using `pip`:

- `numpy`
- `scipy` (version **1.3** - see the note above)
- `matplotlib`
- `emcee`
- `dynesty`
- `tqdm` (optional: provides a progress bar for long processes)
- `pandas` (optional: used to visualize data in some of the example Notebooks)

# Usage

The best way to get familiar with `Bioverse` is to follow the example in `Notebooks/Getting started.ipynb`.

# References

Papers making use of `Bioverse` should reference the code description in the AJ paper.

It would also be great to include the references for the [`emcee`](https://github.com/dfm/emcee) and [`dynesty`](https://github.com/joshspeagle/dynesty) packages which are used for hypothesis testing and parameter fitting.
