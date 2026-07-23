# System modules
import copy
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter, zoom
import numpy as np
import pickle
from bioverse.classes import Table
from matplotlib.gridspec import GridSpec
import datetime

from warnings import warn

# Bioverse modules and constants
from . import analysis
from . import classes
from . import util
from .constants import INT_TYPES, STR_TYPES

# Unmap the save key
try:
    plt.rcParams['keymap.save'].remove('s')
except ValueError:
    pass

# Pyplot parameters
labelfontsize = 12


def plot(d, starID=None, order=None, fig=None, canvas=None):
    show = fig is None
    if fig is None:
        fig = plt.figure(figsize=(20, 10))
    ax = []
    if starID is None:
        starID = np.amin(d['starID'])
    if order is None:
        order = d['order'][d['starID'] == starID].min()
    idx = [starID, order]

    if canvas is None:
        canvas = fig.canvas

    # Top plot is a plot of all of the nearby stars
    ax.append(plt.subplot2grid((9, 16), (3, 0), rowspan=4, colspan=4, projection='polar', fig=fig))
    ax[0] = plot_universe(d, ax=ax[-1], mark=idx[0])

    # Middle plots show the star and system
    ax.append(plt.subplot2grid((9, 16), (0, 6), rowspan=3, colspan=11, fig=fig))
    # ax.append(plt.subplot2grid((9,16),(0,11),rowspan=3,colspan=6,fig=fig))
    ax[1] = plot_system(d, idx[0], ax=ax[1], mark=idx[1])

    # Bottom plots show the planet's atmosphere
    ax.append(plt.subplot2grid((9, 16), (5, 5), rowspan=4, colspan=4, fig=fig))
    ax.append(plt.subplot2grid((9, 16), (5, 11), rowspan=4, colspan=10, fig=fig))
    # ax[2:4] = plot_atmosphere(d,idx[0],idx[1],ax=ax[2:4])

    # Left/right to cycle through systems or planets
    starIDs = np.unique(d['starID'])
    global orders
    orders = d['order'][d['starID'] == idx[0]]

    def onpress(event):
        global orders
        if event.inaxes == ax[1]:
            if event.key in ['left', 'right']:
                idx[0] = util.cycle_index(starIDs, idx[0], 1 if event.key == 'right' else -1)
                orders = d['order'][d['starID'] == idx[0]]
                idx[1] = min(orders)
                for i in range(0, 4): ax[i].clear()
                ax[0] = plot_universe(d, ax=ax[0], mark=idx[0])
                ax[1] = plot_system(d, idx[0], ax=ax[1], mark=idx[1])
                # ax[2:4] = plot_atmosphere(d,idx[0],idx[1],ax=ax[2:4])
                canvas.draw()
        if event.inaxes in ax[2:4]:
            if event.key in ['left', 'right']:
                idx[1] = util.cycle_index(orders, idx[1], 1 if event.key == 'right' else -1)
                for i in range(1, 4): ax[i].clear()
                ax[1] = plot_system(d, idx[0], ax=ax[1], mark=idx[1])
                # ax[2:4] = plot_atmosphere(d,idx[0],idx[1],ax=ax[2:4])
                canvas.draw()

    cid = canvas.mpl_connect('key_press_event', onpress)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=1.2, left=0.05, right=0.95)
    if show:
        plt.show()
    else:
        return ax


def plot_universe(d, N_max=100, ax=None, mark=None):
    # Plots the map of nearby stars, with sizes and colors corresponding to mass/temperature
    if ax is None:
        show = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        show = False

    # Just display the first N_max stars
    st = d.get_stars()
    N_max = min(N_max, len(st))
    st = st[:N_max]

    # Determine the colors (reddest at 2500 K, greenest at 5500 K, bluest at 8000 K)
    r = np.interp(st['T_eff'], [2500, 5500, 8000], [1, 1, 0])
    g = np.interp(st['T_eff'], [2500, 5500, 8000], [0, 1, 0])
    b = np.interp(st['T_eff'], [2500, 5500, 8000], [0, 0, 1])
    colors = [(r[i], g[i], b[i]) for i in range(len(r))]

    # Determine sizes (smallest at 0.1, largest at 2.0)
    sizes = np.interp(st['R_st'], [0.1, 2.0], [10, 100])

    x, y = st['x'], st['y']
    r = (x ** 2 + y ** 2) ** 0.5
    theta = np.arctan2(y, x)
    ax.scatter(theta, r, color=colors, s=sizes)
    ax.scatter(0, 0, marker='+', c='black', lw=2, s=max(sizes))
    ax.set_rlim([0, max(r)])

    # Mark the currently displayed star
    if mark is not None and mark < N_max:
        ax.scatter(theta[mark], r[mark], color='None', edgecolor='black', lw=2, s=1.5 * max(sizes), zorder=10)

    # ax.set_xlabel('x (pc)')
    # ax.set_ylabel('y (pc)')
    # ax.set_zlabel('z (pc)')
    """
    if show:
        # Middle click to show more about the nearest system
        # TODO: factor in the z distance too
        x2,y2,_ = proj3d.proj_transform(x,y,z,ax.get_proj())
        def onclick(event):
            if event.button == 2 and event.inaxes == ax:
                xx,yy = event.xdata,event.ydata
                r2 = ((xx-x2)**2+(yy-y2)**2)
                idx = np.argmin(r2)
                plot_system(d0,idx)

        cid = ax.figure.canvas.mpl_connect('button_press_event',onclick)
    """
    if show:
        plt.show()
    else:
        return ax


def plot_system(d, starID, ax=None, mark=None):
    """ Scatter plot for a single system with one or more planets.

    Parameters
    ----------
    d : Table
        Table of simulated planets.
    starID : int
        Unique identifier of this system in the table (d['starID']).
    ax : Axes, optional
        Matplotlib Axes to plot the figure on. If not given, a new figure is created.
    mark : int, optional
        Indicates which planet to circle (if any).
    """
    if ax is None:
        show = True
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)
    else:
        show = False

    # Get a table of the planets in this system
    planets = d[d['starID'] == starID]
    star = planets[0]

    # Plot the star at (0,0) with a representative size and color
    r = np.interp(star['T_eff'], [2500, 5500, 8000], [1, 1, 0])
    g = np.interp(star['T_eff'], [2500, 5500, 8000], [0, 1, 0])
    b = np.interp(star['T_eff'], [2500, 5500, 8000], [0, 0, 1])
    color = np.array([r, g, b]).T
    size = np.interp(star['R_st'], [0.1, 2.0], [50, 500])
    ax.annotate(' ', xy=(0.04, 0.5), xycoords='axes fraction', ha='center', va='center',
                bbox=dict(boxstyle="circle", fc=color, lw=0), annotation_clip=True)

    # Plot each planet at (a,di) where a is its semi-major axis, di is the inclination relative to the system mean
    inc = np.arccos(planets['cos(i)']) * 180. / np.pi
    for i in range(len(planets)):
        pl = planets[i]
        c_pl, c_or = util.get_planet_colors(pl)
        s_pl = 50
        di = inc[i] - np.mean(inc)
        ax.scatter(pl['a'], di, s=s_pl, c=c_pl)
        if mark == pl['order']:
            ax.scatter(pl['a'], di, s=s_pl * 3, facecolors='none', edgecolors='black', linewidths=2)

    # Habitable zone limits
    lw_hz = 3
    ax.axvline(pl['a_in'], lw=lw_hz, c='green', zorder=-100, linestyle='dashed')
    ax.axvline(pl['a_out'], lw=lw_hz, c='green', zorder=-100, linestyle='dashed')

    # Axis limits and labels
    ax.set_xscale('log')
    xmin = np.amin(planets['a'] / 5)
    xmax = 2 * np.amax(planets['a'])
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([-90,90])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel('Semi-major axis (AU)', fontsize=labelfontsize)
    ax.set_ylabel('Relative inclination\n(degrees)', fontsize=labelfontsize)

    # Title
    title = 'System name: {:s} ({:s})    |  {:s} star, {:d} planet{:s}' \
        .format(star['star_name'], star['SpT'], 'binary' if star['binary'] else 'single', len(planets),
                '' if star['N_pl'] == 1 else 's')

    if show:
        plt.suptitle(title, fontsize=labelfontsize)
    else:
        ax.annotate(title, xy=(0.5, 1.2), va='center', ha='center', xycoords='axes fraction', fontsize=labelfontsize)

    if show:
        plt.subplots_adjust(wspace=0.5)
        plt.show()
    else:
        # plt.subplots_adjust(bottom=0.2)
        return ax


def plot_spectrum(x, y, dy=None, xunit=None, yunit=None, lw=2):
    """ Plots a spectrum with or without errorbars. """
    fig, ax = plt.subplots()
    if dy is None:
        ax.plot(x, y, lw=lw)
    else:
        ax.errorbar(x, y, yerr=dy, linestyle='None', lw=lw)

    ax.set_xlabel('Wavelength' + ('' if xunit is None else '({:s})'.format(xunit)))
    ax.set_ylabel('Value' if yunit is None else yunit)

    if yunit == 'albedo':
        ax.set_ylim([0, 0.6])

    plt.show()


def occurrence_by_class(d, compare=True):
    """ Plots the number of planets per star as a function of size and instellation. """

    # Boundary radii
    R0 = np.array([0.5, 1.0, 1.75, 3.5, 6.0, 14.3])

    # Condensation points (Table 1)
    S0 = np.array([[182, 187, 188, 220, 220, 220],
                   [1.0, 1.12, 1.15, 1.65, 1.65, 1.7],
                   [0.28, 0.30, 0.32, 0.45, 0.40, 0.45],
                   [0.0035, 0.0030, 0.0030, 0.0030, 0.0025, 0.0025]])

    # Comparison values from LUVOIR Final Report
    cvals = [[0.65, 0.32, 1.91],
             [0.45, 0.23, 0.95],
             [0.41, 0.23, 1.17],
             [0.07, 0.08, 0.90],
             [0.07, 0.10, 0.98]]

    # First and second instellation boundaries
    R = np.linspace(min(R0), max(R0), 100)
    S_inner = np.interp(R, R0, S0[1, :])
    S_outer = np.interp(R, R0, S0[2, :])

    # Plot the boundaries in size and instellation
    fig, ax = plt.subplots(figsize=(8, 6))
    for radius in R0:
        ax.axhline(radius, c='black', lw=2)
    ax.plot(S_inner, R, c='black', lw=2)
    ax.plot(S_outer, R, c='black', lw=2)

    # Annotate the occurrence rate in each section
    N_st = np.ptp(d['starID'])
    for i, clas1 in enumerate(['hot', 'warm', 'cold']):
        xmid = [10, 0.6, 0.03][i]
        m1 = d['class1'] == clas1
        for j, clas2 in enumerate(['rocky', 'super-Earth', 'sub-Neptune', 'sub-Jovian', 'Jovian']):
            ymid = (R0[j] + R0[j + 1]) / 2.
            m2 = d['class2'] == clas2
            ax.annotate('{:.2f}'.format((m1 & m2).sum() / N_st), xy=(xmid, ymid), va='bottom', ha='center', fontsize=12)
            if compare:
                ax.annotate('{:.2f}'.format(cvals[j][i]), xy=(xmid, ymid), va='top', ha='center', fontsize=12,
                            fontweight='bold')

    # Axis limits
    ax.set_xlim([190, 0.003])
    ax.set_ylim([0.3, 20])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Instellation ($S_\oplus$)', fontsize=labelfontsize)
    ax.set_ylabel('Radius ($R_\oplus$)', fontsize=labelfontsize)

    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.show()


def plot_binned_average(d, key1, key2, log=True, bins=10, method='mean', match_bin_counts=False, ax=None,
                        return_xy=False,
                        xm=None, ym=None, **kwargs):
    """ Plots the average value of a parameter as a function of another parameter.

    Parameters
    ----------
    d : Table
        Table of simulated planets or measurements.
    key1 : str
        Name of the parameter with which to bin the data set.
    key2 : str
        Name of the parameter for which to calculate the average value.
    log : bool, optional
        Whether to bin the data in log-space.
    bins : int or float array, optional
        Number of bins or list of bin edges.
    method : {'mean','median'}, optional
        Whether to take the mean or median in each bin.
    match_bin_counts : bool, optional
        If True, calculate bins with an equal number of planets in each.
    xm : float array, optional
        X values of a model to be plotted along with the binned data.
    ym : float array, optional
        Y values of a model to be plotted along with the binned data.
    **kwargs
        Keyword arguments, passed to matplotlib.pyplot.errorbar.
    """

    # Extract the values and discard NaNs
    val1, val2 = d[key1], d[key2]
    mask = ~(np.isnan(val1) | np.isnan(val2))
    val1, val2 = val1[mask], val2[mask]
    if log:
        val1 = np.log10(val1)

    # Calculate the bins and the average value in each bin
    if isinstance(bins, INT_TYPES):
        if match_bin_counts:
            bins = np.percentile(val1, np.linspace(0, 100, bins + 1))
        else:
            binsize = np.ptp(val1) / bins
            bins = np.arange(np.amin(val1), np.amax(val1) + binsize, binsize)
    x = (bins[1:] + bins[:-1]) / 2.
    dx = (bins[1:] - bins[:-1]) / 2.
    func = np.mean if method == 'mean' else np.median if method == 'median' else None
    y = np.array([func(val2[(val1 >= bins[i]) & (val1 < bins[i + 1])]) for i in range(len(x))])
    dy = np.array([np.std(val2[(val1 >= bins[i]) & (val1 < bins[i + 1])]) for i in range(len(x))])
    N = np.array([np.sum((val1 >= bins[i]) & (val1 < bins[i + 1])) for i in range(len(x))])
    dy = dy / np.sqrt(N)
    dy[dy == 0] = np.median(dy[dy != 0])

    # Plot
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        show = False

    ax.errorbar(x, y, xerr=dx, yerr=dy, marker='s', linestyle='none', **kwargs)
    # Plot the model values if given
    if xm is not None and ym is not None:
        ax.plot(xm, ym, c='black', lw=5)

    if show:
        plt.show()
        return (x, y, dx, dy) if return_xy else None
    else:
        return (ax, x, y, dx, dy) if return_xy else ax


def Example1_priority(generator, survey, fig=None, ax=None, show=True):
    """ Plots the prioritization of targets according to a_eff and R (or R_eff). """

    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        show = False

    # R or R_eff?
    Rkey = 'R' if survey.mode == 'transit' else 'R_eff'

    # a_eff and R (R_eff) axes
    x = np.logspace(-1.5, 1.5, 100)
    y = np.linspace(0.5, 2.0, 100)
    x, y = np.meshgrid(x, y, indexing='ij')

    # Determine the priority of each grid cell
    data = classes.Table()
    data['a_eff'] = x.flatten()
    data[Rkey] = y.flatten()
    z = survey.measurements['has_H2O'].compute_weights(data)

    # Set 0 for invalid values
    valid = survey.measurements['has_H2O'].compute_valid_targets(data)
    z[~valid] = 0
    z = z.reshape(x.shape)

    # Plot the priority vs a, R_eff
    im = ax.pcolormesh(x, y, z, cmap='Greens', vmin=0, vmax=z.max(), lw=0, rasterized=True)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 1, 10])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r'$a_{\mathrm{eff}}$ (AU)', fontsize=labelfontsize)
    if Rkey == 'R_eff':
        Rkey = r'$R_{\mathrm{eff}}$'
    ax.set_ylabel(Rkey + ' ($R_\oplus$)', fontsize=labelfontsize)

    plt.subplots_adjust(bottom=0.2, left=0.25)
    if show:
        plt.show()
    else:
        return fig, ax


def Example1_targets(data, fig=None, ax=None, show=True, cbar=True, bins=10, vmin=None, vmax=None, cax=None,
                     smooth_sigma=None):
    """ Plots the distribution of targets in log(a_eff) and R (or R_eff). """

    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        show = False

    # R or R_eff?
    Rkey = 'R' if 'R' in data else 'R_eff'

    # Bins
    xbins = np.logspace(-1.5, 1.5, bins)
    ybins = np.linspace(0.5, 2.0, bins)
    x, y = util.compute_bin_centers(xbins)[0], util.compute_bin_centers(ybins)[0]
    x, y = np.meshgrid(x, y, indexing='ij')

    # 2D histogram of values of a_eff and R (or R_eff) for characterized targets
    obs = data[~np.isnan(data['has_H2O'])]
    h = ax.hist2d(obs['a_eff'], obs[Rkey], bins=(xbins, ybins), cmap='Greens')[0]

    # Smooth the data?
    if smooth_sigma:
        h = gaussian_filter(h, smooth_sigma)

    vmin = 0 if vmin is None else vmin
    vmax = h.max() if vmax is None else vmax
    im = ax.pcolormesh(x, y, h * (vmax / h.max()), cmap='Greens', vmin=vmin, vmax=vmax, lw=0, rasterized=True)
    if cbar:
        fig.colorbar(im, cax=cax)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 1, 10])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r'$a_{\mathrm{eff}}$ (AU)', fontsize=labelfontsize)
    if Rkey == 'R_eff':
        Rkey = r'$R_{mathrm{eff}}$'
    ax.set_ylabel(Rkey + ' ($R_\oplus$)', fontsize=labelfontsize)

    if show:
        plt.show()
    else:
        return fig, ax


def Example1_dataset(data, a_inner=1.12 ** -0.5, a_outer=0.37 ** -0.5, show=True, plot_model=True):
    """ Plots data['has_H2O'] versus data['S'] for a simulated data set. Also plots the habitable zone boundaries. """

    fig, ax = plt.subplots(figsize=(12, 3.5))

    obs = ~np.isnan(data['has_H2O'])
    data.compute('a_eff')
    x, y = data['a_eff'][obs], data['has_H2O'][obs]

    if plot_model:
        ax.fill_between([a_inner, a_outer], -2, 2, color='gray', alpha=0.4, lw=0)
    ax.axhline(0, linestyle='dotted', c='black', lw=1)
    ax.axhline(1, linestyle='dotted', c='black', lw=1)
    ax.scatter(x, y, marker='o', zorder=10, c='C2')

    xmin, xmax = 0.1, 10.0
    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 1])
    ax.set_xscale('log')
    ax.set_xlim([xmin, xmax])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticklabels(['no H$_2$O', 'H$_2$O'], x=-0.01)
    ax.set_xlabel('$a_\mathrm{eff}$ (AU)', fontsize=labelfontsize, labelpad=15)

    plt.subplots_adjust(bottom=0.33, top=0.67)
    if show:
        plt.show()
    else:
        return fig, ax


def Example2_priority(generator, survey, fig=None, ax=None, show=True):
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        show = False

    # age axis
    x = np.linspace(0, 10, 100)

    # Determine the priority of each grid cell
    data = classes.Table()
    data['age'] = x
    z = survey.measurements['has_O2'].compute_weights(data)

    # Plot the priority vs age
    ax.plot(x, z, lw=5, c='C0')

    ax.set_xlim([0, 10])

    ax.set_xlabel('Age (Gyr)', fontsize=labelfontsize, labelpad=15)
    ax.set_ylabel('Target priority', fontsize=labelfontsize, labelpad=15)

    if show:
        plt.show()
    else:
        return fig, ax


def Example2_targets(data, fig=None, ax=None, bins=10, show=True):
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        show = False

    # age bins
    xbins = np.linspace(0, 10, bins)

    obs = ~np.isnan(data['has_O2'])
    ax.hist(data['age'][obs], bins=xbins, color='darkgreen', alpha=0.5, lw=0)

    if show:
        plt.show()
    else:
        return fig, ax


def Example2_dataset(data, flife=0.8, thalf=5.0, show=True, plot_model=False):
    """ Plots data['has_O2'] versus data['age'] for a simulated data set. Also plots f(O2 | life)(t). """

    fig, ax = plt.subplots(figsize=(12, 3.5))

    # Data
    obs = ~np.isnan(data['has_O2'])
    x, y = data['age'][obs], data['has_O2'][obs]
    ax.scatter(x, y, c='C0')
    ax.axhline(0, linestyle='dotted', c='black', lw=1)
    ax.axhline(1, linestyle='dotted', c='black', lw=1)

    # Model
    if plot_model:
        xm = np.linspace(0, 13, 100)
        ym = flife * (1 - 0.5 ** (xm / thalf))
        ax.plot(xm, ym, c='grey', lw=5, zorder=-1)

    ax.set_ylim([-0.2, 1.2])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['no O$_3$', 'O$_3$'])

    if plot_model:
        ax2 = ax.twinx()
        ax2.set_ylim([-0.2, 1.2])
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['0%', '100%'], c='grey')
        # ax2.set_ylabel(r'$f_{{O_2 \cup O_3}}(t)$', fontsize=labelfontsize, c='grey', labelpad=15)
        ax2.set_ylabel(r'$f_{{O_3}}(t_*)$', fontsize=labelfontsize, c='grey', labelpad=15)

    ax.set_xlim([0, 10])
    ax.set_xlabel('Age (Gyr)', fontsize=labelfontsize, labelpad=15)

    plt.subplots_adjust(left=0.16, bottom=0.33, top=0.67, right=0.84)
    if show:
        plt.show()
    else:
        return fig, ax


def Example2_model(show=True, t_half=2.3, f_life=0.75):
    """ Plots the distribution of O2/O3-rich planets versus age. """

    # Calculate f(Archean), f(Proterozoic), f(Phanerozoic)
    x = np.linspace(0, 10, 300)
    y = [(1 - f_life) + f_life * np.exp(-x / t_half),
         f_life * (x / t_half) * np.exp(-x / t_half),
         f_life * (1 - (1 + (x / t_half)) * np.exp(-x / t_half))]

    # Plot versus age
    fig, ax = plt.subplots(figsize=(8, 6))
    lw, c = 5, ['C0', 'C1', 'C2']
    for i in range(3):
        ax.plot(x, y[i], lw=lw, c=c[i])

    # Annotations/labels
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1])

    ax.set_xlabel('Age (Gyr)', fontsize=labelfontsize)
    ax.set_ylabel('Fraction of\nexo-Earth candidates', fontsize=labelfontsize)

    plt.subplots_adjust(left=0.2, bottom=0.2)
    if show:
        plt.show()
    else:
        return fig, ax


def compare_posteriors(results_dict, **kwargs):
    fig, axes = None, None
    for i, (method, results) in enumerate(results_dict.items()):
        h = results['h']
        fig, axes = plot_posterior(results['chains'], histtype='step', lw=3, color='C{:d}'.format(i),
                                   bounds=h.bounds, log=h.log, show=False, fig=fig, axes=axes, plot_model=False,
                                   label=method, **kwargs)
    axes[0].legend(loc='best')
    plt.show()


def plot_posterior(chains, params=None, bounds=None, log=None, nbins=30, plot_model=True, show=True,
                   fig=None, axes=None, **hist_kwargs):
    """ Plots the posterior distributions of a set of parameters.

    Parameters
    ----------
    chains : float array (NxM)
        N samples from the posterior distribution of each of M parameters
    params : string list, optional
        List of M parameter names in the order that they appear in `chains`. Use None to designate which
        parameters to exclude. If not specified, label as theta_0, theta_1, ...
    bounds : float array (Mx2), optional
        Describes each parameter's min/max values.
    log : tuple, optional
        Length M tuple of True/False values indicating which posterior distributions should be plotted in log-scale.
    nbins : int, optional
        Number of bins the posterior distribution plot.
    plot_model : bool, optional
        If True, overplot a normal distribution with the same mean and variance as the sample.
    show : bool, optional
        If True, display the plot. Otherwise, return the figure and axes.
    fig : Figure, optional
        Figure in which to create the plot. `axes` must also be passed.
    axes : Axis list, optional
        List of Axis objects in which to create the plots. `fig` must also be passed.

    Returns
    -------
    fig, axes : Figure, Axes array
        Figure and flattened array of Axes objects containing the plots. Only returned if `show` if False.
    """
    # Determine which parameters to include
    if params is None:
        params = [r'$\theta_{:d}$'.format(i) for i in range(chains.shape[1])]
    params = np.array(params)
    exclude = np.in1d(params, None)
    chains, params = chains[:, ~exclude], params[~exclude]

    # Default boundaries
    if bounds is None:
        bounds = [None] * len(params)

    # Log-scale?
    if log is None:
        log = tuple([False] * len(params))

    # Create the figure/axes
    m = chains.shape[1]
    ncols = min(2, m)
    nrows = m // ncols + (m % ncols > 0)
    if fig is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 5))
        axes = axes.flatten()
    else:
        show = False

    # Plot each posterior distribution and medians, confidence intervals
    for param, ax, chain, bound, lg in zip(params, axes, chains.T, bounds, log):
        chain = chain[~np.isnan(chain)]
        if bound is None:
            bound = [np.amin(chain), np.amax(chain)]
        if lg:
            bound = np.log10(bound)
            x = np.log10(chain)
        else:
            x = chain

        bins = np.linspace(*bound, nbins)
        ax.hist(x, bins=bins, density=True, **hist_kwargs)
        ax.set_xlabel(param, fontsize=labelfontsize)

        if plot_model:
            median = np.nanmedian(x)
            LCI, UCI = np.abs(np.percentile(x, [16, 100 - 16], axis=0) - median)
            ax.axvline(median, c='black', lw=5)
            ax.axvline(median + UCI, c='black', lw=2, linestyle='dashed')
            ax.axvline(median - LCI, c='black', lw=2, linestyle='dashed')

        # if lg:
        # ax.set_xscale('log')
        # ax.set_xticks(np.logspace(*bound, 5))
        # sf = StrMethodFormatter('{x:.1f}')
        # sf.set_scientific(False)
        # ax.xaxis.set_major_formatter(sf)
        # ax.xaxis.set_minor_formatter(NullFormatter())

    # Disable unused axes
    for i, ax in enumerate(axes):
        if i >= len(params):
            ax.axis('off')

    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, hspace=0.4, wspace=0.2)
    if show:
        plt.show()
    else:
        return fig, axes


def plot_power_grid(results, axes=('f_water_habitable', 'f_water_nonhabitable'), log=(True, True), labels=None, cbar=True,
                    method='dlnZ', threshold=None, smooth_sigma=None, fig=None, ax=None, show=True, levels=[15, 60, 80],
                    cmap='Greens', zoom_factor=0, **grid_kwargs):
    # Validate length of `axes` and `labels`
    if isinstance(axes, STR_TYPES):
        axes = (axes,)
    if isinstance(labels, STR_TYPES):
        labels = (labels,)
    if labels is None:
        labels = axes

    ndim = len(axes)
    if ndim > 2:
        raise ValueError("'axes' must be a string or 1- or 2- element list")

    # Compute the statistical power
    power = analysis.compute_statistical_power(results, threshold, method)

    # Reduce the grid to the 1 or 2 dimensions in `axes`
    grid = results['grid'].copy()
    seq = [0] * (len(grid) - 1)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in axes:
            seq[i] = slice(None)
            continue
        elif key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        elif key != 'N' and len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]

    # Extract the x (, y) and z coordinates and smooth the data
    z = power[tuple(seq)] * 100
    x, y = grid[axes[0]], None if ndim == 1 else grid[axes[1]]
    if list(grid.keys())[0] != axes[0]:
        z = z.T
    if smooth_sigma:
        z = gaussian_filter(z, smooth_sigma)

    # Create the plot of power(x) or power(x, y)
    if ndim == 1:
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            show = False

        plt.locator_params(nbins=4)

        ax.plot(x, z, lw=5, c='black')

        ax.set_yticks([0, 25, 50, 75, 100])

        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([0, 100])

        ax.set_xlabel(labels[0], fontsize=labelfontsize)
        ax.set_ylabel('Power', fontsize=labelfontsize)

        plt.subplots_adjust(left=0.15, bottom=0.25, right=0.9)

    elif ndim == 2:
        fig, ax = image_contour_plot(x, y, z, labels=(*labels, 'Statistical power (%)'), fmt=' %.0f %% ', levels=levels,
                                     log=log, vmin=0, vmax=100, fig=fig, ax=ax, cmap=cmap, cbar=cbar,
                                     zoom_factor=zoom_factor, smooth_sigma=smooth_sigma)

    if show:
        plt.show()
    else:
        return fig, ax


def plot_requirements_grid(results, axes=('f_water_habitable', 'f_water_nonhabitable'), variable='t_total',
                           log=(True, True),
                           labels=None, var_label=None, levels=None, method='dlnZ', threshold=None, smooth_sigma=None,
                           show=True, min_power=0.80, fmt=' %.0f d ', N_key='N_EEC', vmin=None, vmax=None, fig=None,
                           ax=None,
                           cmap='Greens_r', zoom_factor=0, **grid_kwargs):
    # Validate length of `axes` and `labels`
    if isinstance(axes, STR_TYPES):
        axes = (axes,)
    if isinstance(labels, STR_TYPES):
        labels = (labels,)
    if labels is None:
        labels = axes
    if var_label is None:
        var_label = variable

    ndim = len(axes)
    if ndim != 2:
        raise ValueError("'axes' must be a 2- element list")

    # Reduce the grid to the specified 2 dimensions + time
    grid = results['grid'].copy()
    seq = [0] * (len(grid) - 1)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in axes or key == variable:
            if key == variable:
                axis = i
            seq[i] = slice(None)
            continue
        elif key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        elif len(val) > 1 and key != 'N':
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]

    # Compute the statistical power and smooth
    power = analysis.compute_statistical_power(results, threshold, method)[tuple(seq)]

    # Re-shape `power` to be (len(axes[0]), len(axes[1]), len(variable))
    axis = [list(grid.keys()).index(key) for key in (*axes, variable)]
    power = np.moveaxis(power, axis, np.arange(len(axis)))

    # Translate `variable` to the number of planets observed
    N_pl = np.mean(results[N_key], axis=(*axis[:2], -1))

    # In each grid cell, interpolate to determine the number of planets required to achieve the desired statistical power
    # N_required = t_total[np.argmin(np.abs(power-min_power), axis=-1)].astype(float)
    N_required = np.full(power.shape[:-1], np.nan)
    for i, j in np.ndindex(N_required.shape):
        x, y = power[i, j, :], N_pl
        if (x < min_power).all():
            continue
        N_required[i, j] = np.interp(min_power, x, y)

    # Mask (nan) values larger than vmax
    if vmax is not None:
        N_required[N_required > vmax] = np.nan

    # Plot the required time grid
    fig, ax, ctr = image_contour_plot(grid[axes[0]], grid[axes[1]], N_required, labels=(*labels, var_label), plus=True,
                                      levels=levels, fmt=fmt, ticks=4, log=log, return_ctr=True, vmin=vmin, vmax=vmax,
                                      cmap=cmap, fig=fig, ax=ax, zoom_factor=zoom_factor, smooth_sigma=smooth_sigma)

    # Set the contour labels to read N_EEC and variable
    def N_to_var(N):
        # Converts N_EEC to the corresponding value of `variable`
        idx = np.argsort(N_pl)
        return np.interp(N, N_pl, grid[variable])

    if variable == 'eta_Earth':
        lvl = ctr.levels[0]
        # ctr_labels = {lvl:r"$N_{\mathrm{EEC}}$"+" = {:.0f}\n ($\eta_\oplus$ = {:.1f}%)".format(lvl, 100*N_to_var(lvl))}
        ctr_labels = {lvl: "{:.0f}\n ($\eta_\oplus$ = {:.1f}%)".format(lvl, 100 * N_to_var(lvl))}
        ctr_labels.update({lvl: "{:.0f}\n ({:.1f}%)".format(lvl, 100 * N_to_var(lvl)) for lvl in ctr.levels[1:]})
    elif variable == 't_total':
        lvl = ctr.levels[0]
        # ctr_labels = {lvl:r"$N_{\mathrm{EEC}}$"+" = {:.0f}\n ($t$ = {:.1f} d)".format(lvl, N_to_var(lvl))}
        ctr_labels = {lvl: "{:.0f}\n(t = {:.0f} d)".format(lvl, N_to_var(lvl))}
        ctr_labels.update({lvl: "{:.0f}\n ({:.0f} d)".format(lvl, N_to_var(lvl)) for lvl in ctr.levels[1:]})
    ax.clabel(ctr, ctr.levels, fmt=ctr_labels, use_clabeltext=False, inline_spacing=-20)

    if show:
        plt.show()
    else:
        return fig, ax


def plot_habitable_zone_accuracy(results, a_inner=0.931, a_outer=1.674, smooth_sigma=None):
    x, y = results['grid']['f_water_habitable'], results['grid']['f_water_nonhabitable']

    seq = (0, slice(None), slice(None), slice(None), slice(None), slice(0, 2))
    chains = (10 ** results['chains'][seq]) ** -0.5
    means, stds = np.mean(chains, axis=-2), np.std(chains, axis=-2)

    dBIC = results['dBIC'][seq[:-2]]
    stds[dBIC < 6] = np.nan

    z = np.nanmean(stds, axis=-2)

    if smooth_sigma:
        z[..., 0] = gaussian_filter(z[..., 0], smooth_sigma)
        z[..., 1] = gaussian_filter(z[..., 1], smooth_sigma)

    fig, ax = image_contour_plot(x, y, z[..., 0], levels=[0.5, 0.8], fmt=' %.1f AU ')
    plt.show()


def plot_Example1_constraints(results, fig=None, ax=None, show=True, c='black', lw=1.0, truth=True, **grid_kwargs):
    # Reduce the grid to the 1 or 2 dimensions in `axes`
    grid = results['grid'].copy()
    seq = [0] * len(grid)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        elif key == 'N':
            # Pick a random simulation unless specified
            seq[i] = np.random.choice(grid['N'])
        elif len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]

    # Extract the MCMC samples for this simulation
    samples = results['chains'][tuple(seq)]

    # Plot the +- 95% range of models sampled by MCMC
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        show = False

    x = np.logspace(-1, 1, 500)
    y = (x[:, None] > samples[:, 0]) & (x[:, None] < (samples[:, 0] + samples[:, 1]))

    # Plot the posterior distribution for P(in HZ)
    ax.plot(x, 100 * y.mean(axis=1), c=c, lw=lw)

    # Plot the truth value
    if truth:
        if results['survey'].mode == 'imaging':
            a_inner, delta_a = 1.00, 1.007
        else:
            a_inner, delta_a = 1.034, 1.059

        ax.fill_between([a_inner, a_inner + delta_a], -100, 200, color='green', alpha=0.3, lw=0)

    # Limits, labels
    ax.set_xlabel(r'$a_{\mathrm{eff}}$ (AU)', fontsize=labelfontsize)
    ax.set_xscale('log')
    ax.set_xlim([0.1, 10])
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(['0.1', '1', '10'])

    ax.set_ylabel('P (in HZ)', fontsize=labelfontsize)
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 100])
    ax.set_yticklabels(['{:.0f}%'.format(yt) for yt in ax.get_yticks()])

    if show:
        plt.subplots_adjust(bottom=0.3)
        plt.show()
    else:
        return fig, ax


def plot_Example2_constraints(results, fig=None, ax=None, show=True, c='black', lw=1.0, **grid_kwargs):
    # Reduce the grid to the 1 or 2 dimensions in `axes`
    grid = results['grid'].copy()
    seq = [0] * len(grid)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
            if key == 't_half':
                # Determine the truth value of t_half in this set of simulations
                truth = grid[key][seq[i]]
        elif key == 'N':
            # Pick a random simulation unless specified
            seq[i] = np.random.choice(grid['N'])
        elif len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]

    # Extract the MCMC samples for this simulation
    samples = results['chains'][tuple(seq)]
    power = analysis.compute_statistical_power(results, method='dBIC')
    # print(power[tuple(seq)[:-1]])
    # print(results['p'][tuple(seq)]<0.05, results['dBIC'][tuple(seq)]>6)

    # Plot samples values of t_half
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        show = False

    xmin, xmax = results['h'].bounds[1]

    bins = np.logspace(np.log10(xmin), np.log10(xmax), 50)
    # bins = np.linspace(0, 1, 50)
    ax.hist(samples[..., 1], histtype='step', bins=bins, lw=lw, color='black', density=True)

    # Plot the truth value
    ax.axvline(truth, lw=5, c='C0', alpha=0.75)

    # Axes
    # ax.set_xlim([0, 1])
    ax.set_xscale('log')
    ax.set_xlim([xmin, xmax])

    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    if show:
        plt.show()
    else:
        return fig, ax


def image_contour_plot(x, y, z, colormap=True, labels=None, levels=None, fmt=' %.0f ', ticks=4, vmin=None, vmax=None,
                       linecolor='black', log=None, fig=None, ax=None, return_ctr=False, zoom_factor=None,
                       cmap='Greens', cbar=True,
                       plus=False, smooth_sigma=0):
    """ Plots z(x, y) with a colorbar and contours. """

    if fig is None:
        fig, ax = plt.subplots(figsize=(8 + 2 * colormap, 8))
    # plt.locator_params(nbins=ticks)

    if log is None:
        log = (False, False)

    if np.ndim(x) == 1:
        x, y = np.meshgrid(x, y, indexing='ij')

    # Interpolate data to higher resolution
    if zoom_factor:
        if log[0]:
            x = 10 ** zoom(np.log10(x), zoom_factor)
        else:
            x = zoom(x, zoom_factor)
        if log[1]:
            y = 10 ** zoom(np.log10(y), zoom_factor)
        else:
            y = zoom(y, zoom_factor)
        z[np.isnan(z)] = np.nanmax(z) * 2  # Fill in for nan values... best practice is not to over-zoom
        z = zoom(z, zoom_factor)
    else:
        z[np.isnan(z)] = np.nanmax(z) * 2

    # Smooth the data
    if smooth_sigma:
        z = gaussian_filter(z, smooth_sigma)

    # Color plot
    if colormap:
        im = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap, lw=0, rasterized=True, shading='auto',
                           edgecolors='k', linewidths=4)
        cmap = copy.copy(im.cmap)
        cmap.set_bad(color='white', alpha=0.5)
        im.set_cmap(cmap)
        if cbar:
            colorbar = fig.colorbar(im)
        if plus:
            cticks = colorbar.get_ticks()
            cl = ['{:.0f}'.format(tick) for tick in cticks]
            cl[-1] = '>' + cl[-1]
            colorbar.set_ticks(cticks)
            colorbar.set_ticklabels(cl)

    # Contour plot
    if levels is not None:
        ctr = ax.contour(x, y, z, levels=levels, colors=linecolor, linewidths=3)
        if not return_ctr:
            ax.clabel(ctr, ctr.levels, inline=True, fmt=fmt, inline_spacing=20)

    if labels is not None:
        ax.set_xlabel(labels[0], fontsize=labelfontsize)
        ax.set_ylabel(labels[1], fontsize=labelfontsize)
        if colormap and cbar:
            colorbar.set_label(labels[2], rotation=270, labelpad=25, fontsize=labelfontsize)

    # Log-scale?
    if log[0]:
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    if log[1]:
        ax.set_yscale('log')
        ax.get_yaxis().set_major_formatter(ScalarFormatter())

    plt.subplots_adjust(left=0.25, bottom=0.2, right=0.95)

    if return_ctr:
        return fig, ax, ctr
    else:
        return fig, ax


def plot_number_vs_time(results, smooth_sigma=None, exclude=['N_pl', 'N_EEC'], **grid_kwargs):
    # Determine which set of simulations to plot
    grid = results['grid']
    seq = [0] * len(grid)
    for i, (key, val) in enumerate(grid.items()):
        if key in ['t_total', 'N']:
            seq[i] = slice(None)
            continue
        elif key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        if len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))
    seq = tuple(seq)

    # Planet type keys and line colors
    pl_colors = {'N_pl': 'k', 'N_EEC': 'C0', 'N_hot': 'red', 'N_warm': 'green', 'N_cold': 'blue'}
    for key in exclude:
        del pl_colors[key]

    # Retrieve average number of planets vs time and smooth the y-values
    x = grid['t_total']
    ys = {key: results[key][seq].mean(axis=-1) for key in pl_colors}
    if smooth_sigma:
        for key, val in ys.items():
            ys[key] = gaussian_filter(val, smooth_sigma)

    # Plot each planet type
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, y in ys.items():
        ax.plot(x, y, c=pl_colors[key], lw=5)
    plt.show()


def plot_precision_grid(results, param='a_inner', axes=('f_water_habitable', 'f_water_nonhabitable'), labels=None,
                        cbar=True, method='dlnZ', threshold=None, smooth_sigma=None, show=True, half=False,
                        levels=3, log=None, fmt='%.2f AU', **grid_kwargs):
    # Reduce the grid to the two dimensions in `axes`
    grid = results['grid'].copy()
    seq = [0] * (len(grid) - 1)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in axes:
            seq[i] = slice(None)
            continue
        elif key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        elif key != 'N' and len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]
    seq = tuple(seq)

    # Extract the confidence intervals and determine which should be plotted
    idx_param = list(results['h'].params).index(param)
    CIs = confidence = results['CIs'][..., idx_param]
    if isinstance(levels, int):
        dpct = 100 / levels
        levels = np.percentile(CIs, np.arange(dpct / 2, 100, dpct))

    # Determine the appropriate lower and upper bound for each plotted confidence interval
    lowers, uppers = np.zeros(len(levels)), np.zeros(len(levels))
    for i, level in enumerate(levels):
        dcmin = (level - levels[i - 1]) / 2. if i > 0 else (levels[i + 1] - level) / 2.
        dcmax = (levels[i + 1] - level) / 2. if i < len(levels) - 1 else (level - levels[i - 1]) / 2.
        mask = (CIs < (level + dcmax)) & (CIs > (level - dcmin))

        lowers[i] = (results['medians'] - results['LCIs'])[..., idx_param][mask].mean()
        uppers[i] = (results['medians'] + results['UCIs'])[..., idx_param][mask].mean()

    # Plot the confidence interval contours
    x, y = np.meshgrid(grid[axes[0]], grid[axes[1]], indexing='ij')
    z = CIs.mean(axis=-1)

    # Smooth the data if valid
    if smooth_sigma:
        z = gaussian_filter(z, smooth_sigma)

    fig, ax = plt.subplots(figsize=(8, 8))
    ctr = ax.contour(x, y, z, levels=levels, colors='black', linewidths=3)
    labels = {levels[i]: '\n\n{:.1f} < {:s} < {:.1f} AU'.format(lowers[i], param, uppers[i]) for i in
              range(len(levels))}
    print(labels)
    ax.clabel(ctr, levels, inline=False, fmt=labels, fontsize=16)
    # ax[0].clabel(ctr,ctr.levels,inline=True,fmt=fmt)

    if log is not None:
        ax.set_xscale('log' if log[0] else 'linear')
        ax.set_yscale('log' if log[1] else 'linear')

    plt.show()


def plot_precision(results, params=('a_outer', 'a_inner'), axes=('f_water_habitable', 'f_water_nonhabitable'),
                   labels=None, cbar=True, method='dlnZ', threshold=None, smooth_sigma=None, show=True, half=False,
                   levels=([1.0, 3.0], [0.2, 0.6]), log=None, fmt=' %.2f AU ', **grid_kwargs):
    # Validate length of `axes`, `labels`, `params`
    if isinstance(axes, STR_TYPES):
        axes = (axes,)
    if isinstance(labels, STR_TYPES):
        labels = (labels,)
    if isinstance(params, STR_TYPES):
        params = (params,)
    if labels is None:
        labels = axes

    ndim = len(axes)
    if ndim > 2:
        raise ValueError("'axes' must be a string or 1- or 2- element list")

    # Reduce the grid to the 1 or 2 dimensions in `axes`
    grid = results['grid'].copy()
    seq = [0] * (len(grid) - 1)
    for i, (key, val) in enumerate(results['grid'].items()):
        if key in axes:
            seq[i] = slice(None)
            continue
        elif key in grid_kwargs:
            seq[i] = np.argmin(np.abs(val - grid_kwargs[key]))
        elif key != 'N' and len(val) > 1:
            warn("which value of '{:s}' should be plotted? assuming {:s} = {:.1f}".format(key, key, val[0]))

        # Removes 'N' and other reduced grid dimensions
        del grid[key]
    seq = tuple(seq)

    # Extract the x (, y) and z coordinates and smooth the data
    x, y = grid[axes[0]], None if ndim == 1 else grid[axes[1]]
    fig, ax = None, None
    for i, param in enumerate(params):
        idx_param = list(results['h'].params).index(param)
        z = results['CIs'][(*seq, slice(None), idx_param)].mean(axis=-1)
        lvl = levels[i]
        if smooth_sigma:
            z = gaussian_filter(z, smooth_sigma)

        if half:
            lvl, z = np.array(lvl) / 2, z / 2

        fig, ax = image_contour_plot(x, y, z, labels=(*labels, 'Precision (AU)'), fmt=fmt, colormap=False,
                                     levels=lvl, log=log, fig=fig, ax=ax, linecolor='C{:d}'.format(i))

    if show:
        plt.show()
    else:
        return fig, ax


def plot_simulation_result(results, log=True, method='dlnZ', **grid_kwargs):
    # If a grid of results is given, determine which simulation to plot
    if 'grid' in results:
        grid = results['grid']
        seq = [0] * len(grid)
        for i, (key, val) in enumerate(grid.items()):
            if key in grid_kwargs:
                seq[i] = np.argmin(np.abs(grid_kwargs[key] - val))
            else:
                seq[i] = np.random.choice(np.arange(len(val)))
        seq = tuple(seq)

        # Retrieve the results of this simulation
        x, dx, y, dy = *util.bin_centers(results['bins'][seq]), results['values'][seq], results['errors'][seq]
        chains, medians, stds = results['chains'][seq], results['medians'][seq], results['stds'][seq]
        conf = results[method][seq]
        h, params = results['h'], results['h'].params
        N_pl = {key: results[key][seq] for key in ['N_pl', 'N_hot', 'N_warm', 'N_cold', 'N_EEC']}

    else:
        x, dx, y, dy = *util.bin_centers(results['bins']), results['values'], results['errors']
        chains, medians, stds = results['chains'], results['medians'], results['stds']
        conf = results[method]
        h, params = results['h'], results['h'].params
        N_pl = {key: results[key] for key in ['N_pl', 'N_hot', 'N_warm', 'N_cold', 'N_EEC']}

    # Make the figure and plots
    nr, nc = 2, 1 if chains is None else len(params)
    fig = plt.figure(figsize=(16, 8))
    if 'grid' in results:
        keys, vals = list(grid.keys()), [grid[key][i] for key, i in zip(grid.keys(), seq)]
    else:
        keys, vals = zip(*results['fixed'].items())

    title = ', '.join(['{0} = {1}'.format(key, np.round(val, 2) if isinstance(val, float) else val) \
                       for key, val in zip(keys, vals)])
    fig.suptitle('Results of the following simulation: {:s} = {:.1f}\n'.format(method, conf) + title)

    # Top plot: binned data values with the best-fit model and (optional) the range of posterior models
    ax_top = plt.subplot2grid((nr, nc), (0, 0), rowspan=nr if chains is None else nr // 2, colspan=nc, fig=fig)
    ax_top.errorbar(x, y, xerr=dx, yerr=dy, marker='s', linestyle='None')
    ax_top.axhline(np.mean(y), linestyle='dashed', lw=2)

    xm = np.linspace(x.min(), x.max(), 100)
    ax_top.plot(xm, h(medians, xm), lw=5, c='black')
    if chains is not None:
        ym = [h(chains[i], xm) for i in np.random.choice(np.arange(chains.shape[0]), size=300)]
        mu, std = np.mean(ym, axis=0), np.std(ym, axis=0)
        ax_top.fill_between(xm, mu - std, mu + std, color='grey', alpha=0.5)

    # Annotate with the number of planets in the sample
    for i, key in enumerate(['N_pl', 'N_hot', 'N_warm', 'N_cold', 'N_EEC']):
        N, xy = N_pl[key], (0.02, 0.9 - 0.1 * i)
        ax_top.annotate('{:s} = {:d}'.format(key, int(N)), xy=xy, xycoords='axes fraction',
                        ha='left', va='center', fontsize=16)

    # Bottom plot: plot the 1D posterior distribution for each parameter
    if chains is not None:
        ax_bottom = []
        for i in range(len(params)):
            ax = plt.subplot2grid((nr, nc), (nr // 2, i), rowspan=nr // 2, colspan=1, fig=fig)
            ax_bottom.append(ax)
        fig, ax_bottom = plot_posterior(chains, params=params, log=h.log, fig=fig, axes=ax_bottom)

    if log:
        ax_top.set_xscale('log')
    plt.show()


def plot_clear_cloudy_spectra(x, y_clr, y_cld, f_clouds=0.75, lw=3, bands=[], c=[], alpha=0.3, legend=True, xlim=None,
                              ymax=None, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Clear spectrum
    ax.plot(x, y_clr, lw=lw, c='gray', label='without clouds')

    # Cloudy spectrum
    y_mix = y_cld * f_clouds + y_clr * (1 - f_clouds)
    ax.plot(x, y_mix, lw=lw, c='black', label='with clouds')

    if ymax is None:
        ymax = ax.get_ylim()[1]

    # Bandpasses
    if len(bands) > 0:
        for band, col in zip(bands, c):
            ax.fill_between(band, 0, 1000, zorder=-1, alpha=alpha, color=col, lw=0)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlabel('Wavelength ($\mu$m)', fontsize=labelfontsize)
    # ax.set_ylabel(r'$\xi$', fontsize=labelfontsize)

    if legend:
        ax.legend(frameon=False, loc='lower left')

    ax.set_ylim([0, ymax])

    ax.set_xticks([0.4, 0.6, 1.0, 1.5, 2.0])

    if xlim is not None:
        ax.set_xlim(xlim)
    plt.subplots_adjust(bottom=0.17, left=0.15)

    return fig, ax
"""
This method summarizes the planet yield of a survey by plotting the S-R distribution
of detected planets and histograms of key parameters. It also includes a text block
summarizing the number of planets, stars, and EECs in the sample, as well as any
assumptions provided.

The main scatter plot optionally colors the points by the logarithm of the exposure
time. The small summary panels show distributions of distance, stellar effective
temperature, stellar luminosity, stellar mass, period, transit duration, number of
observed transits, and exposure time. Habitable zone boundaries are marked with
dashed vertical lines on the S-R plot.
"""



def plot_yield_summary(
    d,
    assumptions=None,
    show_colorbar=True,
    S_lim=(6.0, 0.20),
    R_lim=(0.0, 15.0),
    title="Planet Yield Summary — Instellation vs. Radius",
    cmap="viridis_r",
    hist_color="steelblue",
    hz_color="0.35",
    hz_style="--",
    marker_size=30,
    label_fontsize=16,
    tick_fontsize=13,
    HZ_INNER = 1.7763,
    HZ_OUTER = 0.3207
):
    """
    Parameters
    ----------
    d : bioverse Table
        Table object containing the planet yield data.
    assumptions : dict, optional
        A dictionary of assumptions to include in the text summary.
    show_colorbar : bool, optional
        Whether to display the colorbar for log10(t_exp) on the S-R scatter plot.
        Default is True.
    S_lim : tuple, optional
        (left, right) x-axis limits for the instellation axis. Since higher S means
        closer to the star, the axis is reversed (left > right by default).
        Default is (6.0, 0.20).
    R_lim : tuple, optional
        (bottom, top) y-axis limits for the radius axis.
        Default is (0.0, 15.0).
    title : str, optional
        Title displayed above the main S-R scatter panel.
        Default is "Planet Yield Summary — Instellation vs. Radius".
    cmap : str, optional
        Colormap used to color scatter points by log10(t_exp).
        Default is "viridis_r".
    hist_color : str, optional
        Fill color for the histogram bars in the small summary panels.
        Default is "steelblue".
    hz_color : str, optional
        Color for the habitable zone boundary lines.
        Default is "0.35" (dark gray).
    hz_style : str, optional
        Linestyle for the habitable zone boundary lines.
        Default is "--" (dashed).
    marker_size : int or float, optional
        Size of scatter plot markers in the main S-R panel.
        Default is 30.
    label_fontsize : int, optional
        Font size for axis labels. Tick labels are scaled relative to this.
        Default is 16.
    tick_fontsize : int, optional
        Font size for axis tick labels.
        Default is 13.
    HZ_INNER : float, optional
        Inner habitable zone flux boundary
    HZ_OUTER: float, optional
        Outer habitable zone flux boundary

    Returns
    -------
    fig : matplotlib Figure
        The figure containing the yield summary plots.
    """

    # --- Input validation ------------------------------------------------
    if d is None:
        raise ValueError("No data provided for plotting.")
    if not isinstance(d, Table):
        raise ValueError("Input data must be a bioverse Table.")

    required_keys = [
        "S", "R", "P", "M_st", "t_exp", "N_obs", "T_dur", "EEC",
        "d", "T_eff_st", "L_st",
    ]
    for key in required_keys:
        if key not in d:
            raise ValueError(f"Missing required column '{key}' in input data.")

    # --- Layout constants ------------------------------------------------
    TITLE_FS = label_fontsize + 2

    # --- Figure & GridSpec -----------------------------------------------
    fig = plt.figure(figsize=(20, 13), constrained_layout=True)
    gs  = GridSpec(nrows=3, ncols=4, figure=fig,
                   height_ratios=[2.5, 1.3, 1.3])

    ax_main = fig.add_subplot(gs[0, :3])   # main S-R panel
    ax_text = fig.add_subplot(gs[0, 3])    # text / summary panel
    ax_text.axis("off")

    columns = [
        "d",     "T_eff_st", "L_st",  "M_st",
        "P",     "T_dur",    "N_obs",  "t_exp",
    ]
    xlabels = [
        "Distance (pc)",
        r"Stellar $\mathrm{T_{eff}}$ (K)",
        r"Stellar Luminosity ($L_\odot$)",
        r"Stellar Mass ($M_\odot$)",
        "Period (days)",
        "Transit Duration (days)",
        "Number of Observed Transits",
        "Exposure Time (days)",
    ]
    log_cols = {"P", "L_st"}   # these axes use log10 transformation

    small_axes = []
    for idx in range(8):
        row = 1 + idx // 4
        col = idx  % 4
        small_axes.append(fig.add_subplot(gs[row, col]))

    # --- Main S-R scatter ------------------------------------------------
    sc = ax_main.scatter(
        d["S"], d["R"],
        c=np.log10(d["t_exp"]),
        s=marker_size,
        cmap=cmap,
        alpha=0.85,
    )

    ax_main.set_xscale("log")
    ax_main.set_xlim(*S_lim)
    ax_main.set_ylim(*R_lim)
    ax_main.set_xlabel(r"Instellation ($S_\oplus$)", fontsize=label_fontsize)
    ax_main.set_ylabel(r"Radius ($R_\oplus$)",        fontsize=label_fontsize)
    ax_main.tick_params(labelsize=tick_fontsize)
    ax_main.set_title(title, fontsize=TITLE_FS, fontweight="semibold", pad=10)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax_main)
        cbar.set_label(
            r"$\log_{10}(t_\mathrm{exp}\ [\mathrm{days}])$",
            fontsize=label_fontsize,
        )
        cbar.ax.tick_params(labelsize=tick_fontsize)

    # Habitable zone boundaries
    if (HZ_OUTER is not None) and (HZ_OUTER is not None):
        hz_kw = dict(color=hz_color, linestyle=hz_style, linewidth=1.4, alpha=0.75)
        ax_main.axvline(HZ_INNER, **hz_kw, label="Habitable zone")
        ax_main.axvline(HZ_OUTER, **hz_kw)
        ax_main.legend(fontsize=tick_fontsize)

    # --- Eight histogram panels ------------------------------------------
    for ax, col, xlabel in zip(small_axes, columns, xlabels):
        values = d[col]
        if col in log_cols:
            values = np.log10(values)
            xlabel = r"$\log_{10}$(" + xlabel + ")"
        ax.hist(values, bins=25, color=hist_color, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel,  fontsize=label_fontsize)
        ax.set_ylabel("Count", fontsize=label_fontsize - 2)
        ax.tick_params(labelsize=tick_fontsize)

    lines = [
        f"N planets : {len(d)}",
        f"N stars   : {len(d.get_stars())}",
        f"N EEC     : {int(d['EEC'].sum())}",
        "",
        "Assumptions:",
    ]
    lines += (
        [f"  {k}: {v}" for k, v in assumptions.items()]
        if assumptions else ["  (none provided)"]
    )

    ax_text.text(
        0.05, 0.97,
        "\n".join(lines),
        va="top", ha="left",
        fontsize=label_fontsize - 2,
        family="monospace",
        transform=ax_text.transAxes,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ax_text.text(
        0.05, 0.03,
        f"Generated:\n{timestamp}",
        va="bottom", ha="left",
        fontsize=11,
        color="0.4",
        family="monospace",
        transform=ax_text.transAxes,
    )

    return fig


def plot_histogram(d, column, log_x=False, bins=30, xlabel=None, ax=None, fig=None, **kwargs):
    """ Generic histogram of a column in a Table.

    Parameters
    ----------
    d : Table
        Table of simulated planets or stars.
    column : str
        Name of the column to plot.
    log_x : bool, optional
        If True, plot the log10 of the column values on the x axis. Default is False.
    bins : int or array, optional
        Number of bins or bin edges. Default is 30.
    xlabel : str, optional
        Custom label for the x axis. If not given, defaults to the column name
        (or its log10 form if log_x is True).
    ax : Axes, optional
        Matplotlib Axes to plot on. If not given, a new figure is created.
    fig : Figure, optional
        Matplotlib Figure. If not given, a new figure is created.
    **kwargs
        Additional keyword arguments passed to ax.hist.
    """
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        show = False

    values = d[column]
    mask = ~np.isnan(values.astype(float))
    values = values[mask]

    if xlabel is None:
        xlabel = column
        if log_x:
            xlabel = r'$\log_{10}$(' + column + ')'
    if log_x:
        values = np.log10(values)

    ax.hist(values, bins=bins, **kwargs)
    ax.set_xlabel(xlabel, fontsize=labelfontsize)
    ax.set_ylabel('Count', fontsize=labelfontsize)

    plt.subplots_adjust(bottom=0.2, left=0.15)
    if show:
        plt.show()
    return fig, ax


def plot_HR_diagram(d, xlim=None, ylim=None, ax=None, fig=None, **kwargs):
    """ Hertzsprung-Russell diagram: stellar effective temperature vs. luminosity.

    Both axes are logarithmic. The x axis is displayed in reverse order (high T on
    the left) following astronomical convention.

    Parameters
    ----------
    d : Table
        Table of simulated planets or stars. Stellar rows are extracted via get_stars().
    xlim : tuple, optional
        limits for the effective temperature axis.
    ylim : tuple, optional
        Limits for the luminosity axis.
    ax : Axes, optional
        Matplotlib Axes to plot on. If not given, a new figure is created.
    fig : Figure, optional
        Matplotlib Figure. If not given, a new figure is created.
    **kwargs
        Additional keyword arguments passed to ax.scatter.
    """
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        show = False

    st = d.get_stars()
    T = st['T_eff_st']
    L = st['L_st']

    ax.scatter(T, L, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')

    if xlim is None:
        xlim = (np.nanmax(T) * 1.05, np.nanmin(T) * 0.95)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis='x', which='minor', labelsize=labelfontsize - 4)

    ax.set_xlabel(r'Effective Temperature, $T_\mathrm{eff}$ (K)', fontsize=labelfontsize)
    ax.set_ylabel(r'Luminosity ($L_\odot$)', fontsize=labelfontsize)
    ax.set_title('Hertzsprung-Russell Diagram', fontsize=labelfontsize)

    plt.subplots_adjust(bottom=0.15, left=0.15)
    if show:
        plt.show()
    return fig, ax


def plot_distance_luminosity(d, colorbar=False, color_column='t_exp', cmap='viridis',
                              ax=None, fig=None, **kwargs):
    """ Scatter plot of stellar distance vs. luminosity.

    The luminosity axis is logarithmic. Points can optionally be colored by the
    log10 of a chosen column (e.g. exposure time, t_exp).

    Parameters
    ----------
    d : Table
        Table of simulated planets or stars.
    colorbar : bool, optional
        If True, color points by log10(color_column) and display a colorbar.
        Default is False.
    color_column : str, optional
        Name of the column used for the colorbar when colorbar is True.
        Default is 't_exp'.
    cmap : str, optional
        Colormap used for the colorbar. Default is 'viridis'.
    ax : Axes, optional
        Matplotlib Axes to plot on. If not given, a new figure is created.
    fig : Figure, optional
        Matplotlib Figure. If not given, a new figure is created.
    **kwargs
        Additional keyword arguments passed to ax.scatter.
    """
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        show = False
        if fig is None:
            fig = ax.figure

    dist = d['d']
    L = d['L_st']
    kwargs.setdefault('s', 2) 

    if colorbar:
        if color_column not in d:
            raise KeyError(
                f"'{color_column}' not found in table. To compute exposure times, use "
                "method='scaling_relation' in compute_yield() after calling "
                "survey.set_reference_observation(). Alternatively, pass colorbar=False "
                "or a different color_column."
            )
        sc = ax.scatter(dist, L, c=np.log10(d[color_column]), cmap=cmap, **kwargs)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r'$\log_{10}$(' + color_column + ')', fontsize=labelfontsize)
    else:
        ax.scatter(dist, L, **kwargs)

    ax.set_yscale('log')
    ax.set_xlabel('Distance (pc)', fontsize=labelfontsize)
    ax.set_ylabel(r'Luminosity ($L_\odot$)', fontsize=labelfontsize)

    plt.subplots_adjust(bottom=0.15, left=0.15)
    if show:
        plt.show()
    return fig, ax