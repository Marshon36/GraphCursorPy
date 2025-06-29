import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from obspy import read
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Reset matplotlib parameters
fontsize = 12
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['axes.unicode_minus']=False


class DistAz:
    """c
    c Subroutine to calculate the Great Circle Arc distance
    c    between two sets of geographic coordinates
    c
    c Equations take from Bullen, pages 154, 155
    c
    c T. Owens, September 19, 1991
    c           Sept. 25 -- fixed az and baz calculations
    c
    P. Crotwell, Setember 27, 1995
    Converted to c to fix annoying problem of fortran giving wrong
    answers if the input doesn't contain a decimal point.

    H. P. Crotwell, September 18, 1997
    Java version for direct use in java programs.
    *
    * C. Groves, May 4, 2004
    * Added enough convenience constructors to choke a horse and made public double
    * values use accessors so we can use this class as an immutable

    H.P. Crotwell, May 31, 2006
    Port to python, thus adding to the great list of languages to which
    distaz has been ported from the origin fortran: C, Tcl, Java and now python
    and I vaguely remember a perl port. Long live distaz! 
    """

    def __init__(self,  lat1,  lon1,  lat2,  lon2):
        """
        c lat1 => Latitude of first point (+N, -S) in degrees
        c lon1 => Longitude of first point (+E, -W) in degrees
        c lat2 => Latitude of second point
        c lon2 => Longitude of second point
        c
        c getDelta() => Great Circle Arc distance in degrees
        c getAz()    => Azimuth from pt. 1 to pt. 2 in degrees
        c getBaz()   => Back Azimuth from pt. 2 to pt. 1 in degrees
        """
        self.stalat = lat1
        self.stalon = lon1
        self.evtlat = lat2
        self.evtlon = lon2
        if (lat1 == lat2) and (lon1 == lon2):
            self.delta = 0.0
            self.az = 0.0
            self.baz = 0.0
            return

        rad = 2.*math.pi/360.0
        """
	c
	c scolat and ecolat are the geocentric colatitudes
	c as defined by Richter (pg. 318)
	c
	c Earth Flattening of 1/298.257 take from Bott (pg. 3)
	c
        """
        sph = 1.0/298.257

        scolat = math.pi/2.0 - math.atan((1.-sph)*(1.-sph)*math.tan(lat1*rad))
        ecolat = math.pi/2.0 - math.atan((1.-sph)*(1.-sph)*math.tan(lat2*rad))
        slon = lon1*rad
        elon = lon2*rad
        """
	c
	c  a - e are as defined by Bullen (pg. 154, Sec 10.2)
	c     These are defined for the pt. 1
	c
        """
        a = math.sin(scolat)*math.cos(slon)
        b = math.sin(scolat)*math.sin(slon)
        c = math.cos(scolat)
        d = math.sin(slon)
        e = -math.cos(slon)
        g = -c*e
        h = c*d
        k = -math.sin(scolat)
        """
	c
	c  aa - ee are the same as a - e, except for pt. 2
	c
        """
        aa = math.sin(ecolat)*math.cos(elon)
        bb = math.sin(ecolat)*math.sin(elon)
        cc = math.cos(ecolat)
        dd = math.sin(elon)
        ee = -math.cos(elon)
        gg = -cc*ee
        hh = cc*dd
        kk = -math.sin(ecolat)
        """
	c
	c  Bullen, Sec 10.2, eqn. 4
	c
        """
        delrad = math.acos(a*aa + b*bb + c*cc)
        self.delta = delrad/rad
        """
	c
	c  Bullen, Sec 10.2, eqn 7 / eqn 8
	c
	c    pt. 1 is unprimed, so this is technically the baz
	c
	c  Calculate baz this way to avoid quadrant problems
	c
        """
        rhs1 = (aa-d)*(aa-d)+(bb-e)*(bb-e)+cc*cc - 2.
        rhs2 = (aa-g)*(aa-g)+(bb-h)*(bb-h)+(cc-k)*(cc-k) - 2.
        dbaz = math.atan2(rhs1, rhs2)
        if (dbaz < 0.0):
            dbaz = dbaz+2*math.pi

        self.baz = dbaz/rad
        """
	c
	c  Bullen, Sec 10.2, eqn 7 / eqn 8
	c
	c    pt. 2 is unprimed, so this is technically the az
	c
	"""
        rhs1 = (a-dd)*(a-dd)+(b-ee)*(b-ee)+c*c - 2.
        rhs2 = (a-gg)*(a-gg)+(b-hh)*(b-hh)+(c-kk)*(c-kk) - 2.
        daz = math.atan2(rhs1, rhs2)
        if daz < 0.0:
            daz = daz+2*math.pi

        self.az = daz/rad
        """
	c
	c   Make sure 0.0 is always 0.0, not 360.
	c
	"""
        if (abs(self.baz-360.) < .00001):
            self.baz = 0.0
        if (abs(self.az-360.) < .00001):
            self.az = 0.0

    def getDelta(self):
        return self.delta

    def getAz(self):
        return self.az

    def getBaz(self):
        return self.baz

    def degreesToKilometers(self, degrees):
        return degrees * 111.19

    def kilometersToDegrees(self, kilometers):
        return kilometers / 111.19

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, show=False, ax=None, title=True):

    """
    # Purpose:	 Detect peaks in data based on their amplitude and other features.
    # Author:	Jun Zhu, modified from Marcos Duarte
    # Date:		FEB 19 2022
    # Email:	Jun__Zhu@outlook.com

    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
        (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)

    Version history
    ---------------
    '1.0.6':
        Fix issue of when specifying ax object only the first plot was shown
        Add parameter to choose if a title is shown and input a title
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind, x[ind]


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            no_ax = True
        else:
            no_ax = False

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"% \
                        (mode, str(mph), mpd, str(threshold), edge)
            ax.set_title(title)
        # plt.grid()
        if no_ax:
            plt.show()

def normalize(x):
    x_norm = (x - np.mean(x)) / np.std(x)
    return x_norm

def plot_stacked_map(source_predicts, receiver_predicts, source_sca_points, receiver_sca_points, 
                     array, loc, sca_lat_max, sca_lon_max, config, save_fig = False, 
                     output_dir=None, fk_dict = None):
    """
    Create a stacked comparison plot of source and receiver scattering amplitudes.

    Args:
        source_predicts (np.ndarray): Source predictions data (n_points, n_features).
        receiver_predicts (np.ndarray): Receiver predictions data (n_points, n_features).
        source_sca_points (list/np.ndarray): Source scatter points (latitude, longitude).
        receiver_sca_points (list/np.ndarray): Receiver scatter points (latitude, longitude).
        array (str): Name of array.
        loc (str): 'source' or 'receiver' to indicate scatter location.
        sca_lat_max (float): Latitude of maximum scatter point.
        sca_lon_max (float): Longitude of maximum scatter point.
        config (module): A Python module containing configuration settings. 
        fk_dict :
        save_fig (bool): Whether to save the figure (default: False).
        output_dir (str, optional): Path to save figure.
    """
    new_cmap = plt.get_cmap('Spectral_r')
    new_cmap.set_bad(color='none')
    
    # Prepare lat/lon arrays (reshape)
    grid_len = config.grid_len
    so_lats = np.array([pt[0] for pt in source_sca_points]).reshape((grid_len, grid_len))
    so_lons = np.array([pt[1] for pt in source_sca_points]).reshape((grid_len, grid_len))
    re_lats = np.array([pt[0] for pt in receiver_sca_points]).reshape((grid_len, grid_len))
    re_lons = np.array([pt[1] for pt in receiver_sca_points]).reshape((grid_len, grid_len))

    # Scattering values
    so_data = source_predicts.reshape((grid_len, grid_len))
    re_data = receiver_predicts.reshape((grid_len, grid_len))
    vmin = min(np.nanmin(so_data), np.nanmin(re_data))
    vmax = max(np.nanmax(so_data), np.nanmax(re_data))

    # Setup figure and Cartopy axes
    fontsize = 12
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    axs = [ax0, ax1]

    for ax in axs:
        ax.add_feature(cfeature.COASTLINE)

        # Set ticks and labels for x and y axis    
        ax.set_xticks(np.arange(-180, 181, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 10), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xlabel('Longitude', fontsize=fontsize)
        ax.set_ylabel('Latitude', fontsize=fontsize)
        
    # Plot source side
    pcm1 = axs[0].pcolormesh(so_lons, so_lats, so_data,
                             cmap=new_cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree()
                             )
    axs[0].set_title(f'Source-side', fontsize=fontsize)

    # Plot receiver side
    pcm2 = axs[1].pcolormesh(re_lons, re_lats, re_data,
                             cmap=new_cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree()
                             )
    axs[1].set_title(f'Receiver-side', fontsize=fontsize)
    axs[1].set_extent([np.min(re_lons), np.max(re_lons), np.min(re_lats), np.max(re_lats)], crs=ccrs.PlateCarree())

    if fk_dict != None:
        if len(fk_dict['source_lons']) > 0:
            axs[0].scatter(fk_dict['source_lons'], fk_dict['source_lats'],
                           marker = "x", s = 100, label = 'Scatter Location (FK)', color = 'darkgrey',
                           zorder = 5)
        if len(fk_dict['receiver_lons']) > 0:
            axs[1].scatter(fk_dict['receiver_lons'], fk_dict['receiver_lats'],
                           marker = "x", s = 100, label = 'Scatter Location (FK)', color = 'darkgrey',
                           zorder = 5)

    # Plot the location of scatter
    if loc == 'source':
        axs[0].scatter(sca_lon_max, sca_lat_max, s = 200, c = config.c1, 
                      marker = "o", label = 'Scatterer Location', edgecolor = 'black',
                      zorder = 6)
        axs[0].legend(fontsize = fontsize)
    else:
        axs[1].scatter(sca_lon_max, sca_lat_max, s = 200, c = config.c1, 
                      marker = "o", label = 'Scatterer Location', edgecolor = 'black',
                      zorder = 6)
        axs[1].legend(fontsize = fontsize)

    # Add a shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.124, 0.13, 0.779, 0.03])  # [left, bottom, width, height]
    plt.colorbar(pcm2, cax=cbar_ax, orientation='horizontal', label='Scattering Likelihood')
    fig.subplots_adjust(bottom=0.25)

    # Save or display
    if save_fig:
        save_path = os.path.join(output_dir, f"{array.replace(':', '-')}_migration_map.pdf")
        plt.savefig(save_path, dpi = 500, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    pass

def plot_waveforms_predictions(df_array, pred_path, config, save_fig=False, output_dir=None, times = None):
    """
    Plot seismic waveforms and their corresponding GraphCursor model predictions.

    This function visualizes seismic waveform traces alongside predicted probability curves
    from a GraphCursor model. The waveforms are normalized and plotted by epicentral distance.

    Args:
        df_array (pd.DataFrame): DataFrame containing waveform metadata with the following columns:
            - 'fname': Full path to the waveform file (SAC format).
            - 'dist': Epicentral distance in degrees.
            - 'array': (Optional) Array/station name used in title and output filename.
        pred_path (str): Directory path containing prediction .npy files.
        config (module): Configuration module containing parameters such as:
            - waveform_time: Pre-PKPdf time window (seconds)
            - waveform_len: Total waveform samples
            - sr: Sampling rate (Hz)
            - pre_window: Time (seconds) from trace start to PKPdf
            - c1, c4: Colors for fill and line plotting
            - event_split: Boolean for whether predictions are stored in per-event folders
        save_fig (bool): Whether to save the figure as a file. Default is False (show plot).
        output_dir (str): Path to output directory where figures are saved (required if save_fig=True).
        times (list or ndarray, optional): If provided, highlights predicted times (e.g., PKPpre) with red dots.

    Returns:
        matplotlib.figure.Figure: The generated figure object (not returned if `save_fig=True`).
    """
    # Parameter validation
    if save_fig and not output_dir:
        raise ValueError("output_dir must be specified when save_fig=True")
    if 'fname' not in df_array.columns or 'dist' not in df_array.columns:
        raise ValueError("DataFrame must contain 'fname' and 'dist' columns")
    
    # Setup figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.67, 5), sharey=True)
    ax2.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.1)
    
    # Get common parameters
    dists = df_array['dist'].values
    posttime = 10
    time = np.linspace(-config.waveform_time, 0, config.waveform_len)
    time_df = np.linspace(-config.waveform_time, posttime, 
                         int((config.waveform_time + posttime) * config.sr))
    
    # Load and normalize waveforms
    waveforms = []
    for _, row in df_array.iterrows():
        tr = read(row['fname'])[-1]  # Get last trace if multiple components
        pkpdf = tr.stats.starttime + config.pre_window
        tr.trim(pkpdf - config.waveform_time, pkpdf + posttime)
        waveform = tr.data[:int((config.waveform_time + posttime) * config.sr)]
        waveforms.append(waveform)
    
    global_max = max(np.max(np.abs(wave[:int(config.waveform_time * config.sr)])) 
                    for wave in waveforms)
    waveforms = np.array(waveforms) / global_max
    
    # Plot waveforms and predictions
    for j, (_, row) in enumerate(df_array.iterrows()):
        # Plot waveform
        ax1.plot(time_df, -waveforms[j] + row['dist'], 
               color='grey', linewidth=0.7, zorder=1)
        if times is not None:
            ax1.scatter(times[j], row['dist'], 
               color='red', zorder=2)
        
        # Plot prediction
        if config.event_split:
            directory = os.path.dirname(row['fname'])
            event = directory.split('/')[-1]
            pred_file = f"{pred_path}{event}/{os.path.basename(row['fname']).split('.SAC')[0]}.npy"
        else:
            pred_file = f"{pred_path}{os.path.basename(row['fname']).split('.SAC')[0]}.npy"
        gnn_predict = np.load(pred_file)
        ax2.plot(time, -gnn_predict + row['dist'], 
               color=config.c4, linewidth=1, zorder=1)
        if times is not None:
            ax2.scatter(times[j], row['dist'], 
               color='red', zorder=2)
        
        # Highlight significant predictions
        ax2.fill_between(time, -gnn_predict + row['dist'], row['dist'],
                    color=config.c1, alpha=0.5)
    
    # Configure axes
    ax1.set_ylim(max(dists)+0.5, min(dists)-0.5)  # Inverted distance axis
    ax1.set_xlim(-config.waveform_time, posttime)
    ax1.set_xlabel("Relative Time (s)")
    ax1.set_ylabel("Epicentral Distance (deg)")
    ax1.set_title(df_array.iloc[0]['array'].split('_')[0].split('.')[0], fontsize = fontsize)
    ax1.minorticks_on()
    
    ax2.set_xlim(-config.waveform_time, 0)
    ax2.set_xlabel("Relative Time (s)")
    ax2.set_title('GraphCursor Predictions', fontsize = fontsize)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    
    # Save or display
    if save_fig:
        array_name = df_array.iloc[0]['array']
        save_path = os.path.join(output_dir, f"{array_name.replace(':', '-')}.pdf")
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    pass

def plot_waveforms(df_array, config, save_fig=False, output_dir=None, title = None):
    """
    Plot raw normalized seismic waveforms aligned by PKPdf arrival time.

    This function visualizes a set of seismic waveforms ordered by epicentral distance,
    with each trace aligned relative to the predicted PKPdf arrival (based on pre_window).
    Amplitudes are normalized by the maximum pre-PKPdf value across all traces.

    Args:
        df_array (pd.DataFrame): DataFrame containing waveform metadata. Required columns:
            - 'fname': Full path to SAC waveform file.
            - 'dist': Epicentral distance in degrees.
            - 'array': (Optional) array or network identifier for naming and figure title.
        config (module): Configuration module with fields:
            - waveform_time (float): Time before PKPdf in seconds.
            - sr (float): Sampling rate in Hz.
            - pre_window (float): Time (in seconds) from trace start to PKPdf.
        save_fig (bool, optional): If True, saves the figure to output_dir instead of showing. Default is False.
        output_dir (str, optional): Path to save output figure. Required if save_fig=True.
        title (str, optional): Optional custom title for the plot. If not provided, uses array name.

    Returns:
        matplotlib.figure.Figure: The generated figure (returned only if save_fig=False).
    """
    # Parameter validation
    if save_fig and not output_dir:
        raise ValueError("output_dir must be specified when save_fig=True")
    if 'fname' not in df_array.columns or 'dist' not in df_array.columns:
        raise ValueError("DataFrame must contain 'fname' and 'dist' columns")
    
    # Setup figure and axes
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 6.67), sharey=True)
    fig.subplots_adjust(wspace=0.1)
    
    # Get common parameters
    dists = df_array['dist'].values
    posttime = 10
    time_df = np.linspace(-config.waveform_time, posttime, 
                         int((config.waveform_time + posttime) * config.sr))
    
    # Load and normalize waveforms
    waveforms = []
    for _, row in df_array.iterrows():
        tr = read(row['fname'])[-1]  # Get last trace if multiple components
        pkpdf = tr.stats.starttime + config.pre_window
        tr.trim(pkpdf - config.waveform_time, pkpdf + posttime)
        waveform = tr.data[:int((config.waveform_time + posttime) * config.sr)]
        waveforms.append(waveform)
    
    global_max = max(np.max(np.abs(wave[:int(config.waveform_time * config.sr)])) 
                    for wave in waveforms)
    waveforms = np.array(waveforms) / global_max
    
    # Plot waveforms and predictions
    for j, (_, row) in enumerate(df_array.iterrows()):
        # Plot waveform
        ax1.plot(time_df, -waveforms[j] + row['dist'], 
               color='grey', linewidth=0.7, zorder=1)
        
    # Configure axes
    ax1.set_ylim(max(dists)+0.5, min(dists)-0.5)  # Inverted distance axis
    ax1.set_xlim(-config.waveform_time, posttime)
    ax1.set_xlabel("Relative Time (s)")
    ax1.set_ylabel("Epicentral Distance (deg)")
    if title:
        ax1.set_title(title, fontsize=fontsize)
    ax1.minorticks_on()
    plt.tight_layout()
    
    # Save or display
    if save_fig:
        array_name = df_array.iloc[0]['array']
        save_path = os.path.join(output_dir, f"{array_name}_waveforms.pdf")
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    pass


def plot_array_analysis(waveforms, distances, pre_time, post_time, vespagram_data,
                        fk_data, smin, smax, vespa_slowness, fk_slowness, fk_backazimuth, 
                        array_name, stat):

    """
    Generate a three-panel array analysis plot including:
    - Normalized seismic waveforms by distance
    - Vespagram (slowness-time power image)
    - FK (frequency-wavenumber) slowness beamforming result

    Args:
        waveforms (list of ndarray): List of 1D seismic waveforms (same length).
        distances (list of float): Epicentral distances corresponding to waveforms.
        pre_time (float): Start time (s) relative to reference phase (e.g., PKPdf).
        post_time (float): End time (s) relative to reference phase.
        vespagram_data (2D ndarray): [slowness × time] array from beamforming.
        fk_data (2D ndarray): [slowness_y × slowness_x] array from FK analysis.
        smin (float): Minimum slowness (s/deg) for both vespagram and FK plots.
        smax (float): Maximum slowness (s/deg) for both vespagram and FK plots.
        vespa_slowness (float): Slowness value of the vespagram peak (s/deg).
        fk_slowness (float): Peak slowness value from FK analysis (s/deg).
        fk_backazimuth (float): Back-azimuth (deg) corresponding to FK peak.
        array_name (str): Name of array/network for plot title.
        stat (str): If 'power', apply positive-only vespagram normalization.

    Returns:
        None. Displays the plot interactively.
    """
    
    new_cmap = plt.get_cmap('Spectral_r')
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)
    
    # Waveform plot
    ax1 = fig.add_subplot(gs[0, 0])
    global_max = max(np.max(np.abs(wave[:500])) for wave in waveforms)
    for wave, dist in zip(waveforms, distances):
        time = np.linspace(pre_time, post_time, len(wave))
        norm_wave = wave / global_max / 5
        ax1.plot(time, -1 * norm_wave + dist, color='black', linewidth=0.7, zorder=1)
    ax1.set(xlabel="Relative Time (s)", ylabel="Distance (degree)", xlim=(pre_time, post_time))
    ax1.invert_yaxis()
    ax1.set_title(array_name, fontsize = fontsize)

    # Vespagram plot
    ax2 = fig.add_subplot(gs[0, 1])
    max_vespagram = np.max(abs(vespagram_data))
    norm_vespa = vespagram_data / max_vespagram
    
    cmap = new_cmap
    vmin, vmax = (0, 1) if stat == 'power' else (-1, 1)
    
    im = ax2.imshow(
        norm_vespa[::-1],
        cmap=cmap,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        extent=(pre_time, post_time, smin, smax)
    )
    ax2.set(xlabel='Relative Time (s)', ylabel='Slowness (s/deg)')
    ax2.minorticks_on()
    ax2.set_title(f'Vespagram Slowness: {vespa_slowness:.3f} s/deg', fontsize = fontsize)
    add_colorbar(fig, im, ax2, 'Normalized Amplitude')

    # FK plot
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.linspace(-smax, smax, fk_data.shape[1])
    y = np.linspace(-smax, smax, fk_data.shape[0])
    X, Y = np.meshgrid(x, y)
    norm_fk = fk_data / np.max(abs(fk_data))
    levels = np.linspace(0, 1, 21)  # 20 contour levels from 0 to 1

    cf = ax3.contourf(
        X, Y, norm_fk,
        levels=levels,
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    ax3.set(xlabel='East Slowness (s/deg)', ylabel='North Slowness (s/deg)')
    ax3.minorticks_on()
    ax3.set_title(f'FK Slowness and Back-azimuth: {fk_slowness:.3f} s/deg, {fk_backazimuth:.3f}°',
                  fontsize = fontsize)
    add_colorbar(fig, cf, ax3, 'Normalized Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def add_colorbar(fig, im, ax, label: str) -> None:
    """Add colorbar to the given axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    fig.colorbar(im, cax=cax, orientation='vertical', label=label)
    pass

def group_stations2arrays(df, azimuth_range=15, min_stations=10):
    """
    Group stations into virtual arrays where the azimuth coverage is less than azimuth_range degrees
    and each array contains at least min_stations stations.

    Args:
        df (pd.DataFrame): DataFrame containing station information, must include 'longitude' and 'latitude' columns.
        azimuth_range (float): Azimuth coverage range in degrees, default is 15.
        min_stations (int): Minimum number of stations required in each virtual array, default is 10.

    Returns:
        list: A list of DataFrames, where each DataFrame represents a virtual array.
    """

    # Sliding window grouping
    virtual_arrays = []
    for start_azimuth in np.arange(0, 360, 1):  # Slide in 1-degree steps
        end_azimuth = start_azimuth + azimuth_range
        if end_azimuth > 360:
            # Handle cases where the window crosses 360 degrees
            group = df[(df['azimuth'] >= start_azimuth) | (df['azimuth'] <= end_azimuth - 360)]
        else:
            group = df[(df['azimuth'] >= start_azimuth) & (df['azimuth'] <= end_azimuth)]
        if len(group) >= min_stations:
            virtual_arrays.append(group)

    return virtual_arrays

def filter_virtual_arrays(virtual_arrays):
    """
    Filter virtual arrays to remove duplicates and subsets.

    Args:
        virtual_arrays (list): A list of DataFrames, where each DataFrame represents a virtual array.

    Returns:
        list: A filtered list of unique virtual arrays.
    """
    # Sort virtual arrays by size (largest first)
    virtual_arrays.sort(key=lambda x: len(x), reverse=True)

    # Track stations that have already been included
    included_stations = set()
    filtered_arrays = []

    for array in virtual_arrays:
        # Get the stations in the current array
        stations = set(array['fname'])
        
        # Check if the current array adds new stations
        if stations.isdisjoint(included_stations):
            # Add the current array to the filtered list
            filtered_arrays.append(array)

            # Update the set of included stations
            included_stations.update(stations)

    return filtered_arrays