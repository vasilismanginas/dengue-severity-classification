import numpy as np
from scipy import signal
from scipy.stats import skew


def get_moving_average(q, w):
    q_padded = np.pad(q, (w // 2, w - 1 - w // 2), mode='edge')
    convole = np.convolve(q_padded, np.ones(w) / w, 'valid')
    return convole


def get_ROI(s, mva):
    start_pos = []
    end_pos = []
    for idx in range(len(s) - 1):
        if mva[idx] > s[idx] and mva[idx + 1] < s[idx + 1]:
            start_pos.append(idx)
        elif mva[idx] < s[idx] and mva[idx + 1] > s[idx + 1] \
                and len(start_pos) > len(end_pos):
            end_pos.append(idx)
    if len(start_pos) > len(end_pos):
        end_pos.append(len(s) - 1)
    return start_pos, end_pos


def detect_peak_trough_adaptive_threshold(s, adaptive_size=0.75, overlap=0, sliding=1):
    """
    Parameters
    ----------
    s :
        param adaptive_size:
    overlap :
        overlapping ratio (Default value = 0)
    adaptive_size :
            (Default value = 0.75)
    sliding :
            (Default value = 1)
    Returns
    -------
    """
    # number of instances in the adaptive window
    adaptive_window = adaptive_size * 100
    adaptive_threshold = get_moving_average(s, int(adaptive_window * 2 + 1))

    start_ROIs, end_ROIs = get_ROI(s, adaptive_threshold)
    peak_finalist = []
    for start_ROI, end_ROI in zip(start_ROIs, end_ROIs):
        region = s[start_ROI:end_ROI + 1]
        peak_finalist.append(np.argmax(region) + start_ROI)

    trough_finalist = []
    for idx in range(len(peak_finalist) - 1):
        region = s[peak_finalist[idx]:peak_finalist[idx + 1]]
        trough_finalist.append(np.argmin(region) + peak_finalist[idx])

    return peak_finalist, trough_finalist


def detect_peak_trough_default_scipy(s):
    peak_finalist = signal.find_peaks(s)[0]
    trough_finalist = []
    for idx in range(len(peak_finalist) - 1):
        region = s[peak_finalist[idx]:peak_finalist[idx + 1]]
        trough_finalist.append(np.argmin(region) + peak_finalist[idx])

    return peak_finalist, trough_finalist


def detect_peak_trough_billauer(s):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    billauer_peakdet Detect peaks in a vector
            [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
            maxima and minima ("peaks") in the vector V.
            MAXTAB and MINTAB consists of two columns. Column 1
            contains indices in V, and column 2 the found values.
            With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
            in MAXTAB and MINTAB are replaced with the corresponding
            X-values.
            A point is considered a maximum peak if it has the maximal
            value, and was preceded (to the left) by a value lower by
            DELTA.
    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.
    Parameters
    ----------
    v :
        Vector of input signal to detect peaks
    delta :
        Parameter for determining peaks and valleys. A point is considered a maximum peak if
        it has the maximal value, and was preceded (to the left) by a value lower by delta. (Default value = 0.1)
    x :
        (Optional) Replace the indices of the resulting max and min vectors with corresponding x-values
    s :
    Returns
    -------
    """
    #Scale data
    s_min = np.min(s)
    s_max = np.max(s)
    s = np.interp(s, (s_min, s_max), (-1, +1))

    delta = 0.8
    maxtab = []
    mintab = []

    x = np.arange(len(s))
    v = np.asarray(s)
    assert np.isscalar(delta), 'Input argument delta must be a scalar'
    assert delta > 0, 'Input argument delta must be positive'

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True
    return np.array(maxtab), np.array(mintab)



def msq_sqi(signal_segment):
    """
    MSQ SQI as defined in Elgendi et al
    "Optimal Signal Quality Index for Photoplethysmogram Signals"
    with modification of the second algorithm used.
    Instead of Bing's, a SciPy built-in implementation is used.
    The SQI tracks the agreement between two peak detectors
    to evaluate quality of the signal.
    Parameters
    ----------
    signal_segment : sequence
        A signal with peaks.
    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal
    """
    peaks_1, _ = detect_peak_trough_adaptive_threshold(signal_segment)
    peaks_2, _ = detect_peak_trough_billauer(signal_segment)

    if len(peaks_1)==0 or len(peaks_2)==0:
        return 0.0

    peak1_dom = len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)
    peak2_dom = len(np.intersect1d(peaks_2,peaks_1))/len(peaks_2)

    # print("len(peaks_1)", len(peaks_1))
    # print("len(peaks_2)", len(peaks_2))
    # print("peak1_dom", peak1_dom)
    # print("peak2_dom", peak2_dom)
    
    return min(peak1_dom, peak2_dom)


def zero_crossings_rate_sqi(signal_segment, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1):
    """Reuse the function from librosa package.
    This is the rate of sign-changes in the processed signal, that is,
    the rate at which the signal changes from positive to negative or back.
    Parameters
    ----------
    signal_segment :
        list, array of signal
    threshold :
        float > 0, default=1e-10 if specified, values where
        -threshold <= signal_segment <= threshold are clipped to 0.
    ref_magnitude :
        float >0 If numeric, the threshold is scaled
        relative to ref_magnitude.
        If callable, the threshold is scaled relative
        to ref_magnitude(np.abs(signal_segment)). (Default value = None)
    pad :
        boolean, if True, then signal_segment[0] is considered a valid
        zero-crossing. (Default value = True)
    zero_pos :
        the crossing marker. (Default value = True)
    axis :
        axis along which to compute zero-crossings. (Default value = -1)
    Returns
    -------
    type
        float, indicator array of zero-crossings in `signal_segment` along the
        selected axis.
    """
    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(signal_segment))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        signal_segment = signal_segment.copy()
        signal_segment[np.abs(signal_segment) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        signal_segment_sign = np.signbit(signal_segment)
    else:
        signal_segment_sign = np.sign(signal_segment)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * signal_segment.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * signal_segment.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * signal_segment.ndim
    padding[axis] = (1, 0)

    crossings = np.pad(
        (signal_segment_sign[tuple(slice_post)] != signal_segment_sign[tuple(slice_pre)]),
        padding,
        mode="constant",
        constant_values=pad,
    )

    return np.mean(crossings, axis=0, keepdims=True)[0]


def skewness_sqi(signal_segment, axis=0, bias=True, nan_policy='propagate'):
    """Expose
    Skewness is a measure of symmetry, or more precisely, the lack of
    symmetry. A distribution, or data set, is symmetric if it looks the same
    to the left and right of the center point.
    Skewness is a measure of the symmetry (or the lack of it) of a
    probability distribution, which is defined as:
    SSQI=1/N∑i=1N[xi−μˆx/σ]3
    where μˆx and σ are the empirical estimate of the mean and standard
    deviation of xi,respectively; and N is the number of samples in the PPG
    signal.
    Parameters
    ----------
    signal_segment :
        list, the array of signal
    axis :
         (Default value = 0)
    bias :
         (Default value = True)
    nan_policy :
         (Default value = 'propagate')
    Returns
    -------
    """
    return skew(signal_segment, axis, bias, nan_policy)
