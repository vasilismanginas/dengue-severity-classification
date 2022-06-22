import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, butter, freqz, sosfilt, sosfiltfilt



''' Expand for comments:
    Filter creating function:
    Receives:
        - filter parameters. This is a dictionary and includes:
            + filter type (this can be either cheby (for ChebyshevII) or butter (for Butterworth))
            + filter order
            + sampling rate (Hz)
            + lower cutoff frequency (Hz)
            + higher cutoff frequency (Hz)
    Returns:
        - second-order sections representation of the IIR filter (used to do the actual filtering of signals)
        - numerator polynomial of the IIR filter
        - denominator polynomial of the IIR filter '''
def make_filter(filter_params):
    nyquist_f = filter_params['sample_rate_Hz'] / 2
    low_cutoff_normal = filter_params['low_cutoff_Hz'] / nyquist_f
    high_cutoff_normal = filter_params['high_cutoff_Hz'] / nyquist_f

    # Chebyshev II filter
    if filter_params['type'] == 'cheby':
        sos = cheby2(N=filter_params['order'], 
                     rs=40, 
                     Wn=[low_cutoff_normal, high_cutoff_normal], 
                     btype='bandpass', 
                     analog=False, 
                     output='sos')

        num, denom = cheby2(N=filter_params['order'], 
                            rs=40, 
                            Wn=[low_cutoff_normal, high_cutoff_normal], 
                            btype='bandpass', 
                            analog=False, 
                            output='ba')

    # Butterworth filter
    elif filter_params['type'] == 'butter':
        sos = butter(N=filter_params['order'], 
                     Wn=[low_cutoff_normal, high_cutoff_normal], 
                     btype='bandpass', 
                     analog=False, 
                     output='sos')

        num, denom = butter(N=filter_params['order'], 
                            Wn=[low_cutoff_normal, high_cutoff_normal], 
                            btype='bandpass', 
                            analog=False, 
                            output='ba')

    else:
        raise ValueError("Invalid filter type selected. Choose between ['cheby', 'butter']")

    return sos, num, denom



''' Expand for comments:
    Digital filter frequency response plotter:
    Receives:
        - numerator, denominator polynomials of the IIR filter
    Returns:
        - nothing (simply shows a plot of the frequency response) '''
def plot_digital_filter_response(num, denom, sample_rate_hz):
    w, h = freqz(num, denom, worN=10000)
    frequencies = w * sample_rate_hz / (2 * np.pi)
    magnitude = 20 * np.log10(abs(h))

    plt.semilogx(frequencies, magnitude)
    plt.axis([0.001, sample_rate_hz, -100, 5])
    plt.title('Filter magnitude frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()



''' Expand for comments:
    Signal filtering function:
    Receives:
        - signal to be filtered
        - filter parameters. This is a dictionary and includes:
            + filter type (this can be either cheby (for ChebyshevII) or butter (for Butterworth))
            + filter order
            + sampling rate (Hz)
            + lower cutoff frequency (Hz)
            + higher cutoff frequency (Hz)
        - plot response boolean: if True plot filter frequency response, default=False
    Returns:
        - filtered signal '''
def filter_signal(signal, filter_params, plot_response=False):
    
    sos, num, denom = make_filter(filter_params)

    if plot_response:
        plot_digital_filter_response(num, denom, filter_params['sample_rate_Hz'])
        plt.show()

    # filtered_signal = sosfilt(sos, signal)
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal



def normalize_signal(signal_segment):

	# Scaling:
	# 	Normalize between -1, 1
    min_x = min(signal_segment)
    max_x = max(signal_segment)
    normalized_filtered_signal = 2 * ((signal_segment - min_x) / (max_x - min_x)) - 1

    # Averaging (currently not implemented):
	# 	Savetzky Golay
	# 	Rolling mean

    return normalized_filtered_signal

