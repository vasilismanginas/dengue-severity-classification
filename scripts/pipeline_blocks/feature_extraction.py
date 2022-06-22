from multiprocessing.sharedctypes import Value
import os
import scipy
import scipy.fft as scipy_fft
# import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.filtering import filter_signal, normalize_signal
from utils.helpers import plot_signal



def get_num_rows_to_skip_and_read(event_type, row, window_size_samples):
    _, (start_timestamp, end_timestamp), starting_row, label = row

    if event_type == 'Shock' or event_type == 'Reshock':
        if label == 'pre':
            num_rows_to_skip = starting_row - window_size_samples + 1
        else:
            num_rows_to_skip = starting_row + 1
        num_rows_to_read = window_size_samples + 1

    else: 
        # read the entire file, currently used 
        # for 'ICU-FU' or 'severities' events
        num_rows_to_skip = starting_row + 1
        num_rows_to_read = int((end_timestamp - start_timestamp) / 10)

    return num_rows_to_skip, num_rows_to_read



def get_preprocessed_signal(file_path, num_rows_to_skip, num_rows_to_read, filter_params, signal_wavelength):
    signal_dataframe = pd.read_csv(file_path, 
                                   skiprows=num_rows_to_skip, 
                                   nrows=num_rows_to_read,
                                   names=['TIMESTAMP_MS',
                                          'COUNTER',
                                          'DEVICE_ID',
                                          'PULSE_BPM',
                                          'SPO2_PCT',
                                          'SPO2_STATUS',
                                          'PLETH',
                                          'BATTERY_PCT',
                                          'RED_ADC',
                                          'IR_ADC',	
                                          'PERFUSION_INDEX'])

    signal_wavelength = signal_wavelength + '_ADC'
    # normalized_adc = normalize_signal(signal_dataframe[signal_wavelength])
    filtered_adc = filter_signal(signal_dataframe[signal_wavelength], filter_params)

    return filtered_adc



def segment_signal(signal, label, segment_length):
    segment_label_pairs = []
    for i in range(0, len(signal), segment_length):
        current_segment = signal[i:i + segment_length]
        if len(current_segment) == segment_length:
            current_segment = normalize_signal(current_segment)
            segment_label_pairs.append((current_segment, label))

    return segment_label_pairs



def get_shortest_signal_length(base_path, event_type, reduced_patient_matrix, filter_params, signal_wavelength):
    sampling_rate = filter_params['sample_rate_Hz']
    min_duration_row = min(reduced_patient_matrix, key=lambda row: row[1][1] - row[1][0])
    (patient, recording_file_name), (start_timestamp, end_timestamp), _, _ = min_duration_row

    print(f'Patient with shortest signal length: {patient : >23}')
    print(f'Expected signal length: {int((end_timestamp - start_timestamp) / 10) : >27}')
    print(f'Expected signal duration (minutes): {int((end_timestamp - start_timestamp) / 10) / (sampling_rate * 60) : >16}')
    patient_ppg_folder = os.path.join(base_path, patient, 'PPG')

    for (dirpath, _, filenames) in os.walk(patient_ppg_folder):
            for filename in filenames:
                if filename == recording_file_name:
                    file_path = os.path.join(dirpath, filename)

    num_rows_to_skip, num_rows_to_read = get_num_rows_to_skip_and_read(event_type, min_duration_row, None)
    filtered_red_adc = get_preprocessed_signal(file_path=file_path,
                                                num_rows_to_skip=num_rows_to_skip,
                                                num_rows_to_read=num_rows_to_read,
                                                filter_params=filter_params,
                                                signal_wavelength=signal_wavelength)

    print(f'Actual signal length: {len(filtered_red_adc) : >29}')
    print(f'Actual signal duration (minutes): {len(filtered_red_adc) / (sampling_rate * 60) : >16}\n')
    
    return len(filtered_red_adc)



def print_info_about_window_length(window_size_samples, sampling_rate, low_cutoff_frequency, num_stft_windows):
    window_duration_sec = window_size_samples / sampling_rate
    # this is the number of wavelengths of the lowest frequency component 
    # that we're interested in, wavelength = 1 / frequency
    num_wavelengths = window_duration_sec * low_cutoff_frequency

    print(f'Window size required for {num_stft_windows} windows: {window_size_samples : >14}')
    print(f'Window duration (seconds): {window_duration_sec : >23}')
    print(f'Number of wavelengths of lowest frequency: {num_wavelengths : >8}')
    print(f'Length of STFT for {num_stft_windows} windows of length {window_size_samples}: {num_stft_windows * window_size_samples : >6}\n\n')



def get_stft_window_length(base_path, event_type, reduced_patient_matrix, filter_params, signal_wavelength, window_divider):
    sampling_rate = filter_params['sample_rate_Hz']
    low_cutoff_frequency = filter_params['low_cutoff_Hz']   
    
    if event_type == 'ICU-FU':
        num_stft_windows = 8
        shortest_signal_length = get_shortest_signal_length(base_path=base_path, 
                                                            event_type=event_type, 
                                                            reduced_patient_matrix=reduced_patient_matrix, 
                                                            filter_params=filter_params, 
                                                            signal_wavelength=signal_wavelength)
        
        stft_window_length = int(shortest_signal_length / num_stft_windows)
        stft_window_length = int(stft_window_length / window_divider)
        print_info_about_window_length(stft_window_length * window_divider, sampling_rate, low_cutoff_frequency, num_stft_windows)

    elif event_type == 'severities':
        # STFT length equal to 10 wavelengths of the lowest frequency of interest
        stft_window_length = int(10 * (1 / low_cutoff_frequency) * sampling_rate)
        num_stft_windows = 32
        stft_length = stft_window_length * num_stft_windows
        stft_window_length = int(stft_window_length / window_divider)

    else:
        raise ValueError('Invalid event type. Choose between "ICU-FU" and "severities"')

    return stft_window_length, num_stft_windows



def get_fft_window_length(filter_params, window_divider):
    sampling_rate = filter_params['sample_rate_Hz']
    low_cutoff_frequency = filter_params['low_cutoff_Hz']   

    fft_window_length = int(10 * (1 / low_cutoff_frequency) * sampling_rate)
    fft_window_length = int(fft_window_length / window_divider)

    return fft_window_length



def plot_full_and_truncated_FFT(fft_length, signal_slice):
    fft = scipy.fft.rfft(signal_slice)
    xf = scipy.fft.rfftfreq(fft_length, 1 / 100)
    full_fft = scipy.fft.fft(signal_slice)
    full_xf = scipy.fft.fftfreq(fft_length, 1 / 100)
    truncating_point = int(0.4 * len(fft))
    reduced_fft = fft[:truncating_point]
    reduced_xf = xf[:truncating_point]

    plt.figure()
    plt.plot(full_xf, full_fft)
    plt.title('Full FFT Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.figure()
    plt.plot(reduced_xf, np.abs(reduced_fft))
    plt.title('Magnitude of Truncated Half-Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()



def plot_spectrograms(feature_label_pairs, num_stfts, frequency_range, time_range):
    plt.figure()
    num_stfts = int(np.floor(num_stfts))

    for i in range(num_stfts):

        stft = feature_label_pairs[i][0]
        stft = stft.reshape((stft.shape[0], stft.shape[1]))

        plt.subplot(1, num_stfts, i+1)
        plt.pcolormesh(time_range, 
                    frequency_range, 
                    np.abs(stft), 
                    # vmin=0,
                    # vmax=1,
                    shading='gouraud')
        plt.ylim([0, 10])
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

    plt.show()



def extract_features(signal, label, fe_method, **fe_parameters):
    
    feature_label_pairs = []

    # if no FE method selected, the raw signal is split into smaller segments
    if fe_method == 'RAW':
         # simply segment the signal into smaller chunks
        feature_label_pairs  = segment_signal(signal, label, fe_parameters['segment_length'])

    elif fe_method == 'FFT':
        fft_length = fe_parameters['fft_length']
        num_ffts = len(signal) / fft_length
        print(f'Number of FFTs for this patient: {int(np.floor(num_ffts)) : >13}\n')

        for i in range(0, len(signal), fft_length):
            signal_slice = signal[i:i + fft_length]

            # this ensures that we only get slices of length fft_length avoiding
            # the residue of the signal after an integer multiple of fft_length
            if len(signal_slice) == fft_length:
                fft = scipy.fft.rfft(signal_slice)
                    
                if fe_parameters['plot_ffts']:
                    plot_full_and_truncated_FFT(fft_length, signal_slice)

                # # add two extra dimensions (1D -> 3D) since keras CNN model layers 
                # # require input format (batch_size, height, width, channels)
                # fft = np.expand_dims(fft, axis=0)
                # fft = np.expand_dims(fft, axis=2)

                feature_label_pairs.append((fft, label))

    elif fe_method == 'STFT':
        stft_length = fe_parameters['num_stft_windows'] * fe_parameters['stft_window_length']
        num_stfts = len(signal) / stft_length
        print(f'Number of STFTs for this patient: {int(np.floor(num_stfts)) : >13}\n')


        for i in range(0, len(signal), stft_length):
            signal_slice = signal[i:i + stft_length]

            # this ensures that we only get slices of length stft_length avoiding
            # the residue of the signal after an integer multiple of stft_length
            if len(signal_slice) == stft_length:

                signal_slice = normalize_signal(signal_slice)
                # plot_signal(signal_slice, label, show=True)

                frequency_range, time_range, stft = scipy.signal.stft(x=signal_slice,
                                                                     fs=fe_parameters['sampling_rate'],
                                                                     nperseg=fe_parameters['stft_window_length'],
                                                                     noverlap=0)

                # add an extra dimension (2D -> 3D) since keras CNN model layers 
                # require input format (batch_size, height, width, channels)
                stft = np.expand_dims(stft, axis=2)
                feature_label_pairs.append((stft, label))

        feature_label_pairs = np.asarray(feature_label_pairs, dtype=object)

        if fe_parameters['plot_spectrograms']:
            plot_spectrograms(feature_label_pairs, num_stfts, frequency_range, time_range)

    else:
        raise ValueError("Invalid choise of feature extraction method. Choose between ['RAW', 'FFT', 'STFT']")

    return feature_label_pairs



def feature_extraction(base_path, event_type, reduced_patient_matrix, window_size_samples, filter_params, signal_wavelength, fe_method, segment_length, window_divider):
    model_data = []

    print(f'Base dataset path: {base_path}')
    print(f'Feature extraction method chosen: {fe_method}\n')

    if fe_method == 'RAW':
        fe_parameters = {'segment_length': segment_length}
    
    elif fe_method == 'FFT':
        fft_length = get_fft_window_length(filter_params, window_divider)
        fe_parameters = {
            'fft_length': fft_length,
            'plot_ffts': False
        }

    elif fe_method == 'STFT':
        stft_window_length, num_stft_windows = get_stft_window_length(base_path, event_type, reduced_patient_matrix, filter_params, signal_wavelength, window_divider)
        fe_parameters = {
            'sampling_rate': filter_params['sample_rate_Hz'], 
            'stft_window_length': stft_window_length,
            'num_stft_windows': num_stft_windows, 
            'plot_spectrograms': False
        }

    else:
        raise ValueError("Invalid feature extraction method chosen, choose between ['RAW', 'FFT', 'STFT'")


    # iterate through every row of the patient matrix
    # for each row extract a set of feature-label pairs from the signal
    for row in reduced_patient_matrix:
        (patient, recording_file_name), _, _, label = row
        patient_ppg_folder = os.path.join(base_path, patient, 'PPG')
        for (dirpath, _, filenames) in os.walk(patient_ppg_folder):
            for filename in filenames:
                if filename == recording_file_name:
                    file_path = os.path.join(dirpath, filename)


        print(f'Current patient: {patient  : >43}')
        num_rows_to_skip, num_rows_to_read = get_num_rows_to_skip_and_read(event_type, row, window_size_samples)
        filtered_red_adc = get_preprocessed_signal(file_path=file_path,
                                               num_rows_to_skip=num_rows_to_skip,
                                               num_rows_to_read=num_rows_to_read,
                                               filter_params=filter_params,
                                               signal_wavelength=signal_wavelength)

        # plot_signal(filtered_red_adc, label, show=True)

        feature_label_pairs = extract_features(signal=filtered_red_adc, 
                                               label=label, 
                                               fe_method=fe_method,
                                               **fe_parameters)
        

        # append feature map to the data to be used for training, validation, and testing by the model
        model_data.extend(feature_label_pairs)


    return list(zip(*model_data))

