import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.filtering import filter_signal, normalize_signal
from utils.vital_quality_functions import *


# Estimate through the specified SQIs whether the segment is usable
def segment_is_usable(signal_segment, sqi_thresholds):
    
    evaluated_sqis = {}
    segment_is_usable = True
    
    if 'MSQ' in sqi_thresholds.keys():
        msq = msq_sqi(signal_segment)
        evaluated_sqis['MSQ'] = msq
        if msq < sqi_thresholds['MSQ']:
            segment_is_usable = False

    if 'zero_cross' in sqi_thresholds.keys():
        zero_cross = zero_crossings_rate_sqi(signal_segment)
        evaluated_sqis['zero_cross'] = zero_cross
        acceptable_range = sqi_thresholds['zero_cross']
        if zero_cross < acceptable_range[0] or zero_cross > acceptable_range[1]:
            segment_is_usable = False

    # add more SQIs to check

    return segment_is_usable, evaluated_sqis



def plot_segment_and_SQIs(signal_segment, segment_is_usable, evaluated_sqis):

    # legend_SQI_list = [f'{key}: {value}' for key, value in evaluated_sqis.items()]
    # legend_SQI_list.append(f'segment_is_usable: {segment_is_usable}')

    # print(legend_SQI_list)

    plt.figure()
    plt.plot(signal_segment)
    for key, value in evaluated_sqis.items():\
        plt.plot([], [], ' ', label=f'{key}: {value:.3f}')
    plt.plot([], [], ' ', label=f'segment_is_usable: {segment_is_usable}')
    plt.title('Signal Segment and corresponding SQIs')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


# Generation of unusable data lists for a single patient
def get_unusable_segment_list(file_path, segment_length, filter_params, sqi_thresholds, signal_wavelength, num_rows_to_read):
    
    # load one patient's PPG .csv file as a pandas dataframe
    signal_wavelength = signal_wavelength + '_ADC'
    signal_dataframe_column = pd.read_csv(file_path, nrows=num_rows_to_read, usecols=[signal_wavelength])
    signal_length = int(signal_dataframe_column.shape[0])

    # filter the entire signal based on the wavelength (either RED_ADC or IR_ADC)
    filtered_adc = filter_signal(signal_dataframe_column[signal_wavelength], filter_params)
    
    unusable_segment_start_times = []

    # iterate through the signal segments
    for start_index in range(0, signal_length, segment_length):
        signal_segment = filtered_adc[start_index : start_index + segment_length]
        signal_segment = normalize_signal(signal_segment)

        # this ensures that we only get slices of length segment_length avoiding
        # the residue of the signal after an integer multiple of segment_length
        if len(signal_segment) == segment_length:

            # estimate through the SQIs in sqi_list whether the segment is usable
            usable, evaluated_sqis = segment_is_usable(signal_segment, sqi_thresholds)

            # plot the signal segments along with the corresponding SQIs evaluated for that segment and the decision
            # plot_segment_and_SQIs(signal_segment, usable, evaluated_sqis)

            if not usable:
                unusable_segment_start_times.append(start_index)


    return unusable_segment_start_times


# Generation of unusable data lists for all patients
def patient_unusable_segments(base_path, segment_length, filter_params, sqi_thresholds, signal_wavelength, num_rows_to_read):

    patient_unusable_segments = {}

    for patient_folder in os.listdir(base_path):
        patient_ppg_folder = os.path.join(base_path, patient_folder, "PPG")
        try:
            for (dirpath, _, filenames) in os.walk(patient_ppg_folder):
                for filename in filenames:
                    if filename.endswith('.csv'):
                        file_path = os.path.join(dirpath, filename)
                        print(f"current patient and file: {file_path}")
                        unusable_segments = get_unusable_segment_list(file_path, segment_length, filter_params, sqi_thresholds, signal_wavelength, num_rows_to_read)
                        patient_unusable_segments[(patient_folder, filename)] = unusable_segments

        except Exception as e:
            print('==============================================================')
            print(f'ERROR: Reading from {patient_folder} failed due to a {type(e).__name__}')
            print(f'ERROR: Arguments: {e.args}')
            print('==============================================================')

    return patient_unusable_segments



def print_unusable_data(unusable_data):
    for dict_key in unusable_data.keys():
        patient, file = dict_key
        print(f'Patient: {patient} \nFile: {file}')
        print(f'Number of unusable segments: {len(unusable_data[dict_key])} \n')
