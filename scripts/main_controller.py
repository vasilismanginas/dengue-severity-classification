import os
import gc
import csv
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from pipeline_blocks.core_pipeline_function_wrappers import generate_unusable_data_lists
from pipeline_blocks.core_pipeline_function_wrappers import generate_and_reduce_patient_matrix
from pipeline_blocks.core_pipeline_function_wrappers import generate_features
from utils.pickle_functions import pickle_already_exists, load_pickle
from utils.helpers import print_num_patients_with_SC

from pipeline_blocks.event_matching import print_patient_matrix
from pipeline_blocks.model_training import get_train_and_val_data
from pipeline_blocks.model_training import create_and_compile_model, get_binary_class_occurrences, train_and_evaluate_model, plot_model_histories, plot_average_model_history

from baseline_model_functions import get_model, remove_nan_values_from_dataset, print_class_balance, cross_validation, preprocess_fft



def load_or_create_pickle(pickle_path, function_to_generate_pickle, **function_kwargs):
    if pickle_already_exists(pickle_path):
        data = load_pickle(pickle_path)
    else:
        start_time = time.time()
        function_to_generate_pickle(**function_kwargs)
        data = load_pickle(pickle_path)
        end_time = time.time()
        print(f'Running {function_to_generate_pickle.__name__} took {end_time - start_time} seconds')
    
    return data



if __name__ == "__main__":

    root_path = os.getcwd()
    base_path = r"Z:\projects\vital_project\live\01NVa_Dengue\RAW_DATA"

    event_type = 'severities'
    # event_type = 'ICU-FU'

    if event_type == 'severities':
        cohort_of_interest = 'adults'
        dataset_path = os.path.join(base_path, "Adults")
    elif event_type == 'ICU-FU':
        cohort_of_interest = 'children'
        dataset_path = os.path.join(base_path, "Children")
    else:
        raise ValueError("Invalid choice of event type. Pipeline currently thoroughly tested only for ['ICU-FU', 'severities']")
    

    sampling_rate = 100
    
    # this refers to the length of the segments that we examine
    # when determining whether a segment is unusable or not
    # for reference: 525.6 seconds is the duration of STFT used
    unusable_segment_length_seconds = 52.56
    unusable_segment_length_samples = int(unusable_segment_length_seconds * sampling_rate)
    # used for obtaining the unusable segments, should always 
    # be None (non-None values used only for testing)
    num_rows_to_read = None


    # only important for Shock/Reshock events for now this window refers to the 
    # time period that we are interested in before and after an event occurrence
    window_size_minutes = 2
    window_size_samples = window_size_minutes * 60 * sampling_rate


    # only important for RAW FE method (i.e, using raw signal as features)
    # this refers to the length of the segments that we cut the signal 
    # up into to produce training samples in the feature extraction stage
    fe_segment_length_seconds = 30
    fe_segment_length_samples = int(fe_segment_length_seconds * sampling_rate)


    # choose which wavelength to use from the two that the Smartcare sensor collects: 'RED' or 'IR'
    signal_wavelength = 'RED'

    # fe_method = 'RAW'
    # fe_method = 'FFT'
    fe_method = 'STFT'

    # this refers to whether we are using an STFT of the full possible length 
    # (optimally 10 wavelengths of the lowest frequency of interest) or whether 
    # we are using a smaller STFT to increase the number of training examples. 
    # The STFT divider dictates how many times smaller the chosen STFT is then the "optimal" one
    window_divider = 1


    num_of_classes = 2
    test_split = 0.2
    epochs = 100

    filter_params = {
        'type' : 'cheby',
        'order' : 2,
        'sample_rate_Hz' : 100,
        'low_cutoff_Hz' : 0.15,
        'high_cutoff_Hz' : 20
    }

    sqi_thresholds = {
        'MSQ' : 0.8,
        'zero_cross' : [0.01, 0.04]
    }



    # define pickle names and locations for the different stages of the pipeline
    patient_info_pickle_name = 'patient_info.pkl'
    unusable_data_pickle_name = f'unusable_data_{cohort_of_interest}_{signal_wavelength}_{unusable_segment_length_samples}_{num_rows_to_read}.pkl'
    patient_matrix_pickle_name = f'patient_matrix_{cohort_of_interest}_{signal_wavelength}_{event_type}_{window_size_samples}.pkl'
    training_data_pickle_name = f'feature_extraction_{cohort_of_interest}_{signal_wavelength}_{event_type}_{fe_method}_{window_divider}_{fe_segment_length_samples}.pkl'

    patient_info_pickle_path = os.path.join(root_path, 'data_pickles', 'patient_info', patient_info_pickle_name)
    unusable_data_pickle_path = os.path.join(root_path, 'data_pickles', 'unusable_data', unusable_data_pickle_name)
    patient_matrix_pickle_path = os.path.join(root_path, 'data_pickles', 'patient_matrix', patient_matrix_pickle_name)
    training_data_pickle_path = os.path.join(root_path, 'data_pickles', 'feature_extraction', fe_method, training_data_pickle_name)


    # load Stefan's patient info pickle
    patient_info = load_pickle(patient_info_pickle_path)
    print_num_patients_with_SC(patient_info)



    # check if there already exists a pickle with the unusable data for the current parameters chosen
    # currently only accounts for cohort and the length of the segments to be examined for quality checking
    # if the pickle exists, load it, if not, generate it
    unusable_data = load_or_create_pickle(pickle_path=unusable_data_pickle_path, 
                                          function_to_generate_pickle=generate_unusable_data_lists,
                                          base_path=dataset_path,
                                          unusable_segment_length_samples=unusable_segment_length_samples, 
                                          filter_params=filter_params, 
                                          sqi_thresholds=sqi_thresholds, 
                                          unusable_data_pickle_path=unusable_data_pickle_path,
                                          signal_wavelength=signal_wavelength,
                                          num_rows_to_read=num_rows_to_read)


    # check if there already exists a pickle with the patient matrix for the current parameters chosen
    # currently only accounts for cohort, event type, and the size of the window before and after the event
    # if the pickle exists, load it, if not, generate it
    reduced_patient_matrix = load_or_create_pickle(pickle_path=patient_matrix_pickle_path, 
                                                   function_to_generate_pickle=generate_and_reduce_patient_matrix,
                                                   base_path=dataset_path,
                                                   patient_info=patient_info,
                                                   event_type=event_type,
                                                   window_size_samples=window_size_samples,
                                                   unusable_data=unusable_data,
                                                   unusable_segment_length_samples=unusable_segment_length_samples,
                                                   patient_matrix_pickle_path=patient_matrix_pickle_path)
    print('Reduced patient matrix:')
    print_patient_matrix(reduced_patient_matrix, event_type)
    print('\n')


    # check if there already exists a pickle with the training data/features for the current parameters chosen
    # currently only accounts for cohort, window size, and the size of segments we cut up the signal for FE
    # if the pickle exists, load it, if not, generate it
    feature_extraction_data = load_or_create_pickle(pickle_path=training_data_pickle_path,
                                                    function_to_generate_pickle=generate_features,
                                                    base_path=dataset_path,
                                                    event_type=event_type,
                                                    reduced_patient_matrix=reduced_patient_matrix,
                                                    window_size_samples=window_size_samples,
                                                    filter_params=filter_params,
                                                    signal_wavelength=signal_wavelength,
                                                    fe_method=fe_method,
                                                    fe_segment_length_samples=fe_segment_length_samples,
                                                    window_divider=window_divider,
                                                    training_data_pickle_path=training_data_pickle_path)
    


    start_time = time.time()

    model_type = 'SVM'
    model = get_model(model_type)

    print('Model type: ', model_type)

    if fe_method == 'RAW':
        inputs, outputs = feature_extraction_data
        label_encoder = LabelEncoder()
        integer_outputs = label_encoder.fit_transform(outputs)
        inputs = np.asarray(inputs)
        inputs, integer_outputs = shuffle(inputs, integer_outputs)
        print(f'Shape of input data: {inputs.shape}')
        print(f'Shape of output data: {integer_outputs.shape}')

        if event_type == 'severities':
            inputs_contain_nans = np.isnan(inputs).any()
            if inputs_contain_nans:
                print('Input data contains nan values, removing now...')
                inputs, integer_outputs = remove_nan_values_from_dataset(inputs, integer_outputs)
            else:
                print('Input data does not contain nan values, all good!')
            print_class_balance(inputs, integer_outputs)

        gc.collect()
        num_trials = 1
        all_results = {
            "Mean Validation Accuracy": [],
            "Mean Validation Precision": [],
            "Mean Validation Recall": [],
            "Mean Validation Macro F1 Score": [],
            "Mean Validation Weighted F1 Score": [],
        }

        for i in range(num_trials):
            gc.collect()
            print(f'Currently running cross validation for trial {i+1}')
            trial_results = cross_validation(model, inputs, integer_outputs)
            for metric_name, metric_value in trial_results.items():
                all_results[metric_name].append(metric_value)


        average_results = {}
        for metric_name, metric_values in all_results.items():
            average_metric_value = sum(metric_values) / len(metric_values)
            average_results[metric_name] = average_metric_value
        
        print(f'\nAverage metrics over {num_trials} trials for a {model_type} model using {fe_method} for feature extraction:')
        for metric_name, average_metric_value in average_results.items():
            print(f'{metric_name}: {average_metric_value:.3f}')
        
        print('\n')
        for metric_name, average_metric_value in average_results.items():
            print(f'{average_metric_value:.3f}')


    elif fe_method == 'FFT':
        highest_power_of_two = 10
        results_for_all_bins = {}
        for power_of_two in range(2, highest_power_of_two + 1):
            gc.collect()
            inputs, outputs = feature_extraction_data
            label_encoder = LabelEncoder()
            integer_outputs = label_encoder.fit_transform(outputs)

            truncating_point = int(0.4 * len(inputs[0]))
            num_bins = 2 ** power_of_two

            inputs = [preprocess_fft(fft, num_bins, truncating_point) for fft in inputs]
            inputs = np.asarray(inputs)

            label_encoder = LabelEncoder()
            integer_outputs = label_encoder.fit_transform(outputs)

            print(f'\n\nNumber of bins: {num_bins}')
            print(f'Shape of input data: {inputs.shape}')
            print(f'Shape of output data: {integer_outputs.shape}')

            if event_type == 'severities':
                inputs_contain_nans = np.isnan(inputs).any()
                if inputs_contain_nans:
                    print('Input data contains nan values, removing now...')
                    inputs, integer_outputs = remove_nan_values_from_dataset(inputs, integer_outputs)
                else:
                    print('Input data does not contain nan values, all good!')
                print_class_balance(inputs, integer_outputs)


            num_trials = 1
            all_results = {
                "Accuracy": [],
                "Macro Precision": [],
                "Macro Recall": [],
                "Macro F1 Score": [],
                "Weighted F1 Score": [],
            }

            for i in range(num_trials):
                trial_results = cross_validation(model, inputs, integer_outputs)

                for metric_name, metric_value in trial_results.items():
                    all_results[metric_name].append(metric_value)


            average_results = {}
            for metric_name, metric_values in all_results.items():
                average_metric_value = sum(metric_values) / len(metric_values)
                average_results[metric_name] = average_metric_value
        
            results_for_all_bins[str(num_bins)] = average_results
            print(results_for_all_bins)

        
        with open(f'{model_type}.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            header = [model_type]
            writer.writerow(header)

            for metric_name in all_results.keys():
                metric_for_all_bins = []
                metric_for_all_bins.append(metric_name)
                for bin_num, metrics in results_for_all_bins.items():
                    metric_for_all_bins.append(round(metrics[metric_name], 3))

                print(metric_for_all_bins)
                writer.writerow(metric_for_all_bins)


    else:
        raise ValueError("Invalid FE method chosen, choose between ['RAW', 'FFT']")
    
    end_time = time.time()
    print(f'Model training took {end_time - start_time} seconds')
