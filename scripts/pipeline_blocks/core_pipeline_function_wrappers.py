import pickle

from pipeline_blocks.quality_analysis import patient_unusable_segments, print_unusable_data
from pipeline_blocks.event_matching import event_matching, get_FU_ICU_patient_matrix, get_severities_patient_matrix, print_patient_matrix
from pipeline_blocks.quality_checking import reduce_patient_matrix
from pipeline_blocks.feature_extraction import feature_extraction


'''
The functions below are wrappers for the core functions of various pipeline stages.
They print the current state of the pipeline, execute the pipeline functions, and
finally store the results of these functions in pickle files, which are then loaded
in the main controller file.
'''
def generate_unusable_data_lists(base_path, unusable_segment_length_samples, filter_params, sqi_thresholds, unusable_data_pickle_path, signal_wavelength, num_rows_to_read=None):
    print('\n* Generating unusable segments lists...')
    unusable_data = patient_unusable_segments(base_path, 
                                              unusable_segment_length_samples, 
                                              filter_params, 
                                              sqi_thresholds, 
                                              signal_wavelength, 
                                              num_rows_to_read)
    unusable_data_pickle = open(unusable_data_pickle_path, "wb")
    pickle.dump(unusable_data, unusable_data_pickle)
    unusable_data_pickle.close()
    print('* Done generating unusable segments lists! \n\n')



def generate_and_reduce_patient_matrix(base_path, patient_info, event_type, window_size_samples, unusable_data, unusable_segment_length_samples, patient_matrix_pickle_path):
    # event matching is given the patient information in form of the pickle data 
    # and return the patient matrix. Event type specified by input parameter
    print('* Event matching...')
    if event_type == 'ICU-FU':
        patient_matrix = get_FU_ICU_patient_matrix(patient_info)
    elif event_type == 'severities':
        patient_matrix = get_severities_patient_matrix(patient_info)
    else:
        patient_matrix = event_matching(base_path, 
                                        patient_info, 
                                        event_type, 
                                        window_size_samples)
    print('\nPatient matrix:')
    print_patient_matrix(patient_matrix, event_type)
    print('* Done event matching! \n\n')


    print('* Quality checking...')
    # quality checking stage reduces the patient matrix by removing the patients 
    # that are unusable from a quality perspective
    reduced_patient_matrix = reduce_patient_matrix(patient_matrix,
                                                   unusable_data, 
                                                   unusable_segment_length_samples)
    patient_matrix_pickle = open(patient_matrix_pickle_path, "wb")
    pickle.dump(reduced_patient_matrix, patient_matrix_pickle)
    patient_matrix_pickle.close()
    print('* Done quality checking! \n\n')



def generate_features(base_path, event_type, reduced_patient_matrix, window_size_samples, filter_params, signal_wavelength, fe_method, fe_segment_length_samples, window_divider, training_data_pickle_path):
    print('* Extracting features...')
    train_and_val_data = feature_extraction(base_path,
                                            event_type, 
                                            reduced_patient_matrix, 
                                            window_size_samples,
                                            filter_params,
                                            signal_wavelength,
                                            fe_method,
                                            fe_segment_length_samples,
                                            window_divider)

    training_data_pickle = open(training_data_pickle_path, "wb")
    pickle.dump(train_and_val_data, training_data_pickle)
    training_data_pickle.close()
    print('* Done extracting features! \n\n')
