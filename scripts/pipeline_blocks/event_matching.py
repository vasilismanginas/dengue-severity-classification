import os
import pandas as pd
from datetime import datetime
from utils.helpers import patient_is_from_correct_cohort




# functions used for creating the patient matrix for the ICU-vs-Follow-Up question
# ==========================================================================================
def get_follow_up_recording(start_times_and_durations):
    follow_up_recording = max(start_times_and_durations)
    _, _, follow_up_file_name, follow_up_duration = follow_up_recording
    print(follow_up_duration)
    
    follow_up_duration_ms = int(follow_up_duration * 60 * 60 * 1000) # convert hours to milliseconds to add to start datetime
    follow_up_duration_ms = round(follow_up_duration_ms, -1) # round to nearest 10 since sampling rate is 100Hz and samples occur every 10ms
    follow_up_start_timestamp = 0
    follow_up_end_timestamp = follow_up_start_timestamp + follow_up_duration_ms

    return follow_up_file_name, follow_up_start_timestamp, follow_up_end_timestamp, follow_up_duration_ms


# Currently, the ICU segment we grab is from the middle of the recording with the largest duration
# the length of the segment is equal to the length of the follow-up segment to ensure a balanced dataset
def get_icu_recording(start_times_and_durations, follow_up_duration_ms):
    start_times_and_durations = sorted(start_times_and_durations)
    start_times_and_durations.remove(start_times_and_durations[-1])
    max_duration_recording = max(start_times_and_durations, key=lambda tuple_element: tuple_element[3])

    icu_recording = max_duration_recording[:]
    _, _, icu_file_name, icu_duration = icu_recording
    icu_duration_ms = int(icu_duration * 60 * 60 * 1000)
    icu_start_timestamp = int(icu_duration_ms / 2)
    icu_start_timestamp = round(icu_start_timestamp, -1)
    icu_end_timestamp = icu_start_timestamp + follow_up_duration_ms
    icu_starting_row = int(icu_start_timestamp / 10)

    return icu_file_name, icu_start_timestamp, icu_end_timestamp, icu_starting_row


def get_FU_ICU_patient_matrix(patient_info):
    follow_up_patient_list = [2201, 2202, 2203, 2205, 2206, 2208, 2211, 2218, 2219, 2220, 2221, 2222, 2226, 2227, 2231]

    patient_matrix = []
    for patient in follow_up_patient_list:
        patient = '003-' + str(patient)
        full_patient_name = '01NVa-' + patient
        patient_data = patient_info[patient]

        start_times_and_durations = []
        for recording in patient_data['SC']:
            (recording_file_name, start_time, end_time, recording_duration) = recording
            start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
            end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')
            start_times_and_durations.append((start_datetime, end_datetime, recording_file_name, float(recording_duration)))

        
        fu_file_name, fu_start_timestamp, fu_end_timestamp, fu_duration_ms = get_follow_up_recording(start_times_and_durations)
        icu_file_name, icu_start_timestamp, icu_end_timestamp, icu_starting_row = get_icu_recording(start_times_and_durations, fu_duration_ms)
        

        # starting row should be starting timestamp / 10. For follow-up case, both are 0 so no change.
        # for the file name we skip the first character because for some reason file names from the
        # patient info pickle have a space as the first character
        patient_matrix.append([(full_patient_name, fu_file_name), (fu_start_timestamp, fu_end_timestamp), fu_start_timestamp, 'FU'])
        patient_matrix.append([(full_patient_name, icu_file_name), (icu_start_timestamp, icu_end_timestamp), icu_starting_row, 'ICU'])

    return patient_matrix
# ==========================================================================================




# functions used for creating the patient matrix for the severity classification question
# ==========================================================================================
def get_severities_patient_matrix(patient_info):
    between_patient_list = list(range(2000, 2100))
    # remove_between_patients = [2003, 2007, 2009, 2010, 2013, 2015, 2019, 2022, 2024, 2033, 2050, 2051, 2053]
    # between_patient_list = [patient for patient in between_patient_list if patient not in remove_between_patients]
    
    severe_patient_list = list(range(2100, 2150))
    mild_patient_list = list(range(2150, 2201))

    patients_per_severity = {
        'between': between_patient_list,
        'severe': severe_patient_list,
        'mild': mild_patient_list
    }

    patient_matrix = []
    for severity_label, patient_list in patients_per_severity.items():

        for patient in patient_list:
            try:
                print(patient)
                patient = '003-' + str(patient)
                patient_data = patient_info[patient]

                irregular_patients = ['003-2123', '003-2178', '003-2187']
                if patient in irregular_patients:
                    full_patient_name = '01Nva-' + patient
                else:
                    full_patient_name = '01NVa-' + patient

                for recording in patient_data['SC']:
                    (recording_file_name, _, _, recording_duration) = recording

                    # for severity classification question we always use the entire signals
                    # from all patient recordings from that class so the start is always 0
                    start_timestamp = 0

                    # convert hours to milliseconds to add to start timestamp, also round to
                    # nearest 10 since sampling rate is 100Hz and samples occur every 10ms
                    recording_duration_ms = int(float(recording_duration) * 60 * 60 * 1000) 
                    recording_duration_ms = round(recording_duration_ms, -1)
                    
                    # obtain the end by adding the full recording duration to the start
                    end_timestamp = start_timestamp + recording_duration_ms

                    # since we are not interested in a particular type of event (e.g. Shock)
                    # the "event" of interest occurs from the beginning of the recording
                    occurrence_row = 0

                    row_for_patient_matrix = [(full_patient_name, recording_file_name), 
                                                (start_timestamp, end_timestamp), 
                                                occurrence_row, severity_label]
                    patient_matrix.append(row_for_patient_matrix)

            except Exception as e:
                print('==============================================================')
                print(f'ERROR: Reading from patient {patient} failed due to a {type(e).__name__}')
                print(f'ERROR: Arguments: {e.args}')
                print('==============================================================')

    return patient_matrix
# ==========================================================================================




# functions used for remaining events (currently only works for Shock/Reshock)
# ==========================================================================================

# returns the name of the time column of a dataframe
# depending on the type of event
def get_name_of_time_column(event_type):
    if event_type == 'Shock':
        return 'SHOCKSTIME'
    elif event_type == 'Reshock':
        return 'RSSTIME'
    else:
        return 'No name added for this column yet'
        

# returns a pandas series object containing the times of occurrence
# of event_type in a particular patient. Is only called if this type
# of event has occurred for a patient.
def get_times_of_occurrence(patient, patient_events, event_type):
    event_data = patient_events[event_type]
    time_column_name = get_name_of_time_column(event_type)
    times_of_occurrence = pd.to_datetime(event_data[time_column_name])
    num_of_occurrences = times_of_occurrence.shape[0]
    print('Patient:', patient, ':', num_of_occurrences, 'occurrences of event of type "' + event_type.lower() + '" detected')
    return times_of_occurrence


# returns the subset of occurrences that occur DURING one of the periods
# of PPG recording
def remove_unrecorded_occurrences(times_of_occurrence, patient_events):
    ppg_recording_files = patient_events['SC']
    occurrence_files_and_timestamps = []
    for _, event_time in times_of_occurrence.items():
        for recording in set(ppg_recording_files):
            (file_name, start_time, end_time, _) = recording
            # convert start_time and end_time from string to datetime
            # convert event_time from pandas timestamp to datetime
            start_datetime = datetime.strptime(start_time, ' %Y-%m-%d %H:%M:%S.%f')
            end_datetime = datetime.strptime(end_time, ' %Y-%m-%d %H:%M:%S.%f')
            event_datetime = event_time.to_pydatetime()
            if start_datetime <= event_datetime <= end_datetime:
                time_from_start_time = event_datetime - start_datetime
                
                time_difference_milliseconds = int(time_from_start_time.total_seconds() * 1000)
                # round to nearest tenth since sampling rate is 100Hz and so samples occur every 10ms
                time_difference_milliseconds = round(time_difference_milliseconds, -1)
                occurrence_files_and_timestamps.append((file_name, time_difference_milliseconds))
    
    return occurrence_files_and_timestamps


# returns a tuple of:
# event occurrence - Boolean
# timestamp (in milliseconds) of the valid (occurred during a PPG recording)
# event occurrences - list or None if event occurrence is False
def event_occured_during_recording(patient_info, patient, event_type):
    patient_events = patient_info[patient]
    event_types_present = patient_events.keys()
    # if event never occurred in this patient, immediately return False
    # otherwise attempt to extract the times of occurrence and subsequently
    # remove the occurrences that were outside the period of PPG recording
    if event_type not in event_types_present:
        return (False, None)
    else:
        try:
            times_of_occurrence = get_times_of_occurrence(patient, patient_events, event_type)            
            occurrence_files_and_timestamps = remove_unrecorded_occurrences(times_of_occurrence, patient_events)
            if not occurrence_files_and_timestamps:
                print('No occurrences found during recording')
            return (True, occurrence_files_and_timestamps)
        except Exception as e:
            print('=========================================================')
            print(f'ERROR: An error of type {type(e).__name__} occurred.')
            print(f'ERROR: Arguments: {e.args}')
            print('=========================================================')
            return (False, None)
            


def get_timestamp_row_number(base_path, patient, recording_file_name, shock_timestamp):
    patient_ppg_folder = os.path.join(base_path, patient, 'PPG')
    for (dirpath, _, filenames) in os.walk(patient_ppg_folder):
        for filename in filenames:
            if filename == recording_file_name:
                file_path = os.path.join(dirpath, filename)

    timestamp_df = pd.read_csv(file_path, usecols=['TIMESTAMP_MS'])
    event_occurence_row = timestamp_df[timestamp_df['TIMESTAMP_MS'] == shock_timestamp].index[0]

    return event_occurence_row



def event_matching(dataset_path, patient_info, event_type, window_size_samples):
    patient_matrix = []
    for patient in patient_info.keys():
        # function returns True if looking at adult patient in adult 
        # path or child patient in child path, False otherwise
        # need to check since patient_info dict contains all patients
        if patient_is_from_correct_cohort(patient, dataset_path):
            (event_occurred, occurrence_files_and_timestamps) = event_occured_during_recording(patient_info, patient, event_type)
            if event_occurred:
                patient = '01NVa-' + patient
                for recording_file_name, shock_timestamp in occurrence_files_and_timestamps:

                    event_occurence_row = get_timestamp_row_number(dataset_path, patient, recording_file_name, shock_timestamp)

                    # --- the following is only for shock and reshock
                    # We take 10 minutes before the shock as a pre-shock label
                    # and 10 minutes after the shock as a post-shock label
                    window_size_milliseconds = window_size_samples * 10
                    before_shock = shock_timestamp - window_size_milliseconds
                    after_shock = shock_timestamp + window_size_milliseconds

                    patient_matrix.append([(patient, recording_file_name), (before_shock, shock_timestamp), event_occurence_row, 'pre'])
                    patient_matrix.append([(patient, recording_file_name), (shock_timestamp, after_shock), event_occurence_row, 'post'])


    # patient_matrix_numpy = np.array(patient_matrix)
    return patient_matrix



def print_patient_matrix(patient_matrix, event_type):
    for row in patient_matrix:
        print(row)
    print(f'Data points for event of type "{event_type.lower()}": {len(patient_matrix)}')

    patients = [row[0][0] for row in patient_matrix]
    print(f'Number of patients for event of type "{event_type.lower()}": {len(set(patients))} \n')

# ==========================================================================================
