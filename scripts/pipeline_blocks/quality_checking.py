import math
import time


# Quality checking for a particular time period, check if it has sufficiently few unusable segments
# -----------------------------------------------------------------------------------------------------------
def segment_in_interval(segment_start, time_interval):
    start_time, end_time = time_interval
    start_sample, end_sample = int(start_time/10), int(end_time/10)
    return segment_start >= start_sample and segment_start <= end_sample



def num_unusable_segments_in_interval(patient_unusable_segments, time_interval):
    segment_in_interval_lambda = lambda segment: segment_in_interval(segment, time_interval)
    segments_in_interval = filter(segment_in_interval_lambda, patient_unusable_segments)
    return len(list(segments_in_interval))



def reduce_patient_matrix(patient_matrix, unusable_data, unusable_segment_size):
    # this refers to the maximum amount of the signal that we allow to be of poor quality
    # for example, if the signal is of length 8 segments and 1 of them is unusable then this
    # is equivalent to an unusable percentage of 1/8 = 12.5%
    allowed_unusable_percentage = 0.1
    reduced_patient_matrix = []

    for row in patient_matrix:
        (patient, file), time_interval, _, label = row
        # if patient == '01NVa-003-2201':
        patient_unusable_segments = unusable_data[(patient, file)]
        num_unusable_segments = num_unusable_segments_in_interval(patient_unusable_segments, time_interval)
        # print(f'patient_unusable_segments: {patient_unusable_segments}')
        # print(f'time_interval: {time_interval}')
        # print(f'num_unusable_segments: {num_unusable_segments}')
        signal_length = time_interval[1] - time_interval[0]
        unusable_percentage = (num_unusable_segments * unusable_segment_size) / signal_length

        # print(f'Unusable segment size: {unusable_segment_size}'
        # print(f'Patient {patient} time interval: {time_interval}')
        # print(f'Patient {patient} signal length: {signal_length}')
        # print(f'Patient {patient} number of segments that fit: {signal_length / unusable_segment_size}')
        # print(f'Patient {patient} number of unusable segments: {num_unusable_segments}')
        print(f'Patient, label: ({patient}, {label : >3}), Unusable percentage: {(unusable_percentage*100):.1f}%')

        if unusable_percentage < allowed_unusable_percentage:
            reduced_patient_matrix.append(row)


    return reduced_patient_matrix
# -----------------------------------------------------------------------------------------------------------
