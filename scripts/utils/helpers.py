import os
import matplotlib.pyplot as plt

def patient_is_adult(patient):
    return int(patient.split('-')[-1]) <= 2200


def base_path_is_adult_path(base_path):
    return base_path.split(os.path.sep)[-1] == 'Adults'


def patient_is_from_correct_cohort(patient, dataset_path):
    # this if statement is an XNOR, i.e. True only if the two booleans are both True or both False.
    # so True if looking at adult patient in adult path or child patient in child path
    return not (patient_is_adult(patient) ^ base_path_is_adult_path(dataset_path))


def print_num_patients_with_SC(patient_info):
    adult_patients = [patient for patient in patient_info.keys() if patient < '003-2201']
    child_patients = [patient for patient in patient_info.keys() if patient > '003-2200']
    adult_patients_with_SC = []
    child_patients_with_SC = []


    for patient in adult_patients:
        patient_data = patient_info[patient]
        if 'SC' in patient_data.keys():
            list_SC_recordings = patient_data['SC']

            if list_SC_recordings:
                adult_patients_with_SC.append(patient)
    print(f'Number of adult patients with SC recordings in the pickle: {len(adult_patients_with_SC)}')


    for patient in child_patients:
        patient_data = patient_info[patient]
        if 'SC' in patient_data.keys():
            list_SC_recordings = patient_data['SC']

            if list_SC_recordings:
                child_patients_with_SC.append(patient)
    print(f'Number of child patients with SC recordings in the pickle: {len(child_patients_with_SC)} \n')
    

def plot_signal(signal, label, show=True):
    plt.figure()
    plt.plot(signal, label=label)
    plt.title('Plot of signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    if show:
        plt.show()