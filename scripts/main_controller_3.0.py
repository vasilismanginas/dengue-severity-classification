import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from pipeline_blocks.core_pipeline_function_wrappers import generate_unusable_data_lists
from pipeline_blocks.core_pipeline_function_wrappers import (
    generate_and_reduce_patient_matrix,
)
from pipeline_blocks.core_pipeline_function_wrappers import generate_features
from utils.pickle_functions import pickle_already_exists, load_pickle
from utils.helpers import print_num_patients_with_SC

from pipeline_blocks.event_matching import print_patient_matrix

from baseline_model_functions import (
    remove_nan_values_from_dataset,
    print_class_balance,
    preprocess_fft,
)


def load_or_create_pickle(pickle_path, function_to_generate_pickle, **function_kwargs):
    if pickle_already_exists(pickle_path):
        data = load_pickle(pickle_path)
    else:
        start_time = time.time()
        function_to_generate_pickle(**function_kwargs)
        data = load_pickle(pickle_path)
        end_time = time.time()
        print(
            f"Running {function_to_generate_pickle.__name__} took {end_time - start_time} seconds"
        )

    return data


def cross_val_and_feature_importance(model, X, y, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    accuracies_train, accuracies_test = [], []
    macro_f1s_train, macro_f1s_test = [], []
    weighted_f1s_train, weighted_f1s_test = [], []

    for _, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  # type: ignore

        if "sklearn" in getattr(model, "__module__", None):  # type: ignore
            model.fit(X_train, y_train)
            train_outputs = model.predict(X_train)
            test_outputs = model.predict(X_test)
        else:
            continue

        accuracies_train.append(accuracy_score(y_train, train_outputs))
        accuracies_test.append(accuracy_score(y_test, test_outputs))
        weighted_f1s_train.append(f1_score(y_train, train_outputs, average="weighted"))
        weighted_f1s_test.append(f1_score(y_test, test_outputs, average="weighted"))
        macro_f1s_train.append(f1_score(y_train, train_outputs, average="macro"))
        macro_f1s_test.append(f1_score(y_test, test_outputs, average="macro"))

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=1,
            n_jobs=-1,
        )

        features = [round(x, 2) for x in np.linspace(0, 20, num_bins)]
        forest_importances = pd.Series(result.importances_mean, index=features)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_xlabel("Frequency")
        fig.tight_layout()
        plt.show()

    avg_train_acc = round(sum(accuracies_train) / len(accuracies_train), 4)
    avg_test_acc = round(sum(accuracies_test) / len(accuracies_test), 4)
    avg_train_weighted_f1 = round(sum(weighted_f1s_train) / len(weighted_f1s_train), 4)
    avg_test_weighted_f1 = round(sum(weighted_f1s_test) / len(weighted_f1s_test), 4)
    avg_train_macro_f1 = round(sum(macro_f1s_train) / len(macro_f1s_train), 4)
    avg_test_macro_f1 = round(sum(macro_f1s_test) / len(macro_f1s_test), 4)

    print(model.__class__.__name__)
    print(
        f"( train ) - acc: {avg_train_acc}, weighted-f1: {avg_train_weighted_f1}, macro-f1: {avg_train_macro_f1} \t",
        f"( test )  - acc: {avg_test_acc}, weighted-f1: {avg_test_weighted_f1}, macro-f1: {avg_test_macro_f1} \n",
    )

    metrics = {
        "avg_train_acc": avg_train_acc,
        "avg_test_acc": avg_test_acc,
        "avg_train_weighted_f1": avg_train_weighted_f1,
        "avg_test_weighted_f1": avg_test_weighted_f1,
        "avg_train_macro_f1": avg_train_macro_f1,
        "avg_test_macro_f1": avg_test_macro_f1,
    }

    return metrics


if __name__ == "__main__":

    root_path = os.getcwd()
    base_path = r"Z:\projects\vital_project\live\01NVa_Dengue\RAW_DATA"

    event_type = "severities"
    # event_type = "ICU-FU"

    if event_type == "severities":
        cohort_of_interest = "adults"
        dataset_path = os.path.join(base_path, "Adults")
    elif event_type == "ICU-FU":
        cohort_of_interest = "children"
        dataset_path = os.path.join(base_path, "Children")
    else:
        raise ValueError(
            "Invalid choice of event type. Pipeline currently thoroughly tested only for ['ICU-FU', 'severities']"
        )

    sampling_rate = 100

    # this refers to the length of the segments that we examine
    # when determining whether a segment is unusable or not
    # for reference: 525.6 seconds is the duration of STFT used
    unusable_segment_length_seconds = 52.56
    unusable_segment_length_samples = int(
        unusable_segment_length_seconds * sampling_rate
    )

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
    # I think I only used red and that I had done some checks??
    signal_wavelength = "RED"

    # doing interpretability so only care about frequency domain
    fe_method = "FFT"

    # this refers to whether we are using an STFT of the full possible length
    # (optimally 10 wavelengths of the lowest frequency of interest) or whether
    # we are using a smaller STFT to increase the number of training examples.
    # The STFT divider dictates how many times smaller the chosen STFT is then the "optimal" one
    window_divider = 1

    # specify type and parameters of filter for filtering the PPG signal
    filter_params = {
        "type": "cheby",
        "order": 2,
        "sample_rate_Hz": 100,
        "low_cutoff_Hz": 0.15,
        "high_cutoff_Hz": 20,
    }

    # specify which Signal Quality Indices (SQIs) are to be used for quality checking
    # as well as the thresholds that are acceptable for a signal for each of the SQIs
    sqi_thresholds = {"MSQ": 0.8, "zero_cross": [0.01, 0.04]}

    # define pickle names and locations for the different stages of the pipeline
    patient_info_pickle_name = "patient_info.pkl"
    unusable_data_pickle_name = f"unusable_data_{cohort_of_interest}_{signal_wavelength}_{unusable_segment_length_samples}_{num_rows_to_read}.pkl"
    patient_matrix_pickle_name = f"patient_matrix_{cohort_of_interest}_{signal_wavelength}_{event_type}_{window_size_samples}.pkl"
    training_data_pickle_name = f"feature_extraction_{cohort_of_interest}_{signal_wavelength}_{event_type}_{fe_method}_{window_divider}_{fe_segment_length_samples}.pkl"

    patient_info_pickle_path = os.path.join(
        root_path, "data_pickles", "patient_info", patient_info_pickle_name
    )
    unusable_data_pickle_path = os.path.join(
        root_path, "data_pickles", "unusable_data", unusable_data_pickle_name
    )
    patient_matrix_pickle_path = os.path.join(
        root_path, "data_pickles", "patient_matrix", patient_matrix_pickle_name
    )
    training_data_pickle_path = os.path.join(
        root_path,
        "data_pickles",
        "feature_extraction",
        fe_method,
        training_data_pickle_name,
    )

    # load Stefan's patient info pickle
    patient_info = load_pickle(patient_info_pickle_path)
    print_num_patients_with_SC(patient_info)

    # check if there already exists a pickle with the unusable data for the current parameters chosen
    # currently only accounts for cohort and the length of the segments to be examined for quality checking
    # if the pickle exists, load it, if not, generate it
    unusable_data = load_or_create_pickle(
        pickle_path=unusable_data_pickle_path,
        function_to_generate_pickle=generate_unusable_data_lists,
        base_path=dataset_path,
        unusable_segment_length_samples=unusable_segment_length_samples,
        filter_params=filter_params,
        sqi_thresholds=sqi_thresholds,
        unusable_data_pickle_path=unusable_data_pickle_path,
        signal_wavelength=signal_wavelength,
        num_rows_to_read=num_rows_to_read,
    )

    # check if there already exists a pickle with the patient matrix for the current parameters chosen
    # currently only accounts for cohort, event type, and the size of the window before and after the event
    # if the pickle exists, load it, if not, generate it
    reduced_patient_matrix = load_or_create_pickle(
        pickle_path=patient_matrix_pickle_path,
        function_to_generate_pickle=generate_and_reduce_patient_matrix,
        base_path=dataset_path,
        patient_info=patient_info,
        event_type=event_type,
        window_size_samples=window_size_samples,
        unusable_data=unusable_data,
        unusable_segment_length_samples=unusable_segment_length_samples,
        patient_matrix_pickle_path=patient_matrix_pickle_path,
    )
    print("Reduced patient matrix:")
    print_patient_matrix(reduced_patient_matrix, event_type)
    print("\n")

    # check if there already exists a pickle with the training data/features for the current parameters chosen
    # currently only accounts for cohort, window size, and the size of segments we cut up the signal for FE
    # if the pickle exists, load it, if not, generate it
    feature_extraction_data = load_or_create_pickle(
        pickle_path=training_data_pickle_path,
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
        training_data_pickle_path=training_data_pickle_path,
    )

    models_to_test = [
        # DecisionTreeClassifier(random_state=1),
        RandomForestClassifier(n_estimators=100, random_state=1),
        # SVC(kernel="rbf", random_state=1),
        # MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), random_state=1),
    ]

    for model in models_to_test:
        # number of frequency bins
        num_bins = 64

        # is this really needed?
        gc.collect()

        # get I/O for model
        inputs, outputs = feature_extraction_data
        label_encoder = LabelEncoder()
        integer_outputs = label_encoder.fit_transform(outputs)

        # the representation is 0-50Hz but we care only about 0-20Hz
        # since we filter above 20Hz, so we truncate the signal at 2/5
        truncating_point = int(0.4 * len(inputs[0]))

        # truncate the FFT and split it into the appropriate number of frequency bins
        inputs = [preprocess_fft(fft, num_bins, truncating_point) for fft in inputs]
        inputs = np.asarray(inputs)

        print(f"\n\nNumber of bins: {num_bins}")
        print(f"Shape of input data: {inputs.shape}")
        print(f"Shape of output data: {integer_outputs.shape}")

        # remove NaNs
        if event_type == "severities":
            inputs_contain_nans = np.isnan(inputs).any()
            if inputs_contain_nans:
                print("Input data contains nan values, removing now...")
                inputs, integer_outputs = remove_nan_values_from_dataset(
                    inputs, integer_outputs
                )
            else:
                print("Input data does not contain nan values, all good!")
            print_class_balance(inputs, integer_outputs)

        # cross validate and get feature importance
        metrics = cross_val_and_feature_importance(
            model, inputs, integer_outputs, num_folds=5
        )