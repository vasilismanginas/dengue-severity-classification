import os
import gc
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support as get_metrics

from pipeline_blocks.core_pipeline_function_wrappers import generate_unusable_data_lists
from pipeline_blocks.core_pipeline_function_wrappers import generate_and_reduce_patient_matrix
from pipeline_blocks.core_pipeline_function_wrappers import generate_features
from utils.pickle_functions import pickle_already_exists, load_pickle
from utils.helpers import print_num_patients_with_SC

from pipeline_blocks.event_matching import print_patient_matrix
from pipeline_blocks.model_training import get_train_and_val_data
from pipeline_blocks.model_training import create_and_compile_model, get_binary_class_occurrences, train_and_evaluate_model, plot_model_histories, preprocess_stft, remove_nan_values_from_dataset, print_class_balance



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
    


    print('* Preparing datasets for model training...')
    inputs, outputs = feature_extraction_data

    if fe_method == 'STFT':
        num_bins = 128
        truncating_point = int(0.4 * len(inputs[0]))
        num_ffts_in_stft = inputs[0].shape[1]
        start_time = time.time()
        inputs = [preprocess_stft(stft, num_bins, truncating_point, num_ffts_in_stft) for stft in inputs]
        end_time = time.time()
        print(f'Preprocessing STFT inputs took {end_time - start_time} seconds')
        inputs = np.asarray(inputs)

    else:
        raise ValueError("Invalid FE method chosen, CNNs are used only for STFT")


    if event_type == 'severities':
        inputs_contain_nans = np.isnan(inputs).any()
        if inputs_contain_nans:
            print('Input data contains nan values, removing now...')
            inputs, outputs = remove_nan_values_from_dataset(inputs, outputs)
        else:
            print('Input data does not contain nan values, all good!')
        print_class_balance(inputs, outputs)


    learning_rate = 5e-4
    epochs = 200
    batch_size = 32
    num_folds = 5

    # create instance of KFold class for K-Fold cross validation and define a counter for the fold
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # create instance of EarlyStopping callback for training the model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    #start timer to time the training process
    start_time = time.time()
    model_histories = []
    cross_val_losses = []
    cross_val_accuracies = []
    cross_val_precisions = []
    cross_val_recalls = []
    cross_val_macro_f1s = []
    cross_val_weighted_f1s = []
    fold_counter = 1
    gc.collect()


    # split inputs and output using K-Fold cross validation and
    # train the model using the resulting training and validation sets
    for train_index, validation_index in k_fold.split(inputs, outputs):

        X_train, y_train, X_val, y_val = get_train_and_val_data(inputs,
                                                                outputs,
                                                                num_of_classes,
                                                                train_index, 
                                                                validation_index)

        model = create_and_compile_model(input_shape=X_train[0].shape,
                                        output_dim=num_of_classes,
                                        learning_rate=learning_rate)

        if fold_counter == 1:
            print('* Creating and compiling model...')
            model.summary()
            print('* Done creating and compiling model! \n')
            print(f'X_train shape: {X_train.shape}')
            print(f'y_train shape: {y_train.shape}')
            print(f'X_val shape: {X_val.shape}')
            print(f'y_val shape: {y_val.shape} \n')

        # get_binary_class_occurrences(y_train, y_val)

        print(f'* Training and evaluating model over fold {fold_counter}...')
        history, fold_val_loss, fold_val_acc = train_and_evaluate_model(model,
                                                    X_train, y_train,
                                                    X_val, y_val,
                                                    batch_size,
                                                    epochs,
                                                    [early_stop],
                                                    evaluate_on_test_set=True,
                                                    x_test=X_val, y_test=y_val)

        y_pred = model.predict(X_val, verbose=0)
        y_pred_bool = np.argmax(y_pred, axis=1)
        y_true = y_val
        precision, recall, macro_f1_score, _ = get_metrics(y_true, y_pred_bool, average='macro')
        _, _, weighted_f1_score, _ = get_metrics(y_true, y_pred_bool, average='weighted')
        print(f'\nPerformance metrics for fold {fold_counter}')
        print(f'\taccuracy: {fold_val_acc}')
        # print(f'\tloss: {fold_val_loss}')
        print(f'\tprecision: {precision}')
        print(f'\trecall: {recall}')
        print(f'\tmacro_f1_score: {macro_f1_score}')
        print(f'\tweighted_f1_score: {weighted_f1_score}')

        # print(classification_report(y_true, y_pred_bool))

        cross_val_losses.append(fold_val_loss)
        cross_val_accuracies.append(fold_val_acc)
        cross_val_precisions.append(precision)
        cross_val_recalls.append(recall)
        cross_val_macro_f1s.append(macro_f1_score)
        cross_val_weighted_f1s.append(weighted_f1_score)

        model_histories.append(history)
        fold_counter = fold_counter + 1
        print('* Done training and evaluating model! \n')

        gc.collect()


    end_time = time.time()
    print(f'Model training took {end_time - start_time} seconds')

    average_cross_val_loss = sum(cross_val_losses) / len(cross_val_losses)
    average_cross_val_acc = sum(cross_val_accuracies) / len(cross_val_accuracies)
    average_cross_val_precision = sum(cross_val_precisions) / len(cross_val_precisions)
    average_cross_val_recall = sum(cross_val_recalls) / len(cross_val_recalls)
    average_cross_val_macro_f1 = sum(cross_val_macro_f1s) / len(cross_val_macro_f1s)
    average_cross_val_weighted_f1 = sum(cross_val_weighted_f1s) / len(cross_val_weighted_f1s)

    # print(f'\n\nAverage validation loss over {num_folds} folds: {average_cross_val_loss}')
    print(f'\n\nAverage validation accuracy over {num_folds} folds: {average_cross_val_acc}')
    print(f'Average validation precision over {num_folds} folds: {average_cross_val_precision}')
    print(f'Average validation recall over {num_folds} folds: {average_cross_val_recall}')
    print(f'Average validation macro-F1 score over {num_folds} folds: {average_cross_val_macro_f1}')
    print(f'Average validation weighted-F1 score over {num_folds} folds: {average_cross_val_weighted_f1}')

    plot_model_histories(model_histories)