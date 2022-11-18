# Model Development for Severity Classification of Dengue Patients using Photoplethysmography (PPG) Data

## Project Aim:
The broader scope of this project is to assist in the management of dengue patients. This work aims to be one of the primary steps in showing that providing such assistance via the use of wearable PPG sensors is possible. This is achieved by investigating whether a patient's PPG signal can be used as an indicator of the severity of their dengue infection.

In different pages of the [Wiki](https://github.com/vasilismanginas/dengue-severity-classification/wiki) you can find more information on:
* [Repository structure](https://github.com/vasilismanginas/dengue-severity-classification/wiki/Repository-structure)
* [General pipeline operation, execution flow, and pipeline blocks](https://github.com/vasilismanginas/dengue-severity-classification/wiki/General-execution-flow-and-pipeline-blocks)
* [Use of pickles as checkpoints](https://github.com/vasilismanginas/dengue-severity-classification/wiki/Use-of-pickles-as-checkpoints)


## How to Run:

### Requirements:
See the [Wiki page on Requirements](https://github.com/vasilismanginas/dengue-severity-classification/wiki/Requirements).

### Running the code:
Run the `main_controller.py` file. This includes the definition of important parameters for all pipeline stages and subsequently runs each pipeline stage sequentially.

***IMPORTANT:*** You have to change the path at which the dataset is located on your machine. This should be stored in the `base_path` variable created in the beginning of the main of the file.
