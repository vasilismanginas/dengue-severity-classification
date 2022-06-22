# Final Year Project

## How to run:

### Requirements:
- **If you use conda:** Running the command below will create a new conda environment with the name `env_name` which replicates the environment specified in the `conda_environment.yml` file.

    ```
    conda env create --name <env_name> -f conda_environment.yml
    ```

- **Alternatively, if you don't use conda, and use pip:** Running the command below will install all packages specified in the `requirements_pip.txt` file.

    ```
    pip install -r requirements_pip.txt
    ```

    ***WARNING:*** The pip version might potentially not include all dependencies but is sufficient for creating a functional environment, after which you can use `pip install <package_name>` for any package that might be missing.

### Running the code:
Run the `main_controller.py` file. This includes the definition of important parameters for all pipeline stages and subsequently runs each pipeline stage sequentially.

***IMPORTANT:*** You have to change the path at which the dataset is located on your machine. This should be stored in the `base_path` variable created in the beginning of the main of the file.

For more information on the repository structure, and how the pipeline and pickles operate, please visit the [Wiki](https://github.com/vasilismanginas/dengue-severity-classification/wiki).
