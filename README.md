# ML_basic_example_DB
Welcome to the `ML_basic_example_DB` project. The goal of this project is to make the deployment of ML model for classification task. The project has been set up to handle each aspect of the ML pipeline from data processing, training models, and inferencing on new data.

## Forking and Cloning from GitHub
To start using this project, you first need to 'clone' repository to your local machine. To do this, click the 'Code' button on repository, copy the provided link, and use the `git clone` command in your terminal followed by the copied link. This will create a local copy of the repository on your machine, and you're ready to start!

### Setting Up Development Environment
Next, you need to set up a suitable Integrated Development Environment (IDE). For running scripts, open a new terminal in IDE by selecting `Terminal -> New Terminal`. Now you can execute Python scripts directly in the terminal.

### Installing MLFlow on Windows

MLFlow can be easily installed on a Windows local machine using the pip, the Python package installer. To do so, open the command prompt (you can find it by searching for `cmd` in the Start menu) and type the following command:

```python
pip install mlflow
```

After the successful installation, you can start managing and deploying your ML models with MLFlow. For further information on how to use MLFlow at its best, refer to the official MLFlow documentation or use the `mlflow --help` command.

Should you encounter any issues during the installation, you can bypass them by commenting out the corresponding lines in the `train.py` and `requirements.txt` files.

To run MLFlow, type `mlflow ui` in your terminal and press enter. If it doesn't work, you may also try `python -m mlflow ui`  This will start the MLFlow tracking UI, typically running on your localhost at port 5000. You can then access the tracking UI by opening your web browser and navigating to `http://localhost:5000`.

Besides MLFLow, in project is used Docker software.


## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example_DB
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── inference_data.csv
│   └── train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts.

## Data:
Use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Data used in this project is Iris Dataset tat is generated automaticly from sklearn python library.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.
