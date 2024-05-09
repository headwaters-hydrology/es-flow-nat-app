# Flow Naturalisation web app for ES

## Introduction
This repo contains the Flow Naturalisation web app for Environment Southland. It has been re-worked to run locally using a conda python environment instead of relying on external services.
The app is built using Plotly and Dash in Python. 

## Running the app
### Batch files and conda - Windows
First, open up the command line (or powershell) within this folder.
Next, install miniconda via the following commands in the command line (or install it yourself in the PC manually):
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```

Next, install the conda environment either via the install_env.bat file (using the command line) or simply running the following command:
```
conda env create -f env.yml
```

Then, you can run the web app using the main.bat file or via the command line by activating the conda environment then running the \app\main.py script:
```
cd app
conda activate flow-nat-app
python main.py
```


