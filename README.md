# Flow Naturalisation web app for ES

## Introduction
This repo contains the Flow Naturalisation web app for Environment Southland. It has been re-worked to run locally using a conda python environment instead of relying on external services.
The app is built using Plotly and Dash in Python. 

## Running the app
### Batch files and conda - Windows
First, open up the command line (or powershell) within this folder.
Next, install miniconda.
You can download it via the command line (or download it manually with the link):
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
```
Then execute the miniconda.exe file. Select the option to install it for your user only. In addition to the already selected installation options, make sure to select the option to register miniconda in the PATH variables. This will make it possible to run conda commands.

Next, install the conda environment either via the **install_env.bat** file (double click the bat file) or by running the following command:
```
conda env create -f env.yml
```

Then, you can run the web app using the **main.bat** file or via the command line by activating the conda environment then running the \app\main.py script:
```
cd app
conda activate habitat-app
python main.py
```


