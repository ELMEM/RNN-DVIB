# AAAI2026 Project

## Datasets
We evaluated the ConvTran model using a combination of 30 datasets from the UEA archive and two additional datasets, Actitracker HAR and Ford. To obtain these datasets, you have two options:
### Manual download:
You can manually download the datasets using the provided link and place them into the pre-made directory.

UEA: https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip
Copy the datasets folder to: Datasets/UEA/


## Dataset Setup

To use this repository, please place the thesis dataset in the `Dataset` folder. 

### Instructions:
1. Create a folder named `Dataset` in the project root directory if it doesn't exist
2. Place all dataset files in this folder
3. Ensure the dataset structure matches the expected format in the code

The project code will automatically look for the dataset in this location.

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run

To see all command options with explanations, run: `python main.py --help`
In `main.py` you can select the datasets and modify the model parameters.
For example:

`self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')`

or you can set the paprameters:

`python main.py --epochs 1500 --data_dir Datasets/UEA/`
Use the `Net_Type` parameter to set the model type, where:  

- `'O'` represents our proposed method  
- `'C'` represents ConvTran  
- `'F'` represents LSTM-FCN  
- `'E'` represents MLSTM-FCN
- To reproduce the **Log-Neural-CDE** results, you can use the following GitHub repository:  

🔗 [Benjamin-Walker/Log-Neural-CDEs](https://github.com/Benjamin-Walker/Log-Neural-CDEs)  

### Usage Notes:  
- The repository contains the official implementation of **Log-NCDE**.  
- Follow the provided instructions to replicate the experimental results.  

