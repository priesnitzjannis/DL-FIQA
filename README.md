# Introduction
This repository contains the training and testing framework and our proposed models for deeplearning-based fingerprint quality assessment. In our corresponding paper, we modify and fine-tune several CNN and a ViT-architectures for the special purpose. We train our model exclusively on synthetic data and test them on seven publicly available data sets. As baseline serves the established NFIQ 2.3 method. For benchmarking the obtained results, we use the error vs. discard characteristic. Our research shows that all proposed methods outperform NFIQ 2.3 (cf. Table below) wheras the best performing algorithm improves the predictive performance by 12.29%. 

|                   |   FVC2006   |             |             |   ISPFDv1   |     MOLF    |             |             |   average   |   std dev   | improvement |
|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|                   |     DB2     |     DB3     |     DB4     |     DB3     |     DB1     |     DB2     |     DB3     |             |             |             |
|       VGG16       | 0.41651     | 0.66411     | 0.62767     | 0.64358     | 0.72737     | 0.58021     | 0.34653     | **0.57228** | 0.12869     | **12.29%**  |
| InceptionResNetV2 | 0.67326     | 0.65053     | 0.60654     | **0.52916** | 0.64095     | **0.57988** | 0.36256     | 0.57755     | 0.09850     | 11.48%      |
|        ViT        | **0.51992** | 0.70912     | **0.57214** | 0.63133     | **0.60018** | 0.60021     | 0.46494     | 0.58540     | **0.07253** | 10.28%      |
|       VGG19       | 0.53421     | 0.67595     | 0.63883     | 0.71227     | 0.73806     | 0.58003     | **0.33486** | 0.60203     | 0.12758     | 7.73%       |
|    DenseNet121    | 0.60318     | 0.74439     | 0.61952     | 0.66611     | 0.62091     | 0.59923     | 0.44784     | 0.61445     | 0.08258     | 5.83%       |
|    MobileNetV2    | 0.54475     | **0.63594** | 0.64210     | 0.80837     | 0.69078     | 0.59402     | 0.41606     | 0.61886     | 0.11286     | 5.15%       |
|     MobileNet     | 0.53174     | 0.69096     | 0.62274     | 0.79922     | 0.70490     | 0.61226     | 0.40391     | 0.62367     | 0.11882     | 4.41%       |
|      NFIQ 2.3     | 0.57893     | 0.74350     | 0.64997     | 0.80939     | 0.76445     | 0.58110     | 0.43994     | 0.65247     | 0.12019     | 0.00%       |
|      ResNet50     | 0.66298     | 0.74678     | 0.64189     | 0.64115     | 0.70543     | 0.70028     | 0.79925     | 0.69968     | 0.05382     |             |


We invite interested researchers to use the provided framework for testing on further database or train own models. 

## Citing
If you test our models on your own data or find this framework useful for developing your own, please cite:

TBC

# Set up
## Requirements
### Hardware
The models were trained on a NVIDIA Quadro RTX 8000 with 48GB RAM. 


### Software
Conda environments are used to manage dependencies and ensure reproducibility. 
The code was tested on Conda version 25.7. 


## Installation
It is required to set up two conda environments via the provided yml files. 
To set up the conda environment, use the follow these commands:

### CNNs: 
```bash
conda env create -f cnn_env.yml
```

### ViT: 
```bash
conda env create -f vit_env.yml
```
# Working with the code 
We provide a framework for testing our proposed models on own data set and training own models. 

## Testing
For testing our models on your data you only need to prepare a .csv file with a list of abolsute or relative filepaths. 
First activate the corresponding environment. Then call the CNN.py resp. ViT.py script as presented below. 

```bash
conda activate CNN-FIQA
python CNN.py --mode test --model_file_path "pretrained_models/MobileNetV2.h5" --labels_file_path test_data.csv
```

```bash
conda activate DL-FIQA
python ViT.py --mode test --model_name pretrained_models/ViT_small_16x16.pth --csv_file test_data.csv

```


## Training
For traning your own  models you need to prepare a .csv file with a list of abolsute or relative filepaths and a training label in the range between 0 and 100 denoted as integer. The values are separated by a comma as shown in training_data.csv. 
First activate the corresponding environment. Then call the CNN.py resp. ViT.py script as presented below. 

```bash
conda activate CNN-FIQA python CNN.py --mode train --model_name my_model --model_architecture ResNet50 --labels_file_path training_data.csv --num_epochs 10 --learning_rate 1e-5 --activation_function linear
```

```bash
conda activate DL-FIQA
python ViT.py --mode train --model_name my_model --csv_file "training_data.csv" --test_size 0.2 --epochs 2 --learning_rate 1e-5

```

