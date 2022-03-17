# HiCLSTM
Hi-C-LSTM is a framework to build Hi-C intrachromosomal representations which are useful for element identification and in-silico alterations. 

<p align="center">
    <a href="https://www.biorxiv.org/content/10.1101/2021.08.26.457856v1.abstract">
        <img alt="Paper" width="80" height="40" src="https://github.com/kevinbdsouza/Hi-C-LSTM//blob/main/data2/penguin.svg?raw=true">
    </a>
    <a href="https://github.com/kevinbdsouza/Hi-C-LSTM/releases">
        <img alt="Release" width="80" height="40" src="https://github.com/kevinbdsouza/Hi-C-LSTM//blob/main/data2/crow.svg?raw=true">
    </a>
</p>

## Hi-C-LSTM Model 
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/Hi-C-LSTM//blob/main/data2/HiC_model.png?raw=true">
</p>

## Requirements 
The following software was installed on Ubuntu 16.04
* Python 3.7.10
* CUDA 10.1 with libcudnn.so.7
* torch 1.8.0
* captum 0.3.1
* numpy 1.21.0
* pandas 1.2.4
* scipy 1.7.0
* matplotlib 3.4.2
* tensorboard 2.5.0
* seaborn 0.11.1

Install the above dependencies using installers like pip. The typical install time is about 1 hour. No non-standard hardware is required. 

## To prepare data:
1. get HiC data: run ./download_data.sh then ./hic2txt.sh
2. run compute_genome_length.py to create file with rounded, cumulative chromosome lengths

## Demo 
1. Use the ```hic_chr22.txt``` file as input for demo
2. Partition the file based on training and testing needs
3. Model parameters, hyperarameters, and output directories can be changed in ```./code/config.py```.

### Train model:
```./code/train_model.py```

1. Specify the ```model_name``` of your choice
2. In the ```DataLoader```, under the ```get_data_loader(cfg, cell)``` function, specify the chromosomes to be used
3. For the demo case, use chromosome 22. Change the directory of input Hi-C data to ```.data2/```
4. Expected output is a trained model called ```model_name```. Expected training time is less than 8 minutes per epoch on GeForce GTX 1080 Ti GPU. 


### Test model:
```./code/test_model.py```

1. Use the trained model ```model_name``` to test on the remainder of the chromosome 22 data
2. Expect the MSE of the model as the output along with predictions and representations. Expected testing time is less than 5 minutes. 

## How to use Representations

1. Extract the representations of size ```representation_size``` from the prediction file
2. Align them with the genome at 10Kbp resolution
3. Use for downstream tasks of preference like classification of genomic phenomena and in-silico mutagenesis.  