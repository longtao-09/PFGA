# Generalizing Personalized Federated Graph Augmentation via Min-max Adversarial Learning

The code of this project is modified based on FedPUB. Please refer to the original code: https://github.com/JinheonBaek/FED-PUB

## Requirement
- Python 3.9.16
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0
- METIS (for data generation), https://github.com/james77777778/metis_python

## Data Generation
Following command lines automatically generate the dataset.
```sh
$ cd data/generators
$ python disjoint.py
$ python overlapping.py
```

## Run on disjoint
Following command lines run the experiments for FedAvg  on Cora dataset.
```sh
python main.py --gpu 0 --n-workers 1 --model fedavg --dataset Cora --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 5 --seed 1024 --acg_model 0 --print 1 
```
Following command lines run the experiments for PFGA-A on Cora dataset.
```sh
python main.py --gpu 0 --n-workers 1 --model fedpfgaa --dataset Cora --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 5 --seed 1024 --acg_model 0 --print 1 
```
The following command line runs a parameter search experiment for PFGA-A on the Cora dataset.
```sh
python optuna_main.py --gpu 0 --n-workers 1 --model fedpfgaa --dataset Cora --mode disjoint --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 5 --seed 1024 --acg_model 0 --print 0
```


## Run on overlapping
Following command lines run the experiments for FedAvg  on Cora dataset.
```sh
python main.py --gpu 0 --n-workers 1 --model fedavg --dataset Cora --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 10 --seed 1024 --acg_model 0 --print 1 
```
Following command lines run the experiments for PFGA-A on Cora dataset.
```sh
python main.py --gpu 0 --n-workers 1 --model fedpfgaa --dataset Cora --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 10 --seed 1024 --acg_model 0 --print 1 
```
The following command line runs a parameter search experiment for PFGA-A on the Cora dataset.
```sh
python optuna_main.py --gpu 0 --n-workers 1 --model fedpfgaa --dataset Cora --mode overlapping --frac 1.0 --n-rnds 200 --n-eps 1 --n-clients 10 --seed 1024 --acg_model 0 --print 0 
```
