# Hourglass
my own implementation of paper [Hourglass: Enabling Efficient Split Federated Learning with Data Parallelism](https://dl.acm.org/doi/pdf/10.1145/3689031.3717467)
## start
```python
python main.py --model resnet50 --dataset cifar10 --M_GPU 3 --local_lr 0.001
```
## recommend hyperparameters
|model|optimizer|lr|
|-----|---------|--|
|ResNet50|SGD|0.001|
|Vgg16|SGD|0.001|
|ViT|SGD|0.001|
|CharCNN|Adam|0.0001|
## options
```
options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of rounds of training
  --num_users NUM_USERS
                        number of users: K
  --frac FRAC           the fraction of clients: C
  --local_ep LOCAL_EP   the number of local epochs: E
  --local_bs LOCAL_BS   local batch size: B
  --local_lr LOCAL_LR   learning rate
  --local_mtm LOCAL_MTM
                        SGD momentum (default: 0.5)
  --optimizer OPTIMIZER
                        the optimizer for model
  --dataset {cifar10,cinic10}
                        name of dataset
  --num_classes NUM_CLASSES
                        number of classes
  --model MODEL         model to use
  --device DEVICE       To use cuda, set to a specific GPU ID. Default set to use CPU.
  --verbose VERBOSE     verbose
  --seed SEED           random seed
  --strategy {FCFS,SFF,DFF}
                        feature scheduleing strategys
  --n_clusters N_CLUSTERS
                        clusters for kmeans
  --split_method SPLIT_METHOD
                        method used to ensure data heterogeneity
  --M_GPU M_GPU         the number of gpu for server to train model
```
## result
to be continued...