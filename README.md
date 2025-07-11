# Hourglass
my own implementation of paper [Hourglass: Enabling Efficient Split Federated Learning with Data Parallelism](https://dl.acm.org/doi/pdf/10.1145/3689031.3717467)
## environment
```
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
numpy==2.2.6
einops==0.8.1
scikit-learn==1.7.0
```
## start
```
python Hourglass/main.py --model resnet50 --dataset cifar10 --M_GPU 3 --local_lr 0.001
```
## recommend hyperparameters
|model|optimizer|lr|
|-----|---------|--|
|ResNet50|SGD|0.001|
|VGG16|SGD|0.001|
|ViT|SGD|0.001|
|CharCNN|Adam|0.0001|
|LSTM|Adam|0.0001|
## dataset details
|dataset|type|num_classes|orgin_shape|input_shape|
|-------|----|-----------|-----------|-----------|
|CIFAR-10|image|10|(3,32,32)|(3,224,224)|
|CINIC-10|image|10|(3,32,32)|(3,224,224)|
|AG News|text|4|-------|(70,1014)|
|SpeechCommands|audio|35|(1,4000~20000)|(1,8000)|
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
### overall performance of Hourglass
As clients and servers are parallel,we can approximate the end_to_end time by calculating the time spent on  every stage
$$ time_{kmeans} = max(time_{ClientForward})+time_{cluster}+max(time_{Server})+max(time_{ClientBackward})\\
time_{LSH} = min(time_{ClientForward})+min(time_{Server})+max(time_{ClientBackward})$$
|dataset|model|target_acc|FCFS|SFF|DFF|
|-------|-----|----------|----|---|---|
|CIFAR-10|VGG16|------|-----|----|----|
|CIFAR-10|ResNet50|---|-----|----|----|
|CIFAR-10|ViT|--------|-----|----|----|
|CINIC-10|VGG16|------|-----|----|----|
|CINIC-10|ResNet50|------|-----|----|----|
|CINIC-10|ViT|------|-----|----|----|
|AG News|CharCNN|-----|-----|----|----|
|AG News|LSTM|------|-----|----|----|
|Speech Commands|VGG16|---|----|----|----|

### detailed performance
To check details of accuracy and loss, run the command below and open [localhost:6006](localhost:6006)
```
tensorboard --logdir ./save --port 6006
```