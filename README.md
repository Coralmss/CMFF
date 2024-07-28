# CMFF
The official implementation of "Self-Supervised Learning and Cross-Modal Feature Fusion for Online Knowledge Distillation".  <br>

![image](https://github.com/Coralmss/CMFF/blob/main/method.png) <br>

## Requirements
Ubuntu 20.04.5 LTS  <br>
Python 3.8.10 (Anaconda is recommended)  <br>
CUDA 10.1  <br>
PyTorch 1.8.0  <br>


## Quick start
### Dataset
CIFAR-100  <br>
TinyImageNet <br>
unzip to the <code>./data</code> folder<br>
### Running
#### Example
<div style="position: relative; padding: 5px; background-color: #f6f8fa; border-radius: 3px;">
  <pre style="margin: 0; padding: 5px; background-color: #f6f8fa; border: 1px solid #d1d5da; border-radius: 3px;">
    <code>nohup python train_vgg.py --lr 0.1  --epoch 400  --cu_num 0 </code>
  </pre>
  <button onclick="copyCode()" style="position: absolute; right: 5px; top: 5px; background: none; border: none; cursor: pointer;">
  </button>
</div>


    
## Results 
### Top-1 Acc on CIFAR100

| Method   | Resnet-32 | Resnet-110 | VGG-16 |WRN-20-8 |DenseNet-40-12 |WRN-16-2|WRN-40-2|  
|:--- |:---|:---|:---|:---|:---|:---|:---|
|      Method         | Resnet-32        | Resnet-110      | VGG-16           | WRN-20-8        | DenseNet-40-12   | Resnet-32       | Resnet-56        |
| Student1           | 70.02            | 76.49           | 74.50            | 78.23           | 71.02            | 72.56           | 74.49            |
| Student2           | 70.02            | 76.49           | 74.50            | 78.23           | 71.02            | 72.56           | 74.49            |
| DML          | 73.59            | 77.66           | 75.07            | 79.76           | 73.05            | 71.69           | 73.25            |
| ONE           | 73.64            | 79.73           | -                | 81.27           | 71.27            | 71.53           | 72.03            |
| AMLN           | -                | 74.69           | 79.87            | -               | -                | 74.65           | 75.29            |
| FFL         | 74.44            | 80.66           | 73.56            | 80.94           | 71.31            | 72.94           | 73.77            |
| OKDDip             | 74.60            | 80.88           | 74.20            | 81.32           | 72.01            | 75.24           | 75.01            |
| PCL             | 74.14            | 80.39           | 73.16            | 80.07           | 70.89            | 74.51           | 75.08            |
| KDCL              | 73.76            | -               | -                | 81.04           | -                | 74.82           | 75.43            |
| DCCL       | 75.38            | 80.74           | -                | 73.98           | 71.27            | 74.87           | 76.31            |
| CMFF(Ours)  | 75.72(+0.34)     | 80.54(+0.45)    | 79.36(+2.1)      | 81.83(+1.09)    | 74.52(+0.54)     | 75.22(+0.35)    | 77.09(+0.78)     |


### Top-1 Acc on TinyImageNet



|    Method      | Resnet-18        | Resnet-34       | 
|:--- |:---|:---|
|    Method      | Resnet-18        | Resnet-34       | 
| Student1  | 65.30            | 67.45           | 
| Student2  | 65.30            | 67.45           | 
| DML | 67.32            | 69.14           | 
| ONE | -                | -               | 
| AMLN | -               | -               |        
| FFL | 68.89            | 69.69           | 
| OKDDip | -             | -               | 
| PCL | -                | -               | 
| KDCL | -               | -               | 
| DCCL | 69.59           | 69.92           | 
| CMFF(Ours)| 74.99(+5.4)     | 81.11(+11.19)   | 


## Citation
<div style="position: relative; padding: 5px; background-color: #f6f8fa; border-radius: 3px;">
  <pre style="margin: 0; padding: 5px; background-color: #f6f8fa; border: 1px solid #d1d5da; border-radius: 3px;">
    <code>@xxxx{xx}</code>
  </pre>
  <button onclick="copyCode()" style="position: absolute; right: 5px; top: 5px; background: none; border: none; cursor: pointer;">
  </button>
</div>


## Acknowledgements
The implementation of models is borrowed from DCCL
