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
| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |

### Top-1 Acc on TinyImageNet
| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |


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
