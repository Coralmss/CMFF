# CMFF
The official implementation of "Self-Supervised Learning and Cross-Modal Feature Fusion for Online Knowledge Distillation" (TMM).  <br>

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
<code>nohup python train_vgg.py --lr 0.1  --epoch 400  --cu_num 0  </code>

<div style="position: relative; padding: 10px; background-color: #f6f8fa; border-radius: 6px;">
  <pre style="margin: 0; padding: 10px; background-color: #f6f8fa; border: 1px solid #d1d5da; border-radius: 6px;">
    <code>python train_baseline_cifar.py --arch wrn_16_2 --data ./data/ --gpu 0</code>
  </pre>
  <button onclick="copyCode()" style="position: absolute; right: 10px; top: 10px; background: none; border: none; cursor: pointer;">
    📋
  </button>
</div>

<script>
  function copyCode() {
    const code = document.querySelector('pre code').innerText;
    navigator.clipboard.writeText(code).then(() => {
      alert('Code copied to clipboard!');
    });
  }
</script>

    

### Results



## Citation


## Acknowledgements
