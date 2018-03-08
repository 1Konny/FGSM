# FGSM(Fast Gradient Sign Method)
<br>

### Overview 
Simple pytorch implementation of FGSM([from this paper])
![Figure0](misc/fig0.PNG)
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
tensorboardX(optional)
tensorflow(optional)
```
<br>

### Usage
1. start the visdom server
```
python -m visdom.server --port [PORT]
```
2. train a simple MNIST classifier
```
python main.py --mode train --port [PORT] --env_name my_model
```
3. load trained classifier, generate adversarial examples, and then see result on the visdom server
```
python main.py --mode generate --port [PORT] --iteration 1 --epsilon 0.03 --env_name my_model --load_ckpt best_acc.tar
```
<br>

### Result
from the left, legitimate examples, perturbed examples, and indication of perturbed images that changed predictions of the classifier

1. iteration : 1, epsilon : 0.03
![Figure1](misc/fig1.PNG)
2. iteration : 5, epsilon : 0.03
![Figure2](misc/fig2.PNG)
1. iteration : 1, epsilon : 0.5
![Figure3](misc/fig3.PNG)
<br>

### To Do
- [ ] add cifar10
- [ ] remove visdom dependency
- [ ] add targeted/untargeted generation mode

[from this paper]: https://arxiv.org/abs/1412.6572
