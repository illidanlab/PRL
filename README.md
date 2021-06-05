# Learning Deep Neural Networks under Agnostic Corrupted Supervision
This is the official implementation of the paper: Learning Deep Neural Networks under Agnostic Corrupted Supervision

<br>

## Abstract
Training deep neural models in the presence of corrupted supervision is challenging as the corrupted data points may significantly impact the generalization performance.
To alleviate this problem, we present an efficient robust algorithm that achieves strong guarantees without any assumption on the type of corruption and provides a unified framework for both classification and regression problems. Unlike many existing approaches that quantify the quality of the data points (e.g., based on their individual loss values), and filter them accordingly, 
the proposed algorithm focuses on controlling the collective impact of data points on the average gradient. 
Even when a corrupted data point failed to be excluded by our algorithm, the data point will have very limited impact on the overall loss, as compared with state-of-the-art filtering methods 
based on loss values. Extensive experiments on multiple benchmark datasets have demonstrated the robustness of our algorithm under different types of corruptions.


<br>

## Usage

It should be very easy to use our algorithm in your training pipeline if you think your training data contains corrupted supervision. 
Just for every update, first, calculate the loss-layer gradient norm and throw data with large loss gradient norm, only using the remaining data to perform gradient descent.
We provide a classification example with cross-entropy loss below: 

```python
from model import YourModel
import torch.nn.functional as F
...
epsilon=0.3  # your estimated noisy label ratio
batch_size = 128
model = YourModel()
base_optimizer = torch.optim.Adam  # can be changed to any optimizer
optimizer = base_optimizer(model.parameters(), lr=3e-4)
...

for input, target in data:

  pred = YourModel(input)
  loss = F.cross_entropy(pred, target, reduction='none')  # loss to minimize
  filtering_score = F.mse_loss(F.softmax(pred, dim=1), F.one_hot(target, num_classes=10), reduction='none') # get individual loss-layer gradient norm. The loss layer gradient norm for cross entropy is the MSE loss.
  with torch.no_grad():
    _, index = torch.sort(filtering_score)
    selected_index = index[:int(batch_size * (1-epsilon))]  # you can also use dynamic filtering ratio. For example, you can linearly increase epsilon from 0 to your estimated noisy label ratio in first 10 epochs.
  loss =  loss[index].mean()  # only use loss with small loss layer gradient norm
  loss.backward()
  optimizer.step()
...
```

Usually, the above code will bring robustness against supervision corruption. However, if it fails, you may want to consider the below tricks if you find our algorithm failed to defense the supervision corruptions.
> try to overestimate the corruption ratio and increase your batch size. 
>
> When using our code for regression, if your supervision value has an extremely long tail, try to remove the tail part first.

Also, you can add components such as mix-up (add mix-up on selected samples in each minibatch) to our algorithm to further boost the performance.


<br>

## Our Implementation
Our code is based on the official implementation of [co-teaching](https://github.com/bhanML/Co-teaching). In their code, no data augmentation is used and the backbone network is a single 9-layer CNN. You can change the network to resnet and add standard data augmentation(i.e. crop, horizontal flip) and [mixup](https://arxiv.org/abs/1710.09412) to boost performance.
We did not include those techniques in our experiment since we want to focus on the effect of the filtering step.


dependency:
> python 3.7
>
> pytorch 1.6.0 torchvision 
>
> cudatoolkit 10.1
>
> CUDA 10.1
>
> tqdm (python package)


Below are some examples to run our code.

Run co-PRL-L in cifar10 with 45% pairflipping label noise using 9-layer CNN
>python main.py --algorithm co-PRL-L --dataset cifar10 --noise_rate 0.45 --noise_type pairflip --seed 1

Run co-PRL-L in cifar10 with 45% pairflipping label noise using resnet32
>python main.py --algorithm co-PRL-L --dataset cifar10 --network resnet32 --noise_rate 0.45 --noise_type pairflip --seed 1

Run PRL-L in cifar10 with 45% pairflipping label noise using resnet32
>python main.py --algorithm PRL-L --dataset cifar10 --network resnet32 --noise_rate 0.45 --noise_type pairflip --seed 1

Run PRL-G in cifar10 with 45% pairflipping label noise using resnet32 with group normalization (PRL-G is not compatible with batch normalization)
>python main.py --algorithm co-PRL-G --dataset cifar10 --network resnet32_GN --noise_rate 0.45 --noise_type pairflip --seed 1


<br>

## Acknowledgements
This research is funded by NSF IIS-2006633, EF-1638679, IIS-1749940, Office of Naval Research N00014-20-1-2382, National Institue on Aging RF1AG072449.
Our backbone code is based on [co-teaching](https://github.com/bhanML/Co-teaching). For the PRL-G, we use the [opacus](https://github.com/pytorch/opacus) to calculate the individual gradient.

