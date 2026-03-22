---
title: 'GAN 3'
subtitle: 'MNIST Linear GAN'
summary: 'MNIST Linear GAN'
authors: 
- admin
tags:
- Deep Learning post
- GAN post## Supervised learning
categories: ['Deep Learning', 'Python', 'PyTorch']
date: "2019-05-21T03:43:00Z"
lastmod: ""
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  caption: 'Image credit: [**Gimages**](https://unsplash.com/photos/CpkOjOcXdUY)'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []


---

We saw an [Intro to GANs](https://shangeth.github.io/post/gan-1/) and the [Theory of Game between Generator and Discriminator](https://shangeth.github.io/post/gna-2/) in the previous posts. In this post we are going to implement and learn about how to train GANs in PyTorch. We will start with MNIST dataset and in the future posts we will implement different applications of GANs and also my research paper on one of the application of GANs.

So the task is to use the MNIST dataset to generate new MNIST alike data samples with GANs.
![](https://cdn-images-1.medium.com/max/1200/1*M2Er7hbryb2y0RP1UOz5Rw.png)

# Let's Code GAN

## Get the Data
Import all the necessary libraries like Numpy, Matplotlib, torch, torchvision.
```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
```

Now lets get the MNIST data from the torchvision datasets.
```python
transform = transforms.ToTensor()
data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=1024)
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADFCAYAAAARxr1AAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBo%0AdHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAC8xJREFUeJzt3V2MVPUZx/HfI2KixQuQsK5Ku1o3%0AGCRKI0ENxEDESgkKeyGRCyQpKVygqS+JgjeK1YRYXoqxNKGALhFQEt+IL6Vm01RICBEJEdTyIlqX%0AlRcBoxAuDPL0Ys6m6+5//js7c2b2zOH7ScjOPHtmzv8Efpxz/jPnOebuAhB2UX8PAMgyAgJEEBAg%0AgoAAEQQEiCAgQAQBASIICBBBQICIiyt5sZlNlrRC0gBJq919cS/L87E9MsPdrbdlrNyvmpjZAEn7%0AJd0l6bCkjyTNdPfPIq8hIMiMUgJSySHWWEkH3f2Qu/8o6VVJ0yp4PyBzKgnI1ZLauzw/nNR+xszm%0AmtlOM9tZwbqAflHROUgp3H2VpFUSh1ioP5XsQTokDe/y/JqkBuRGJQH5SFKzmV1rZpdIul/S5nSG%0ABWRD2YdY7n7OzB6UtEWFad617v5paiMDMqDsad6yVsY5CDKk2tO8QO4RECCi6tO8SN/AgQOD9UWL%0AFgXrCxcuDNYfffTRHrXly5eXP7AcYg8CRBAQIIKAABEEBIggIEAEs1h1qKWlJVifPn16sH7+/Plg%0AnbazvWMPAkQQECCCgAARBASI4CS9Dm3atClYHzNmTLA+YsSIYP2WW25JbUx5xR4EiCAgQAQBASII%0ACBBBQIAIZrHq0BVXXBGsjx8/vk/vs3Tp0jSGk2uVNq/+StJpST9JOufu4XlGoE6lsQeZ6O4nUngf%0AIHM4BwEiKg2IS/qnmX1sZnNDC9C8GvWs0kOs8e7eYWbDJH1gZv9x9w+7LkDzatSzigLi7h3Jz+Nm%0A9qYK9wz5MP4qVGrjxo3B+q233hqs79mzJ1hvb28P1vF/ZR9imdkvzOzyzseSfitpb1oDA7Kgkj1I%0Ag6Q3zazzfTa4+z9SGRWQEZV0dz8k6eYUxwJkDtO8QAQBASL4LlaGjR07Nli/8cYbg/WzZ88G60uW%0ALAnWT548Wd7ALiDsQYAIAgJEEBAggoAAEQQEiOAutxnQ1NQUrG/dujVYv+qqq4L1DRs2BOuzZs0q%0Aa1x5x11ugQoRECCCgAARBASIICBABN/FyoB58+YF68Vmq9ra2oL1Rx55JLUxoYA9CBBBQIAIAgJE%0AEBAggoAAEb3OYpnZWklTJR1391FJbYik1yQ1SfpK0gx3/656w8yHcePGBeuzZ8/u0/s8//zzwfqJ%0AE5W3SB46dGiwfv311wfr3377bbD+xRdfVDyWLChlD/KypMndagsktbl7s6S25DmQO70GJGkleqpb%0AeZqk1uRxq6TpKY8LyIRyPyhscPcjyeOjKjSRC0qaWgcbWwNZV/En6e7uses8aF6NelZuQI6ZWaO7%0AHzGzRknH0xxUHlx22WU9aosWLQou29AQ3gFv2bIlWN+1a1ewPnjw4GD9pptuCtbnzJnTo3bzzeFm%0AmaNGjQrWv/nmm2D9nnvuCdZ3794drGdVudO8myV1Tr3MlvR2OsMBsqXXgJjZRknbJY0ws8NmNkfS%0AYkl3mdkBSZOS50Du9HqI5e4zi/zqzpTHAmQOn6QDEQQEiKDtT5XcfffdPWrvvfdecNn9+/cH68W+%0AmlLs72zdunXB+pQpU4L1alq5cmWw/tBDD9V4JMXR9geoEAEBIggIEEFAgAgCAkTQ9qdCjY2NwXqx%0AGaWQF198MVg/dar7VQYFL730UrDe19mqbdu29aht3749uGyxC6CWLVvWp3XWG/YgQAQBASIICBBB%0AQIAIAgJEMItVoWJX64Xa57z77rvBZdevXx+s33fffcF6S0tLsH78ePjCztbW1mD9mWee6VE7e/Zs%0AcNmpU6cG6wMHDgzW84I9CBBBQIAIAgJEEBAggoAAEeU2r35a0h8kdXYuftLdw5fL5cSll14arD/+%0A+OMlv8crr7wSrBe71drq1auD9UGDBgXrO3bsCNYXLKi8dfKQIUOC9fb29mB9xYoVFa8zC8ptXi1J%0Ay919dPIn1+HAhavc5tXABaGSc5AHzewTM1trZuGelyo0rzaznWa2s4J1Af2i3ID8TdKvJY2WdETS%0A0mILuvsqdx/j7mPKXBfQb8oKiLsfc/ef3P28pL9LGpvusIBsKOu7WJ2d3ZOnLZL2pjekbLrjjjuC%0A9QkTJgTrp0+f7lH7/vvvg8suX748WC82W7VmzZpgvVj3+L5oamoK1p944olg/YUXXgjWDx48WPFY%0AsqCUad6NkiZIGmpmhyU9JWmCmY2W5Crco3BeFccI9Jtym1eH/wsDcoZP0oEIAgJEEBAggisKS3Tv%0Avff2afmjR4/2qA0bNiy47KRJk4L1YlcIFutF1dHRUeLoCkJXJj777LPBZW+44YZgvVjvrrxgDwJE%0AEBAggoAAEQQEiOAkvYauvPLKPi1frAXPxIkT+1Rvbm4O1ufPn9+jdvHF4X8S+/btC9bff//9YD0v%0A2IMAEQQEiCAgQAQBASIICBBhxW5KX5WVmdVuZSm77rrrgvUDBw4E6+fPn+9RK3Zbtttvvz1YHzFi%0ARImjS8/ChQuD9ZUrVwbrZ86cqeZwqsrdrbdl2IMAEQQEiCAgQAQBASIICBDR6yyWmQ2XtE5Sgwpd%0ATFa5+wozGyLpNUlNKnQ2meHu3/XyXnU7i3XRReH/S4q14HnggQeqOZw+OXToULA+eXLPlstffvll%0AcNnQrFy9S2sW65ykx9x9pKTbJM03s5GSFkhqc/dmSW3JcyBXSmlefcTddyWPT0v6XNLVkqZJ6rw7%0AZKuk6dUaJNBf+vR1dzNrkvQbSTskNXTprnhUhUOw0GvmSppb/hCB/lPySbqZDZL0uqSH3f2Hrr/z%0AwolM8PyC5tWoZyUFxMwGqhCO9e7+RlI+ZmaNye8bJYVbcAB1rJRZLFPhHOOUuz/cpf5nSSfdfbGZ%0ALZA0xN2j9yOr51msYvr6Ha00fP3118F6sebVxW79du7cudTGVI9KmcUq5RxknKRZkvaY2e6k9qSk%0AxZI2mdkcSf+VNKPcgQJZVUrz6m2SiiXtznSHA2QLn6QDEQQEiCAgQARXFOKCxRWFQIUICBBBQIAI%0AAgJEEBAggoAAEQQEiCAgQAQBASIICBBBQIAIAgJEEBAggoAAEQQEiCAgQESvATGz4Wb2LzP7zMw+%0ANbM/JvWnzazDzHYnf6ZUf7hAbZXSF6tRUqO77zKzyyV9rEIf3hmSzrj7kpJXxhWFyJBU+mIl/XeP%0AJI9Pm1ln82og9/p0DtKtebUkPWhmn5jZWjMbXOQ1c81sp5ntrGikQD8ouWlD0rz635Kec/c3zKxB%0A0gkVmlb/SYXDsN/38h4cYiEzSjnEKikgSfPqdyRtcfdlgd83SXrH3Uf18j4EBJmRSleTpHn1Gkmf%0Adw1HZ2f3RIukveUMEsiyUmaxxkvaKmmPpM4b1T0paaak0SocYn0laV6XG+oUey/2IMiM1A6x0kJA%0AkCU0jgMqRECACAICRBAQIIKAABEEBIggIEAEAQEiCAgQUcp90tN0QoV7qkvS0OR53rGd2fSrUhaq%0A6VdNfrZis53uPqZfVl5DbGd94xALiCAgQER/BmRVP667ltjOOtZv5yBAPeAQC4ggIEBEzQNiZpPN%0AbJ+ZHTSzBbVefzUl7Y+Om9neLrUhZvaBmR1IfgbbI9WTSLfN3G1rTQNiZgMk/VXS7ySNlDTTzEbW%0AcgxV9rKkyd1qCyS1uXuzpLbkeb07J+kxdx8p6TZJ85O/x9xta633IGMlHXT3Q+7+o6RXJU2r8Riq%0Axt0/lHSqW3mapNbkcasKbVvrmrsfcfddyePTkjq7beZuW2sdkKsltXd5flj5b2Pa0KXby1FJDf05%0AmLR167aZu23lJL2GvDCnnpt59aTb5uuSHnb3H7r+Li/bWuuAdEga3uX5NUktz451NtlLfh7v5/Gk%0AIum2+bqk9e7+RlLO3bbWOiAfSWo2s2vN7BJJ90vaXOMx1NpmSbOTx7Mlvd2PY0lFsW6byuO21vqT%0A9ORGO3+RNEDSWnd/rqYDqCIz2yhpggpf/T4m6SlJb0naJOmXKnzVf4a7dz+RryuRbps7lLdt5asm%0AQHGcpAMRBASIICBABAEBIggIEEFAgAgCAkT8D+0/bl0Rjxl0AAAAAElFTkSuQmCC)

## The Model
As we have already seen in [Theory of Game between Generator and Discriminator](https://shangeth.github.io/post/gna-2/), the GAN models generally have 2 networks Discriminator D and Generator G.
We will code both of these network as seperate classes in PyTorch.
![](https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/gan-mnist/assets/gan_network.png)

### Discriminator
The discriminator is a just a classifier , which takes input images and classifies the images as real or fake generated images. So lets make a classifier network in PyTorch. 

```python
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(D, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)      
        
    def forward(self, x):
        # flatten image
        x = x.view(-1, 28*28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = F.log_softmax(self.fc4(x))
        return out
```

The D network has 4 linear layers with leaky relu and dropout layers in between.

Here the input size will be 28*28*1 (size of MNIST image)\\
hidden dim can be anything of your choice.\\
output_size = 2 (real or fake)

I am also adding a log softmax in the end for computation purpose.

Lets make a Discriminator object
```python
D_network = D(28*28*1, 50, 2)
print(D_network)
```
output :
```
D(
  (fc1): Linear(in_features=784, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=50, bias=True)
  (fc4): Linear(in_features=50, out_features=2, bias=True)
  (dropout): Dropout(p=0.3)
)
```

### Generator
The Generator takes a random vector(z)(also called latent vector) and generates a sample image with a distribution close to the training data distribution. We want to upsample z to an image of size 1*28*28. Tanh was used as activation in the output layer(as used in the original paper) , but feel free to try other activations and check which gives good result.

```python
class G(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(G, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc4 = nn.Linear(hidden_dim*4, output_size) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        out = F.tanh(self.fc4(x))
        return out
```
The G network architecture is same as D's architecture except now we upsample the z to 28*28*1 size image.
```python
G_network = G(100, 50, 1*28*28)
print(G_network)
```

```
G(
  (fc1): Linear(in_features=100, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=200, bias=True)
  (fc4): Linear(in_features=200, out_features=784, bias=True)
  (dropout): Dropout(p=0.3)
)
```

## Loss

The discriminator wants the probability of fake images close to 0 and the generator wants the probability of the fake images generated by it to be close to 1.

So we define 2 losses

* Real Loss (loss btw p and 1)
* Fake loss (loss btw p and 0)

p is the probability of image to be real.

* For Generator :
minimize real_loss(p) or p to be closer to 1. ie: fool generator by making realistic images.

* For Discriminator :
minimize real_loss + fake loss. ie: p of real image close to 1 and p of fake image close to 0.

```python
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    criterion = nn.NLLLoss()
    loss = criterion(D_out.squeeze(), labels.long().cuda())
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.NLLLoss()
    loss = criterion(D_out.squeeze(), labels.long().cuda())
    return loss
```

[label smoothing](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b) is also done for better convergence.

## Training

We will use 2 optimizers

- One for Generator, which optimizes the real_loss of fake images. ie: it tries to make the classification prediction of fake images equal to 1.
- Next is discriminator, which tries to optimize real+fake loss. ie: it tries to make the prediciton of fake images to 0 and real images to 1.

Adjust the no of epochs, latent vector size, optimizer parameters, dimensions etc.

```python
num_epochs = 100
print_every = 400

# train the network
D.train()
G.train()
for epoch in range(num_epochs):
    for batch_i, (images, _) in enumerate(train_loader):         
        batch_size = images.size(0)
        
        ## Important rescaling step ## 
        real_images = images*2 - 1  
        # rescale input images from [0,1) to [-1, 1)

        d_optimizer.zero_grad()
        D_real = D(real_images)
        d_real_loss = real_loss(D_real, smooth=True)
        
        
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
      
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        

        g_optimizer.zero_grad()
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake) 
        g_optimizer.step()

        if batch_i % print_every == 0:
            print('Epoch {:5d}/{:5d}\td_loss: {:6.4f}\tg_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
```

```
Epoch     1/  100 d_loss: 1.3925  g_loss: 0.6747
Epoch     2/  100 d_loss: 1.2275  g_loss: 0.6837
Epoch     3/  100 d_loss: 1.0829  g_loss: 0.6959
Epoch     4/  100 d_loss: 1.0295  g_loss: 0.7128
Epoch     5/  100 d_loss: 1.0443  g_loss: 0.7358
Epoch     6/  100 d_loss: 1.0362  g_loss: 0.7625
Epoch     7/  100 d_loss: 0.9942  g_loss: 0.8000
Epoch     8/  100 d_loss: 0.9445  g_loss: 0.8455
Epoch     9/  100 d_loss: 0.9005  g_loss: 0.9073
Epoch    10/  100 d_loss: 0.8604  g_loss: 0.9908
...
```

## Generate new MNIST Samples

```python
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

sample_size=16
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()

G.eval()
rand_images = G(rand_z)
view_samples(0, [rand_images])
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZkAAAGRCAYAAAC39s6jAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBo%0AdHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJztnWm4FdWVhveNcWIScEBBBBRBUVEB%0ABYdGjVPHiVbj1BgHNImtJumgSWur7ZDY2q3RjlOMQ7TFJI5xwqgxYsQRFQFFQVQEZJJBAXGM5vSP%0Afmr5HqjFrbqn9hku3/vre4q6dar2rjrF+s5aazeVSqUghBBCxOAbtT4BIYQQrRe9ZIQQQkRDLxkh%0AhBDR0EtGCCFENPSSEUIIEQ29ZIQQQkRDLxkhhBDR0EtGCCFENPSSEUIIEY1v5tm5qalJ7QEqoKmp%0AKYQQQqlUCqVSqanA42peCqKoedGcFMqiUqm0YREH0rwUSqZ5yfWSaWSSL/gQ/v9LvhasueaaIYQQ%0A/va3v9Xk84VoUGZW88NifFdkOeY3v/n/X8dfffVV6t/9/e9/z/WZa6yxhmkesxK+8Y2vza+///3v%0AmeZFdpkQQoho1CSSKep/Ciu8VVe5b5bPyXO8lvDFF18UfkwhRMvwvodiOB08Jj+X3zlffvllxZ/D%0AYxcVvZCWfC8qkhFCCBENvWSEEEJEoyZ2WXM/fIXgh45Z9mkpMSwyIWpFPSS71DNZxiTLGHIf7/ie%0ARZZmafEHe+87yTuvepxzRTJCCCGioZeMEEKIaNRVnUwW+6u5jAmGorEzRoSoZ3TPZydvplkWu6pt%0A27amWRvHLFNaY8lxaJHx3/n96J0Xt2epk6mGvaZIRgghRDT0khFCCBGNurLLstBcSJclGyNp77Kq%0Av/WOw+3t2rUzvXz58lWel0iH8+JlDhaVBSSyMXjwYNMcy8mTJ5v+5JNPqnpORcIeggme5UW4j3ev%0AesWQ3GettdYyTXs/0Rz/JUuWmJ40aVKzn+99h3nWWefOnVM/q8hCTkUyQgghoqGXjBBCiGgUbpfl%0A7f9VlM2RhIM8Bj+f4WWHDh1M9+zZ0/TWW29tesCAAaYvu+wy03PnzjUtiyyd9u3bmz7xxBNDCCF8%0A9NFHtm3EiBGmR40aZXrmzJmpeurUqc1+piyyyuC9/+STT5qeNWuW6WXLlpnebbfdTDdaV/G0eyXL%0A99A666xj+vPPPze9ySabmH7//fdNr7322qb5vcjx4nfRt7/97RBCCH379rVt06ZNM7106VLTfD74%0A3cZje9Ydt3/wwQemYxV1KpIRQggRDb1khBBCRKNwuyxG/68soVtaNgT/jqEr7a9tttnGNK2wsWPH%0Aml60aFGLz6u1sd5665led911TZ9wwgmmN910U9Pf//73Qwjl2S0cN9ouHMPFixebPuCAA0y/8sor%0ApvPea7Q7Pvvss1x/2xphltO4ceNM81np06eP6V/+8pem119/fdO0iBr1OchSdMl7hvcSr59Wsbc/%0A2WqrrUwn30U8BuHnkLw9HHmtzLSl1cbMWVrdLUGRjBBCiGjoJSOEECIaLbbLimq574WpDONotbDv%0AT3M9hpjRwYyNI444wjT7C40ePdr0NddcY9rLnmlUayAvHMcjjzzS9LvvvmuaduWee+5pOpk7b61y%0Ar9cSi8TOOuss08cdd5xpzkuW4k1ZZD6cE68YkVmB5513nunW8BzkbenP7yGOHTPAvP1pM7/11lum%0A58yZE0Iov5ffeecd016GGPGeLT7DfM6877ZKLTKiSEYIIUQ09JIRQggRjRbbZUWvShmC38eK2z3b%0Apbl9/+Vf/sU0Mz0Yxp599tmmGUa2BjsgLxy7Tp06mX7++edN09KidUYLcv78+SGEEC688ELbRotg%0Aww03NH3uueeaZpHsXnvtZfq//uu/TP/oRz/KcCVfk7dQuLXTsWNH0xwb7xkbM2aMaRYjtgaytM7P%0AexyOKb9zevfubXr27Nmm//znP4cQQjjooINsG6396dOnm164cKFp2mj8TA9+t1XjmVAkI4QQIhp6%0AyQghhIhGXbX6Z5jOQrHu3bubZi+ftOwy2my0c1j0x/CSPYBYdLk6WmQetEZoi7Hv0dtvv216yJAh%0AphcsWBBC8MeT88x9LrnkEtNdunQx7dmoWUL9DTbYYKXzam3kKRDmmDHjyeORRx4x7a3Y2Ejwmj/9%0A9FPTWVr9e2PLcWFRKy0y9oDj5ybbWcy88847m+7fv7/p8ePHpx6Pc8rvOc9Sq0Z/SUUyQgghoqGX%0AjBBCiGgUYpcVVZhJ62SzzTYzzYIkL7xLijdZxMnMJVphPPZ9991nmmFnXphRRXupllZCJWEuw372%0AK9tiiy1Mz5s3z/R7771n2iuYTeD9svvuu5s+44wzTG+88camGeqzYJZksQDqwSKL3e8uzzG/853v%0A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%0A)

Linear GAN Model does a decent job in generating MNIST images. In next post we will look into DCGAN(Deep Convolutional GAN), to use CNNs for generating new samples.

[Check this Awesome Repo](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN) on comparing Linear GAN and DCGAN for MNIST.
Also [this notebook](https://github.com/Yangyangii/GAN-Tutorial/blob/master/MNIST/VanillaGAN.ipynb) for pytorch implementation of vanilla GAN(Linear).