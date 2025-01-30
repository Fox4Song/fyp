# Few-Shot Learning vs In-Context Learning: any difference?

## Table of Contents
- [Neural Processes](#neural-processes)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Examples](#examples)
- [References](#references)

## Neural Processes

Deep learning excels in data-driven predictions but struggles in small data settings where reliable uncertainty estimation is critical, such as predicting patient treatment outcomes where individual data are limited. Traditional deep neural networks often lack the ability to quantify their uncertainty, risking confident but incorrect predictions. The **Neural Process Family (NPF)** is a collection of models that addresses these challenges by *meta-learning* a distribution over predictors through neural processes. This approach leverages data from multiple related tasks to effectively model uncertainty using a stochastic process framework. Consequently, NPFs provide more accurate and reliable predictions in scenarios with limited data and high uncertainty requirements.

## Examples on 1D Toy Regression Dataset

### CNP
<img src="images/1d_toy_regression/CNP-160000.png" width="240" height="180"> <img src="images/1d_toy_regression/CNP-180000.png" width="240" height="180"> <img src="images/1d_toy_regression/CNP-200000.png" width="240" height="180">

- Suffers from underfitting.
- Often does not pass through all context points.
- Overestimates uncertainty.

### LNP 
<img src="images/1d_toy_regression/LNP-160000.png" width="240" height="180"> <img src="images/1d_toy_regression/LNP-180000.png" width="240" height="180"> <img src="images/1d_toy_regression/LNP-200000.png" width="240" height="180">

- Produces coherent sampling from the posterior predictive.
- Beyond [-1,1], model seems to disregard context points and uncertainty despite being trained in the range [-2,2].

### AttnCNP
<img src="images/1d_toy_regression/AttnCNP-140000.png" width="240" height="180"> <img src="images/1d_toy_regression/AttnCNP-160000.png" width="240" height="180"> <img src="images/1d_toy_regression/AttnCNP-180000.png" width="240" height="180">

- Still demonstrates underfitting issues.
- 'Kinks' in predictive distribution.

### Inference plots for NPs and Transformers

<img src="images/1d_toy_regression/np_vs_tf_2_context.png" width="240" height="180"> <img src="images/1d_toy_regression/np_vs_tf_4_context.png" width="240" height="180"> <img src="images/1d_toy_regression/np_vs_tf_10_context.png" width="240" height="180">

- NP generally captures the overall ground truth function better
- Transformer appears to struggle in regions where the function has high variability, especially around sharp peaks.

<img src="images/1d_toy_regression/np_vs_tf_20_context.png" width="240" height="180">

#### As the number of context points increase...

- NP improves its fit and produces narrower uncertainty regions.
- Transformer performance deteriorates.
- Transformer's sensitivity to the number of context points might indicate it is not effectively utilising global dependencies.

## References
- [1]. Dubois Y, Gordon J, Foong AYK. Neural Process Family. September 2020. Available from: http://yanndubs.github.io/Neural-Process-Family/  
- [2]. Tuan Anh Le, Hyunjik Kim, Marta Garnelo, Dan Rosenbaum, Jonathan Schwarz, and Yee Whye Teh. Empirical evaluation of neural process objectives. In NeurIPS workshop on Bayesian Deep Learning. 2018.  
- [3]. Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M, Teh YW, Rezende DJ, Eslami SMA. Conditional Neural Processes. CoRR. 2018;abs/1807.01613. Available from: http://arxiv.org/abs/1807.01613.  
- [4]. Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S. M. Ali Eslami, and Yee Whye Teh. Neural processes. CoRR, 2018. URL: http://arxiv.org/abs/1807.01622, arXiv:1807.01622.  
- [5]. Kim H, Mnih A, Schwarz J, Garnelo M, Eslami SMA, Rosenbaum D, Vinyals O, Teh YW. Attentive Neural Processes. CoRR. 2019;abs/1901.05761. Available from: http://arxiv.org/abs/1901.05761.
- [6]. Vaswani A. Attention is all you need. Advances in Neural Information Processing Systems. 2017. Available from: https://arxiv.org/abs/1706.03762

