Personal notes on classification

### Definition
Problems where observations are defined in a set are **classification problems**
### The Bayes Classifier
**Purpose** : minimize the classification error $$\mathcal{L}(f) = \mathbb{P}(Y \ne f(X))$$
***Ex*** : for binary classification, $$\mathcal{L}(f) = \mathbb{E}[(1-\mathbb{P}(Y = 0 | X))\mathbb{1}_{f(X)=0} + (1-\mathbb{P}(Y=1|X))\mathbb{1}_{f(X)=1}]$$
#### The Bayes Classifier formula
The optimal classifier is given by : $$f(x) = \underset{k}{Argmax}~\mathbb{P}(Y=k|x)$$
***Ex*** : for binary classification,
$$f(x) = 
\begin{cases}
    1 & \text{if } \frac{1}{1+\frac{(1-p) \times p(x)}{p \times p(x)}} > 0.5\\
    0 & \text{else}
\end{cases}$$
Where $p(x) = \mathbb{P}(Y=1|x)$.

### Discriminative approach

Discriminative approaches to classification approximate directly the **posterior distribution** : $p_{k}(x) = \mathbb{P}(Y=k|x)$
#### 1) [[K Nearest Neighbors]]
#### 2) [[Logistic regression]]
