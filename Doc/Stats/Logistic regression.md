Extension of linear regression to classification
Is often used for binary classification

### Logistic function

To determine the posterior probability, we use a **logistic function** (value between 0 and 1)

$$p(X)=\mathbb{P}(Y=1|x) = \frac{e^{\beta x}}{1~+~e^{\beta x}}$$
Where $\beta x= \sum_{i}\beta_{i}x_{i}$

### Log odds logit
**Odds** are a ratio of probabilties, explaining tendency of the distribution.

**Log Odds** are defined as : $$log \bigg(\frac{p(X)}{1-p(X)}\bigg) = \beta x$$
Betas are found using maximum likelihood, which means maximizing the quantity : $$\prod_{i~;~y_{i}=1}p(x_{i})\prod_{i~;~y_{i}=0}(1-p(x_{i}))$$