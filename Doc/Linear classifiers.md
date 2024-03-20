### The Bayes Classifier
Goal : classify the response Y given a set of observations X through Y $\approx$ f(X)
We need to minimize the classification error $$\mathbb{E}(\sum_{i}(1-\mathbb{P}(Y=i))\mathbb{1}_{f(x_{0})=i})$$
To do so, we need to find a function f so that $$f(x_{0}) = \underset{k}{Argmax}(\mathbb{P}(Y=k|X \in \mathcal{B}_{\epsilon}(x_{0})))$$
### Discriminative approaches
**Focuses on learning the decision boundary between classes and approximate directly the posterior distribution**

#### KNN
**Uses the nearest neighbors from an observation to capture its class membership**
Under KNN we approximate the posterior prob using : $$\mathbb{P}(Y=k|X \in \mathcal{B}_{\epsilon}(x_{0})) \approx \sum_{i}\mathbb{1}_{y=k}(y_{i})\mathbb{1}_{B_{\epsilon}(x_{0})}(x_{i}) * (\sum_{i}\mathbb{1}_{B_{\epsilon}(x_{0})}(x_{i}))^{-1} \approx \frac{1}{K}\sum_{x_{i} \in \mathcal{N}_{K}(x_{0})}\mathbb{1}_{y=k}(y_{i})$$
Where ${B_{\epsilon}}(x_{0})$ is approximated by the N nearest neighbors of $x_{0}$, N is determine using cross validation.

##### Advantages 
1. Non parametric approaches
2. No hypothesis on the data
3. Works well in low dimension

##### Disdvantages
1. No efficient in larger dimensions

#### Logistic regression
**Focuses on applying linear regression through probability**

$$\mathbb{P}(Y=k|X \in \mathcal{B}_{\epsilon}(x_{0})) \approx \frac{\exp(\phi(x))}{1+\exp(\phi(x))}$$
Where $$ \phi(x) = \beta_{0} + \sum_{k=1}^{p}\beta_{k}x_{k} $$
##### Decision boundaries
**Hyperplan** verifying $\phi(x)=0$ 

##### Finding betas
Betas are found using Maximum Likelihood Estimation (MLE)
**In this model, observations are Bernoulli random variables**

In practice (for two classes here), we try to compute gradient descent or second order method to maximize the quantity (**loss function is concave**) $$\mathcal{L}(\beta) = \sum_{i}y_{i}log(\pi(x_{i};\beta)) + (1-y_{i})log(1-\pi(x_{i};\beta))$$
Where $\pi(x_{i};\beta)$ is the posterior related to class 1.

##### Multinomial regression
**If more than two classes**, we replace $\exp(\phi(x))$ with $\sum_{i}\exp(\phi_{i}(x))$

### Generative approaches
**Try to modeling the full probability distribution of the data**

#### Gaussian Model

##### LDA (Linear Discriminant Analysis)
##### QDA (Quadratic Discriminant Analysis)
##### Naive Bayes
**see the original paper for more details**

