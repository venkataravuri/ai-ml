# Machine Learning
- [Balanced and Imbalanced datasets]()
- [Feature Engineering]()
     - [Encoding Methodologies]()
     - [Feature Selection & Elemination](#feature-selection--elemination)
- [Regularization](#regularization)
- [Loss Functions](#loss-functions)
- [Optimization Methods]()
- [Activation Functions](#activation-functions)
- [Evaulation Metrics](#evaluation-metrics)

- [Information Theory (Entropy, KL Divergence, Cross Entropy)](https://gaussian37.github.io/ml-concept-infomation_theory/)

### What are Logits in machine learning?

Logits interpreted to be the unnormalised (or not-yet normalised) predictions (or outputs) of a model. These can give results, but we don't normally stop with logits, because interpreting their raw values is not easy.

[Logits Explanation](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)|

In a classification problem, the model’s output is usually a vector of probability for each category. Often, this vector is usually expected to be “logits,” i.e., real numbers to be transformed to probability using the softmax function, or the output of a softmax activation function.

#### Why we want to wrap everything with a logarithm?

1. It’s related to the concept in information theory where you need to use log(x) bits to capture x amount of information.
2. Computers are capable of almost anything, except exact numeric representation.

## Balanced and Imbalanced datasets

In machine learning, class imbalance is the issue of target class distribution.

|Balanced datasets|Imbalanced datasests|
|---|---|
|For balanced datasets, the target class distribution is nearly equal.| For imbalanced datasest, the target distribution is not equal.|
|Balanced datasets<br/>• A random sampling of a coin trail<br/>• Classifying images to cat or dog<br/>• Sentiment analysis of movie reviews|Class Imbalance dataset<br/>• Email spam or ham dataset<br/>• Credit card fraud detection<br/>• Network failure detections|

### Techniques for handling imbalanced data

|Oversampling|Undersampling|Ensemble Techniques|
|---|---|---|
|Increase the number of samples in minority class to match up to the number of samples of the majority class.|Decrease the number of samples in the majority class to match the number of samples of the minority class.|Ensemble methods, which combine multiple base learners to produce a more accurate and robust model, can be adapted to handle imbalanced data effectively|
|||Bagging for Imbalanced Data - Bagging, or bootstrap aggregating, involves training multiple base learners on different random subsets of the data and combining their predictions.<br/>Boosting for Imbalanced Data - Boosting is an ensemble method that trains a series of base learners sequentially, where each learner tries to correct the errors made by its predecessor. 

[Reference](https://dataaspirant.com/handle-imbalanced-data-machine-learning/)


## Feature Engineering

These methods can help improve the performance of machine learning models by providing more meaningful and relevant information1.

### Encoding Categorical Variables

#### Nominal Encoding - Where Order of data does not matter

Convert Naming or label variables (categorical variables) into a numerical format for ML models to process them efficiently.

|One Hot Encoding|One Hot Encoding with many categorical (like Pincode)|Mean Encoding|Frequency Encoding|
|---|---|---|---|
|Frequency of the categories as labels|• Suppose there are more than 20 categories of a variable then we can’t apply direct One Hot Encoding. <br />• Find most top k categories repeating most frequent. <br />• Then take that k category and create k new features.|• Mean Encoding or Target Encoding is similar to label encoding, except here labels are correlated directly with the target.<br/>• For example, in mean target encoding for each category in the feature label is decided with the mean value of the target variable on a training data.||
|<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*UB3J51jn2XtIkmIp4HWRQA.png" height="100%" width="100%" />||<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*8lK9mSxuPJ4b9SUXA3dN-A.png" height="100%" width="100%" />|<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*l0mPlpqFEK_DSu4OqSnvLg.jpeg" height="100%" width="100%" />|
|For high cardinality features, this method produces a lot of columns that slows down the learning significantly.<br/> One hot encoding produces the number of columns equal to the number of categories and dummy producing is one less.||Advantages: <br/>• Capture information within the label, therefore rendering more predictive features<br/>• Creates a monotonic relationship between the variable and the target. <br/>Disadvantages:<br/>• It may cause over-fitting in the model.||

### Ordinal Encoding — Where Order of data matters

Ordinal data is a categorical data type where the variables have natural, ordered categories and the distances between the categories are not known. Example: Rating, rank, etc.

|Label Encoding|Target guided Ordinal Encoding|Count Encoding|
|---|---|---|
|Map each categorical feature value to an integer number starting from 0 to cardinality-1, where cardinality is the count of the feature’s distinct values.|Map each category to a new feature vector that contains 1 and 0 denoting the presence of the feature or not. The number of new feature vectors depends on the categories which we want to keep.|Labels are given on the bases of the mean. Highest the mean, highest the label<br/>Ordering the labels according to the target variable.<br/>Replace the labels by the joint probability.|Replace the categories with there count.<br/>It is used when are lots of categories of a variable.<br/>It does not create a new feature.<br/>Disadvantage: If the same labels have the same count then replaced by the same count and we will lose some valuable information.|
|<img src="https://miro.medium.com/v2/resize:fit:550/format:webp/1*UhTyTyIIOaos5jVlbeQllw.png" height="100%" width="100%" />|||

### Binning or Discretization

Binning or Discretization is the process of transforming continuous numerical variables into discrete categorical ‘bins’ for grouped analysis.

In a marketing dataset, the age of customers can be discretized into bins (like 18-25, 26-35, etc.) to create age groups, which can be more meaningful for analysis.

### Date and Time Features
Extracting features from date and time such as day of the week, month, quarter, year, etc

### Feature Interaction
Creating new features by combining two or more features

 In a credit scoring model, interaction between features like income and loan amount can be important. If the loan amount is small compared to the income, the likelihood of loan repayment might be higher.

### Feature Scaling

This method is used to standardize the range of independent variables or features of data.

### Feature Transformation

This includes methods like logarithmic transformation, square root transformation, etc., to transform the feature to fit non-linear relationships.

### Text Data Preprocessing

This involves cleaning and converting text data into a format that can be used by machine learning algorithms.

In a sentiment analysis task, raw text reviews can be preprocessed by techniques like tokenization, stopword removal, and stemming to prepare the data for a machine learning model.

## Feature Selection & Elemination

Methods for optimal feature selection,

- Manual Feature Elimination
     - Drop features that are least helpful in prediction (high p-value)
     - Drop features that are redundant (using correlations, VIF)
- Automated Approach
     - Recursive Feature Elimination (RFE)
 
**Variance Inflation Factor (VIF)**

VIF helps explain the relationship of one independent variable with all the other independent variables. The formulation of VIF is given below,
$$\frac{1}{1-R_{i}^{2}}$$

The common heuristic for VIF is that while a VIF greater than 10 is definitely high, a VIF of greater than 5 should also not be ignored and inspected appropriately.

### Optimization Methods

#### Gradient Descent

**Gradient Descent** is an optimization method used to optimize the parameters of a model using the gradient of an objective function (loss function in NN). It optimizes the parameters until the value of the loss function is the minimum (of we've reached the minima of the loss function). It is often referred to as back propagation in terms of Neural Networks.

#### Stochastic Gradient Descent

**Stochastic Gradient Descent** computes the gradients for each and every sample in the dataset and hence makes an update for every sample in the dataset. For a dataset of 100 samples, updates occur 100 times.

#### Batch Gradient Descent

**Batch Gradient Descent**
The samples from the whole dataset are used to optimize the parameters i.e to compute the gradients for a single update. For a dataset of 100 samples, updates occur only once.

#### Mini-batch Gradient Descent
**Mini Batch Gradient Descent**, instead of a single sample ( Stochastic GD ) or the whole dataset ( Batch GD ), we take small batches or chunks of the dataset and update the parameters accordingly. For a dataset of 100 samples, if the batch size is 5 meaning we have 20 batches. Hence, updates occur 20 times.

## Regularization

**Overfitting** occurs when a model learns to perform exceptionally well on the training data but fails to perform well on new, unseen data.

**Regularization** helps in preventing the overfitting of the model and the learning process becomes more efficient.

Regularization techniques,
- Early stopping
- Dropout
- Weight initialization techniques
- and _batch normalization_.

**Regularization** is a set of techniques designed to prevent overfitting and enhance the generalization ability of a model.

Regularization is a technique used to avoid overfitting where the coefficients, if needed, are restricted or shrunken to zero. 

Reducing the impact of less important features directly affects the quality of predictions as it reduces the _degree of freedom_ which in turn makes it harder for the model to get more complex or overfit the data.

Regularization methods introduce additional constraints or penalties to the learning process to ensure that the model does not become overly complex and is better suited for making accurate predictions on new data.

A penalty term is added to the cost function which lets us control the type and amount of regularization to be performed on the model at hand. This is done by modifying the traditional Linear Regression Cost function that is shown below.

```math
J(\theta) = MSE(\theta) = (\frac{1}{m}).\displaystyle\sum_{i=1}^{n}(\theta^T x^{(i)} - y^{(i)})^2
```
<p align="center">Linear Regression Cost Function</p>

Ridge Regression :

This type of regularized regression has a penalty term representing half the square of L2 norm added to the cost function. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. The equation for Ridge Regression is shown below.

This term tends to shrink all of the weight coefficients, but unlike L1 regularization, it does not set any weight coefficients to zero.

L2 regularization, or Ridge regularization, adds a term to the cost function that is proportional to the square of the weight coefficients:

```math
J(\theta)_{Ridge} = MSE(\theta) + \lambda.(\frac{1}{2}).\displaystyle\sum_{i=1}^{n}(\theta^2_{i})
```
<p align="center">Ridge Regression Cost Function</p>

> The L2 norm is the sum of the squares of the differences between predicted and target values over the feature vector. Its also known as Euclidean Distance and Root Mean Square Error (RMSE).

The shrinkage hyperparameter λ (lambda) controls the amount of regularization and needs to be chosen properly because if λ = 0, then Ridge Regression is the same as Linear Regression and on the other hand, if λ is very large, then all weights end up very close to zero resulting in an underfitting model. One good way to select the right λ is to perform cross-validation.

#### Lasso Regression known as L1 regularization

This type of regularization is also known as Lasso regularization. It adds a term to the cost function that is proportional to the absolute value of the weight coefficients:

```math
J(\theta)_{Ridge} = MSE(\theta) + \lambda.\displaystyle\sum_{i=1}^{n}(|\theta_{i}|)
```




It tends to shrink some of the weight coefficients to zero. The sum of the term is multiplied by lambda, which controls the amount of regularization. If lambda is too high, the model will be simple, and the risk of underfitting arises.

Short for Least Absolute Shrinkage and Selection Operator Regression, this type of Regularized Regression uses the L1 norm instead of half the square of L2 norm as the penalty term in the cost function. An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features and thus, automatically performs feature selection.

> The L1 norm is the sum of the magnitudes of the differences between predicted and target values over the feature vector or could be understood as the sum of absolute differences. Its also known as Manhattan Distance, Taxicab Norm, and Mean Absolute Error (MAE).


<p align="center">Lasso Regression Cost Function</p>

The shrinkage hyperparameter λ works similar to as in Ridge Regression, too little results in no regularization and too much ends up in an underfit model.

The key difference between Ridge and Lasso regression is that even though both the regression techniques shrink the coefficients closer to zero, only Lasso regression actually sets them to zero if the shrinkage parameter is large enough. Thus, resulting in a model having a selected set of features (sparse model) making it much easier to interpret and work with.

Elastic Net Regression :

This kind of regression is simply a mix of both, Ridge and Lasso Regressions. The penalty term in Elastic Nets is a combination of both absolute value and squared value penalties.

> Elastic Net first emerged as a result of critique on Lasso, whose variable selection can be too dependent on data and thus unstable. The solution is to combine the penalties of Ridge regression and Lasso to get the best of both worlds. (Source)
>
> 
```math
J(\theta)_{Ridge} = MSE(\theta) + r.\lambda.\displaystyle\sum_{i=1}^{n}(|\theta_{i}|) +  [{(1-r)/2}].\alpha.\displaystyle\sum_{i=1}^{n}(\theta^2_{i})
```
<p align="center"Elastic Nets Cost Function</p>

The mix between Ridge and Lasso regularization can be controlled by the Ratio hyperparameter (r). When r = 0, Elastic Net is equivalent to Ridge Regression and when r = 1, it is equivalent to Lasso Regression.

#### What is Normalization?

Normalization is a data pre-processing tool used to bring the numerical data to a common scale without distorting its shape.

#### Normalization Vs. Standardization

|<img height="50%" width="50%" src="assets/Normalization-Standardization.png">|[![Alt text](https://img.youtube.com/vi/of4-jeKtyB4/0.jpg)](https://www.youtube.com/watch?v=of4-jeKtyB4)|
|----|----|

#### Batch Normalization

Batch Normalization is a supervised learning technique that converts interlayer outputs into of a neural network into a standard format, called normalizing. 

This approach leads to faster learning rates since normalization ensures there’s no activation value that’s too high or too low, as well as allowing each layer to learn independently of the others. For each layer in the neural network, batch normalization normalizes the activations by adjusting them to have a standardized mean and variance.

In a deep learning network, batch normalization affects the output of the previous activation layer by subtracting the batch mean, and then dividing by the batch’s standard deviation.

## Distance Measurement in Text Mining

In NLP, To find the similarity among sentence or document.

Before any distance measurement, text have to be tokenzied. 4 basic distance measurements:
- Euclidean Distance
- Cosine Distance
- Jaccard Similarity

<img src="https://pbs.twimg.com/media/GXMwYt5W8AA8nQp?format=jpg&name=small" width="50%" height="50%" />


|Euclidean Distance|Cosine similarity|Jaccard similarity|
|---|---|---|
|<img src="https://miro.medium.com/v2/resize:fit:640/0*Bd8VtxN8ql4qw4vo" weight="25%" height="25%" />|<img src="https://miro.medium.com/v2/resize:fit:1170/format:webp/1*MhX64CBNBUQdQyM30jiaYA.png" width="40%" height="50%" />||
|Shortest distance among two vectors. Uses Pythagorean Theorem learnt in secondary school.|Cosine similarity is based on the angle between two vectors that represent the documents. Cosine similarity is calculated using only the dot product and magnitude of each vector. It is thus a judgment of orientation and not magnitude (length)|Jaccard similarity is based on the ratio of the intersection to the union of the sets of words that represent the documents. | 
||- Two vectors with the same orientation have a cosine similarity of 1 (The cosine of 0° is 1)<br/>- Two vectors oriented at 90° relative to each other have a similarity of 0<br/>- and Two vectors diametrically opposed have a similarity of -1, independent of their magnitude.|The higher the ratio, the more similar the documents are.|

#### What is difference between Cosine Similarity & Ecludian Distance?

### Bayes Theorem

**Conditional Probability** is the probability that something will happen, given that something else has already happened.

```
P(A | B) = P(B | A) x P(A)/P(B)
```

Pr(A | B): Conditional probability of A : i.e. probability of A, given that all we know is B;  Probability of A happening given that B has already happened. P(A|B) is the conditional probability.

"Probability of A given B" is the same as the "probability of B given A" times the "probability of A" divided by the "probability of B".

**Understanding Bayes Rule**

P(Outcome given that we know some Evidence) = P(Evidence given that we know the Outcome) times Prob(Outcome), scaled by the P(Evidence)

The classic example to understand Bayes' Rule:

```
Probability of Disease D given Test-positive = 

               P(Test is positive|Disease) * P(Disease)
     _______________________________________________________________
     (scaled by) P(Testing Positive, with or without the disease)
```

**Naive Bayes'**


## Evaluation Metrics

- [Machine Learning tips and tricks cheatsheet](https://stanford.edu/%7Eshervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks#)

### Confustion Matrix

A confusion matrix is a table used to evaluate the performance of a _classification model_ by comparing its predictions to the actual ground truth labels.

It provides a summary of the model’s true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions for each class in a multi-class classification problem or for the positive class in a binary classification problem.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yshTLPeqgL9Nm5vTfGpMFQ.jpeg" width="50%" height="50%" />

|Metric|Description|Formula|Interpretation|
|---|---|---|---|
|**Accuracy**|Accuracy measures the overall correctness of the model’s predictions.|$\frac{(TP + TN)}{(TP + TN + FP + FN)}$|• It is calculated as the ratio of the correctly predicted instances (TP + TN) to the total number of instances in the dataset.<br/>• In **imbalanced datasets**, _accuracy may not be a reliable measure_, as a high accuracy score could be driven by the model’s ability to predict the majority class accurately while performing poorly on the minority class.|
|**Precision**|Precision refers to "out of the total values predicted as positive, how many are actually positive"?|$\frac{(TP)}{(TP + FP)}$|• How accurate the positive predictions are?<br/>• The **higher** it is, the more the Machine Learning model **minimizes the number of False Positives**.<br/>• Use precision when the cost of false positives is high and you want to minimize false alarms or false positive predictions. (minimizing false positives)<br/>• For example, in fraud detection, precision is crucial because it indicates how many flagged cases are actually true frauds, reducing the need for manual investigation of false positives.|
|**Recall** (Sensitivity or True Positive Rate)|Recall refers to "out of the total number of positive samples, how many are correctly identified"?|$\frac{(TP)}{(TP + FN)}$|• It is the **number of well predicted positives** (True Positive) **divided by the total number of positives** (True Positive + False Negative).<br />• Use recall when the cost of false negatives is high, and you want to ensure that you capture as many positive instances as possible, even at the expense of some false positives. (minimizing false negatives)<br/>• For example, in medical diagnosis, a high recall rate is crucial because it means correctly identifying individuals with a disease, even if it leads to some false alarms.|
|**Specificity**||$\frac{(TN)}{(TN + FP)}$|• Coverage of actual negative sample.|
|**F1 score**| is the harmonic mean of precision and recall, providing a balanced metric for situations where both precision and recall are important.|$\frac{1}{\frac{1}{precision}+\frac{1}{recall}}$<br/> ~ $\frac{2 x precision x recall}{precision + recall}$|• 1 score assesses the predictive skill of a model by elaborating on its class-wise performance rather than an overall performance as done by accuracy. F1 score combines two competing metrics- precision and recall scores of a model, leading to its widespread use in recent literature.<br/>• For example, in sentiment analysis of customer reviews, F1 score is a suitable metric when dealing with imbalanced sentiment classes. It helps strike a balance between correctly identifying positive and negative sentiment reviews, taking into account both precision and recall.|

> The higher the recall, the more positives the model finds
> The higher the precision, the less the model is wrong on the positives

### ROC
ROC The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold. These metrics are are summed up in the table below:
|Metric|Formula|Equivalent|
|---|---|---|
|True Positive Rate<br/>TPR|$\frac{TP}{(TP + FN)}$|Recall, sensitivity|
|False Positive Rate<br/>FPR|$\frac{TP}{(TN + FP)}$|1-specificity|

AUC-ROC (Area Under the Receiver Operating Characteristic) curve is a graphical representation that showcases the relationship between the true positive rate (TPR)(sensitivity) and the false positive rate (FPR) as the classification threshold varies.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9JT5mrixTcelpsk4yyTjgQ.jpeg" width="50%" height="50%" />

The AUC ranges from 0 to 1, where:

- AUC = 0.5 implies that the model’s performance is no better than random guessing.
- AUC > 0.5 and < 1 implies better-than-random performance, where higher values indicate better discrimination between classes.
- AUC = 1 implies that the model is a perfect classifier, meaning it achieves a TPR of 1 for all FPR values. This suggests that the model can completely separate the two classes without any errors.

- While precision, recall, and F1 score are threshold-specific, the AUC-ROC curve considers multiple thresholds simultaneously.

|Category|Task|Metric|Metric Summary|Reference|
|-----|-----|------|------|-----|
|-|Binary Classification|Confusion Matrix, Accuracy, Precision Recall and F1 Score|Shouldn’t use accuracy on imbalanced problems. Its easy to get a high accuracy score by simply classifying all observations as the majority class.|[Confusion Matrix, Accuracy, Precision, Recall, F1 Score](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd)[Beyond Accuracy: Recall, Precision, F1-Score, ROC-AUC](https://medium.com/@priyankads/beyond-accuracy-recall-precision-f1-score-roc-auc-6ef2ce097966)|
|LLM / NLP|Text Summary & Translation|ROGUE|Used for evaluating test summarization and machine translation. Metric compares an automatically produced summary or translation against human-produced summary or translation. It measures how many of the n-grams in the references are in the predicted candidate.|[An intro to ROUGE, and how to use it to evaluate summaries](https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/)|
|LLM / NLP|Text Summary & Translation|Perplexity|Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is. Perplexity is calculated as exponent of the loss obtained from the model. Perplexity is usually used only to determine how well a model has learned the **training set**. Other metrics like BLEU, ROUGE etc., are used on the **test set** to measure test performance.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|
|LLM / NLP|Text Summary & Translation|GLUE benchmark|GLUE benchmark that measures the general language understanding ability.|[Perplexity in Language Models](https://chiaracampagnola.io/2020/05/17/perplexity-in-language-models/)[Perplexity of Language Models](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)|

## Activation Functions

- Activation function decides whether a neuron should be activated or not.
- Activation functions transforms the weighted sum of a neuron so that the output is non-linear.

<img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d12431fbd5e61913b7423_60be4975a399c635d06ea853_hero_image_activation_func_dark.png" height="90%" width="90%">

### Vanishing Gradients Vs. Exploding Gradients

- [Gradient Descent From Scratch](https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc)

**Vanishing Gradient** occurs when the derivative or slope will get smaller and smaller as we go backward with every layer during backpropagation.

When weights update is very small or exponential small, the training time takes too much longer, and in the worst case, this may completely stop the neural network training.

**Exploding gradient** occurs when the derivatives or slope will get larger and larger as we go backward with every layer during backpropagation. This situation is the exact opposite of the vanishing gradients.

This problem happens because of weights, not because of the activation function. Due to high weight values, the derivatives will also higher so that the new weight varies a lot to the older weight, and the gradient will never converge. So it may result in oscillating around minima and never come to a global minima point.

||||
|---|---|---|
|Sigmoid or Logistic Activtion Function|It is generally used in logistic regression and binary classification models in the output layer.<br/>The output of sigmoid activation function lies between 0 and 1, making it perfect to model probability. Hence it is used to convert the real-valued output of a linear layer to a probability.<br/>Vanishing Gradient problem occurs with the sigmoid activation function because the derivatives of the sigmoid activation function are between 0 to 0.25.||
|Softmax|Extension of sigmoid activation function taking advantage of range of the output between 0 and 1. This is mainly used in the output layer of a multiclass, multinomial classification problem with a useful property of sum of the output probabilities adding up to 1.<br/>Softmax is used for multi-classification, the probabilities sum will be 1.||
|Tanh or hyperbolic tangent Activation Function|tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).<br/>A vanishing Gradient problem occurs with the tanh activation function because the derivatives of the tanh activation function are between 0–1. |
|ReLU (Rectified Linear Unit) Activation Function|The ReLU is the most used activation function in the world right now.Since, it is used in almost all the convolutional neural networks or deep learning.||
|Swish||
|GeLU||

## Loss Functions

In neural networks, loss functions help optimize the performance of the model. They are usually used to measure some penalty that the model incurs on its predictions, such as the deviation of the prediction away from the ground truth label.

All machine learning models are one optimization problem or another, the loss is the objective function to minimize. In neural networks, the optimization is done with gradient descent and backpropagation. [Source](https://machinelearningmastery.com/loss-functions-in-pytorch-models/)

### Loss functions for Regression

|Loss Function|Description|Formula|Notes|
|---|---|---|---|
|Mean Absolute Error (MAE) Or L1 Loss|measures the absolute difference between the true and predicted value||not sensitive to outliers and it is also not differentiable at zero.|
|Mean Squared Error (MSE)|This loss function handles outliers in an efficient manner as outliers are detected due to the quadratic loss.||Convergence is also smooth as the gradient becomes smaller as the loss decreases.|
|Root Mean Squared Error (RMSE)||||
|Root Mean Squared Logarithmic Error (RMSLE)|The root mean squared logarithmic error is determined by applying log to the actual and predicted numbers and then subtracting them. RMSLE is resistant to outliers when both minor and large errors are considered||The loss function is scale independent as it is a difference of two log values which is the same as log of the ratio of the values. Due to the loss being log it penalizes underestimates more than overestimates. Same as MSLE but the root mean square version of it.|
|Huber Loss|Huber loss is an ideal combination of quadratic and linear scoring algorithms||Huber loss is a combination of two loss functions quadratic and linear. The behaviour of the loss is defined by the value of the threshold. for loss values beyond the threshold the loss is linear else quadratic.|

**Others**

Quantile Loss, Log Cosh Loss and more ...

### Loss functions for classification

### Entropy
Entropy as a term is often used to measure the randomness in a given function/object.
Entropy in simple words is the element of surprise expressed mathematically.

|||||
|---|---|---|---|
|Log Loss (Or) Binary Cross-Entropy Loss|refers to the difference of randomness between two given features (or variables). The term keeps getting smaller as this difference decreases.||Less intuitive and may have many local minima.<br/>While applying this, the activation function in output layer must be SIGMOID.|
|Categorical cross entropy|This comes into picture when we have multiclass classification — number of classes becomes more than 2.|||
|Hinge Loss| Another commonly used Loss Function for classification problems — specially designed for SVM (support vector machines) classification algorithm (with labels as -1 and 1, not 0 and 1). It facilitates in finding the maximum margin of separation, from the hyperplanes to the respective classes.| | |

## Ensemble methods

Ensemble learning is a machine learning paradigm where multiple models (often called “weak learners”) are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

- **bagging**, that often considers homogeneous weak learners, learns them independently from each other in parallel and combines them following some kind of deterministic averaging process
- **boosting**, that often considers homogeneous weak learners, learns them sequentially in a very adaptative way (a base model depends on the previous ones) and combines them following a deterministic strategy
- **stacking**, that often considers heterogeneous weak learners, learns them in parallel and combines them by training a meta-model to output a prediction based on the different weak models predictions

Very roughly, we can say that bagging will mainly focus at getting an ensemble model with less variance than its components whereas boosting and stacking will mainly try to produce strong models less biased than their components (even if variance can also be reduced).

[Source](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)

### References

- https://sharkyun.medium.com/complete-guide-to-confusion-matrix-accuracy-precision-recall-and-f1-score-easy-to-understand-8772c2403df3
- https://datascience.stackexchange.com/questions/53870/how-do-gd-batch-gd-sgd-and-mini-batch-sgd-differ
- https://medium.com/@sasi24/cosine-similarity-vs-euclidean-distance-e5d9a9375fc8
- [High Bias and Variance problem in Machine Learning [Cause & Solution]](https://www.apdaga.com/2019/07/high-bias-and-variance-problem-in-machine-cause-solution.html)
