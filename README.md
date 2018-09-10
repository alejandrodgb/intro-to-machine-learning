# Notes
## Bayes Rule
Prior probability * test evidence ->  posterior probability

Taking some prior probability and updating it with evidence to arrive at a posterior probability

### Cancer Example
Prior: 
- P(C) = 0.01; P(nC) = 0.99
- sensitivity: P(Pos|C) = 0.9; P(Neg|C) = 0.1
- specificity: P(Neg|nC) = 0.9; P(Pos|nC) = 0.1

Joint Probability:
- P(C|Pos) = P(C) * P(Pos|C) = 0.009
- P(nC|Pos) = p(nC) * P(Pos|nC) = 0.099

Normalisation:
- P(Pos) = P(Pos,C) + P(Pos,nC) = 0.108

Posterior Probability:
- P(C|Pos) = 0.083
- P(nC|Pos) = 0.917

### Bayes Rule for Classification - Text Learning
**Priors**
| P(Word) | Chris | Sarah |
|---------|-------|-------|
| Love | 0.1 | 0.5 |
| Deal | 0.8 | 0.2 |
| Life | 0.1 | 0.3 |

**Who wrote it?: Love Life!**

- P(Chris) * P(Love|Chris) * P(Life|Chris) = 0.5 * 0.1 * 0.1 = 0.005
- P(Sarah) * P(Love|Sarah) * P(Life|Sarah) = 0.5 * 0.3 * 0.3 = 
0.045

**Who wrote it? Life Deal**

- P(Chris) * P(Life|Chris) * P(Deal|Chris) = 0.5 * 0.1 * 0.8 = 0.04
- P(Sarah) * P(Life|Sarah) * P(Deal|Sarah) = 0.5 * 0.3 * 0.2 = 
0.03

What is the posterior probability?

- P(Chris|Life Deal)?
- P(Sarah|Life Deal)?

Normalisation

P(Life Deal) = P(LifeDeal,Chris) + P(LifeDeal,Sarah) = 0.07

- P(Chris|Life Deal) = 0.04/0.07 = 0.57
- P(Sarah|Life Deal) = 0.03/0.07 = 0.43

**Who Wrote it? Love Deal**

- P(Chris) * P(Love|Chris) * P(Deal|Chris) = 0.5 * 0.1 * 0.8 = 0.04
- P(Sarah) * P(Love|Sarah) * P(Deal|Sarah) = 0.5 * 0.5 * 0.2 = 
0.05

What is the posterior probability?
- P(Chris|Love Deal)?
- P(Sarah|Love Deal)?

Normalisation
- P(Love Deal) = P(LoveDeal,Chris) + P(LoveDeal,Sarah) = 0.09

- P(Chris|Life Deal) = 0.04/0.09 = 0.44
- P(Sarah|Life Deal) = 0.03/0.09 = 0.56

### Strengths and Weaknesses
<table>
    <tr>
        <td>Strengths</td>
        <td>Weaknessess</td>
    </tr>
    <tr>
        <td>
            <ul>
                <li>Easy to implement</li>
                <li>Deals with big feature spaces</li>
                <li>Simple to run, efficient</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>Phrases that encompass multiple words do not work well with Naive Bayes</li>
            </ul>
        </td>
    </tr>
</table>

## SVM
SVM = Support Vector Machines

Support vector machines try to find a decision line where there is as much separation between the two classes as possible - maximum margin.

### Parameters
Arguments passed when creating the classifier.

Parameters for an SVM (not exhaustive list):
- kernel: Function that maps a low-dimentional feature space and map it to a high dimensional space. This makes non-linearly separable data, linearly separable.
  - Process of the kernel trick: x,y (non separable) > X<sub>1</sub>X<sub>2</sub>X<sub>3</sub>X<sub>4</sub> (solution) > back to x,y with a non-linear separation
- C: controls tradeoff between smooth decision boundary and classifying training points correctly. Large C will classify more training points correctly.
- gamma: defines how far the influence of a single training example reaches.
  - Low values - far reach: points further away from the decision bondary are taken into consideration when creating the decision boundary.
  - High values - close reach: points close to the decision boundary are taken into consideration when creating the decision boundary. This might cause the model to ignore points further away.

### Strengths and Weaknesses
<table>
    <tr>
        <td>Strengths</td>
        <td>Weaknessess</td>
    </tr>
    <tr>
        <td>
            <ul>
                <li>Work well in complicated domains with a clear margin of separation</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>Don't perform well in large training datasets because training time = (data set size)<sup>3</sup></li>
                <li>Don't work well with noice, so when the classes are overlapping you need to count independent evidence. Naive Bayes works better.</li>
            </ul>
        </td>
    </tr>
</table>

# Datasets and References
- Sentiment classification using Naive Bayes: https://www.kaggle.com/marklvl/sentiment-classification-using-naive-bayes/notebook
- Fraud Detection with Naive Bayes Classifier: https://www.kaggle.com/lovedeepsaini/fraud-detection-with-naive-bayes-classifier

#To do
- Create a plotter for decision boundary and data points

