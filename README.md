# Classification-metrics
In this blog, we will be covering several classification metrics and advantages and disadvantages of each and so when to use each of them.

First let’s understand some terms-

**True Positive (TP):** When you predict an observation belongs to a class and it actually does belong to that class.

**True Negative (TN):** When you predict an observation does not belong to a class and it actually does not belong to that class.

**False Positive (FP):** When you predict an observation belongs to a class and it actually does not belong to that class.

**False Negative(FN):** When you predict an observation does not belong to a class and it actually does belong to that class.

All classification metrics work on these four terms. Let’s start classification metrics-

## Accuracy Score-

Classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of input samples.

![image](https://user-images.githubusercontent.com/65160713/131231123-0c5ff0e2-7cab-4f87-81a8-ccd6e28f04ef.png)

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

    Accuracy=(TP+TN)/(TP+TN+FP+FN)

It works well only if there are equal number of samples belonging to each class. For example, consider that there are 98% samples of class A and 2% samples of class B in our training set. Then our model can easily get 98% training accuracy by simply predicting every training sample belonging to class A. When the same model is tested on a test set with 60% samples of class A and 40% samples of class B, then the test accuracy would drop down to 60%. Classification Accuracy is great, but gives us the false sense of achieving high accuracy.

**So, you should use accuracy score only for class balanced data.**

You can use it by-

    from sklearn.metrics import accuracy_score

In sklearn, there is also _balanced_accuracy_score_ which works for imbalanced class data. The _balanced_accuracy_score_ function computes the balanced accuracy, which avoids inflated performance estimates on imbalanced datasets. It is the macro-average of recall scores per class or, equivalently, raw accuracy where each sample is weighted according to the inverse prevalence of its true class. Thus for balanced datasets, the score is equal to accuracy.

![image](https://user-images.githubusercontent.com/65160713/131231144-f907d2fb-a684-44d9-873b-a580fe08d384.png)

## Confusion matrix-

A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.

![image](https://user-images.githubusercontent.com/65160713/131231156-48ecc271-1002-4bb4-bf57-5cd889d36f3b.png)

It is extremely useful for measuring Recall, Precision, Specificity, Accuracy and most importantly AUC-ROC Curve.

    from sklearn.metrics import confusion_matrix

## Precision-

It is the ratio of the true positives and all the positives. It tells you that Out of all the positive classes we have predicted, how many are actually positive.

![image](https://user-images.githubusercontent.com/65160713/131231171-d3cdd389-6b12-43e7-94a4-a69e62f895e1.png)

    from sklearn.metrics import precision_score

## Recall(True Positive Rate)-

It tells you that Out of all the positive classes, how much we predicted correctly. It should be high as possible. It is also called as sensitivity.

![image](https://user-images.githubusercontent.com/65160713/131231182-11d14f3c-1445-4f8f-aa34-0880ff1e7af3.png)

    from sklearn.metrics import recall_score

## F-Score-

It is difficult to compare two models with low precision and high recall or vice versa. So to make them comparable, we use F-Score . F-score helps to measure Recall and Precision at the same time.

![image](https://user-images.githubusercontent.com/65160713/131231189-768a26c3-2f5b-4f1d-ab65-e94c54c77186.png)

It uses Harmonic Mean in place of Arithmetic Mean by punishing the extreme values more.

    from sklearn.metrics import f1_score

We use it when we have imbalanced class data. In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on than accuracy.

But it is Less interpretable. Precision and recall is more interpretable than _f1-score_, since it measures the type-1 error and type-2 error. However, _f1-score_ measures the trade-off between this two. So instead of working with both and confusing ourselves, we use _f1-score_.

**Specificity(True Negative Rate):** It tells you what fraction of all negative samples are correctly predicted as negative by the classifier.. To calculate specificity, use the following formula.

![image](https://user-images.githubusercontent.com/65160713/131231204-30fba613-d008-445c-90d7-bf4a94dff89b.png)

**False Positive Rate:** FPR tells us what proportion of the negative class got incorrectly classified by the classifier.

![image](https://user-images.githubusercontent.com/65160713/131231213-96559f5a-a2a3-4e91-abaa-8664ad174021.png)

**False Negative Rate:** False Negative Rate (FNR) tells us what proportion of the positive class got incorrectly classified by the classifier.

![image](https://user-images.githubusercontent.com/65160713/131231228-535349b3-28b5-472a-9159-dec3cf784bef.png)

## ROC-AUC curve-

Not only numerical metrics, we also have plot metrics like ROC(Receiver Characteristic Operator) and AUC(Area Under the Curve) curve.

AUC — ROC curve is a performance measurement for the classification problems at various threshold settings. This graph is plotted between true positive and false positive rates The **area under the curve** (AUC) is the summary of this curve that tells about how good a model is when we talk about its ability to generalize.

If any model captures more AUC than other models then it is considered to be a good model among all or we can conclude more the AUC the better model will be classifying actual positive and actual negative.

If the value of AUC = 1 then the model will be perfect while classifying the positive class as the positive and negative class as negative. If the value of AUC = 0, then the model is poor while classifying the same. The model will predict positive as negative and negative as positive. If the value is 0.5 then the model will struggle to differentiate between positive and negative classes. If it’s between 0.5 and 1 then there are more chances that the model will be able to differentiate positive class values from the negative class values.

Let’s take a predictive model for example. Say, we are building a logistic regression model to detect whether breast cancer is malignant or benign. A model that returns probability of 0.8 for a particular patient, that means the patient is more likely to have malignant breast cancer. On the other hand, another patient with a prediction score of 0.2 on that same logistic regression model is very likely not to have malignant breast cancer. Then, what about a patient with a prediction score of 0.6? In this scenario, we must define a classification threshold to map the logistic regression values into binary categories. By default, the logistic regression model assumes the classification threshold to be 0.5, but thresholds are completely problem-dependent. In order to achieve the desired output, we can tune the threshold. But now the question is how do we tune the threshold?

For different threshold values we will get different TPR and FPR. So, in order to visualize which threshold is best suited for the classifier we plot the ROC curve. The following figure shows what a typical ROC curve look like.

![image](https://user-images.githubusercontent.com/65160713/131231237-8540de68-9661-4bb4-b974-39a413ddb03c.png)

The ROC curve of a random classifier with the random performance level (as shown below) always shows a straight line. This random classifier ROC curve is considered to be the baseline for measuring the performance of a classifier. Two areas separated by this ROC curve indicates an estimation of the performance level — good or poor.

The ROC curve of a random classifier with the random performance level (as shown below) always shows a straight line. This random classifier ROC curve is considered to be the baseline for measuring the performance of a classifier. Two areas separated by this ROC curve indicates an estimation of the performance level — good or poor.

ROC curves that fall under the area at the top-left corner indicate good performance levels, whereas ROC curves fall in the other area at the bottom-right corner indicate poor performance levels. An ROC curve of a perfect classifier is a combination of two straight lines both moving away from the baseline towards the top-left corner.

Smaller values on the x-axis of the plot indicate lower false positives and higher true negatives. Larger values on the y-axis of the plot indicate higher true positives and lower false negatives.

Although the theoretical range of the AUC ROC curve score is between 0 and 1, the actual scores of meaningful classifiers are greater than 0.5, which is the AUC ROC curve score of a random classifier. The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 — FPR).

Note that many discrete classifiers can be converted to a scoring classifier by ‘looking inside’ their instance statistics. For example, a decision tree determines the class of a leaf node from the proportion of instances at the node.

    from sklearn.metrics import roc_curve, auc

Sometimes, we remove the y-axis by precision and then the plot is called as precision recall curve which does the same thing(calculates the value of precision and recall at different thresholds). But it is restricted only to binary classification in sklearn.

    from sklearn.metrics import precision_recall_curve

## Extending the above to multiclass classification-

![image](https://user-images.githubusercontent.com/65160713/131231278-ddefc678-9b47-42d9-a924-4899d6c3048a.png)

So in confusion matrix for multiclass, we don’t use TP,FP,FN and TN. We just use predicted classes on y-axis and actual classes on x-axis. In above figure, (1,1) denotes how many classes were facebook and actually predicted facebook and (1,2) denotes how many classes were instagram but predicted as facebook.

The true positive, true negative, false positive and false negative for each class would be calculated by adding the cell values as follows:

![image](https://user-images.githubusercontent.com/65160713/131231286-ff7380f0-15fe-4ca2-86cf-c0e9c0d23b91.png)

Precision and recall scores and F-1 scores can also be defined in the multi-class setting. Here, the metrics can be “averaged” across all the classes in many possible ways. Some of them are:

**micro:** Calculate metrics globally by counting the total number of times each class was correctly predicted and incorrectly predicted.

**macro:** Calculate metrics for each “class” independently, and find their unweighted mean. This does not take label imbalance into account.

**None:** Return the accuracy score for each class corresponding to each class.

ROC curves are typically used in binary classification to study the output of a classifier. To extend them, you have to convert your problem into binary by using OneVsAll approach, so you'll have _n_class_ number of ROC curves.

In sklearn there is also classification_report which gives summary of _precision_, _recall_ and _f1-score_ for each class. It also gives a parameter support which just tells the occurence of that class in the dataset.

    from sklearn.metrics import classification_report
