<h1>kNN_Vs_Logistic_regression</h1>
This repo consists of 3 classification projects in completely different industries. 
<br>
<ui>
  <li>First project is about predicting the status of completion of tasks which will help HR department in better resource allocation</li>
  <li>In second project I am predicting the flights which might get delayed</li>
  <li>Where as the third project is creating a loan approval automation.</li>
</ui>
<br>
All the three datasets are alredy pretty much cleaned up, thus the main focus of this project is compairing the kNN and Logistic regression models. Datasets are received in CIS 508: Data Mining class (ASU-MSBA)<br><br>

* [1. Classification Models](#classification-models)
* [2. Performance Evaluation](#performance-evaluation)

<a class="anchor" id="classification-models"></a>
<b><h3><i>Classification Models under consideration:</i></h3></b>
<ol>
  <li>
      <b>kNN</b>
      <br><br>The k-Nearest Neighbours algorithm is a supervised machine learning algorithm which is used for both classification (discrete values) and regression questions (real values).<br>
      The kNN algorithm assumes that similar things exist in close proximity and this proximity is calculated by calculating the distance between two points (generally using Euclidean distance)<br><br>
      How kNN works:
      <br>(source: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)</br>
      <ul>
  <li>Load the data</li>          
  <li>Initialize K to your chosen number of neighbors</li>
  <li>For each example in the data calculate the distance between the query example and the current example from the data.</li>
  <li>Add the distance and the index of the example to an ordered collection</li>
  <li>Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances</li>
  <li>Pick the first K entries from the sorted collection</li>
  <li>Get the labels of the selected K entries</li>
  <li>If regression, return the mean of the K labels</li>
  <li>If classification, return the mode of the K labels</li>
      </ul><br>
          Choosing the right value of k:
      <ul>
  <li>Low values of k (say k = 1) will result in over-fitting</li>
  <li>High values of k will result in under-fitting</li>
      </ul>
  </li>
<br>
<br>
<li>
  <b>Logistic Model</b>
  <br>
  <br>
      Logistic regression is used for classification and predictive analysis. It is a supervised machine learning tool<br>
      It estimates the probability of an event occurring, based on given dataset of independent variables.
  <br>
  <br>
  There are 3 types of logistic regression models as mentioned below:<br>
      (_source: https://www.ibm.com/topics/logistic-regression)
  <br>
  <ul>
    <li>Binary logistic regression:<br>
      The resource allocation problem is of the binary type of logistic regression model, as the dependent variable  is dichotomous in nature, i.e the outcome can only be binary (0 or 1).<br>
      In this case, we are trying to predict whether the task has been completed (1) or if it has not been completed (0).<br>
      This type is generally most common for binary classification.<br>
    </li>
    <li>Multinomial logistic regression:<br>
      Here, the dependent variable has 3 or more possible outcomes, however there is no specified order.
    </li>
    <li>Ordinal logisitc regression:<br>
      This is similar to multinomial type where the dependent variable has 3 or more possible outcomes, however, here the values have a defined order.
    </li>
</li>
</ol>
<a class="anchor" id="performance-evaluation"></a>
<b><h3><i>Performance Evaluation of Model:</i></h3></b>
<ol>
  <li><b>Confusion Matrix</b><br><br>
      <img src = 'https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fa7fb3ffc-5c0e-4db5-89c6-021994823e01%2FUntitled.png&blockId=d5474d00-6501-48b7-a9a1-59d5bbb640d8'
           align = "center"
             width = "300"
             height = "300"/>
      <br><br>
      <img src = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/1200px-Precisionrecall.svg.png'
           width = "300"
           height = "500"/>
      <ul>
        <li>Recall/ True Positive Rate / Sensitivity: Sensitivity tells us what proportion of the positive class got correctly classified</li>
        <li>False Negative Rate: FNR tells us what proportion of the positive class got incorrectly classified by the classifier.</li>
        <li>True Negative Rate/ Specificity: Tells us what proportion of the negative class got correctly classified</li>  
        <li>False Positive Rate: FPR tells us what proportion of the negative class got incorrectly classified by the classifier</li>
      </ul>
      <br>
      Higher TPR and lower FNR is desirable since we want to correctly classify the positive class<br>
      Higher TNR and a lower FPR is desirable since we want to correctly classify the negative class.<br>
      <br>
      <img src = 'https://miro.medium.com/max/676/1*k6qWU7kXeCfk2KK2y3Cysg.png'/>
  </li>
  <li><b>ROC Curve</b><br><br>
    This is an evaluation metric for binary classification problems at various thresholds
    <ul>
      <li>ROC (Receiver Operating Characteristic) curve is a probability curve/ graph that shows the performance of a classification measure at all classification methods.</li>
      <li>This curve plots 2 parameters: True Positive Rate (recall) and False Positive Rate at different classification thresholds</li>
      <li>Lowering the classification threshold classifies more items as positive, thereby increasing both False Positives and True Positives</li>
      <li>The receiver operating characteristic (ROC) curve, which is defined as a plot of test sensitivity as the y coordinate versus its 1-specificity or false positive rate (FPR) as the x coordinate, is an effective method of evaluating the performance of diagnostic tests.</li>
    </ul>
    </li>
  <br>
  <li><b>AUC-ROC Curve</b><br><br>
      AUC is nothing but Area Under ROC Curve 
      The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes
      <ul>
        <li>For AUC = 1, classifier is able to distinguish between all the Positive and Negative class points correctly</li>
        <li>However if AUC = 0, classifier predicts all the Negatives as Positives and Positives as Negatives.</li>
        <li>When 0.5 < AUC < 1, there is a high chance that the classifier will be able to differentiate between the Positive class values from the Negative class values. This is so because the classifier is able to detect more number of True positives and True negatives than False positives and False negatives</li>
        <li>When AUC = 0.5, classifier is not able to distinguish between Positive and Negative class points. Meaning either the classifier is predicting random class or constant class for all the data points.</li>
    </ul>
  </li>
  </ol>
  
