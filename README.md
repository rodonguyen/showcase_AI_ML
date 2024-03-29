# **My AI/ML Projects** 😉

Some of my most hightlight AI/ML projects are present in this repo.  
A report is included in each project too.

<br>

## **1. Twitter Spam Detection -** [Click to explore](/1.%20Twitter%20Spam%20Detection/)

- **Able to identify spam tweets with >95% accuracy** using 4 Machine Learning methods: LSTM, Transformer, SVM and Naive Bayes
- Incorporated more useful tweet metadata by expanding existing model design, leading to better learning and thus higher accuracy
- Attained insightful conclusions of the model behaviours through rigorous analysis on failed cases
- Evaluated each method’s suitability on real world application in terms of training & inference time, scalability and computational demand. Naive Bayes has shown to be the best model.

    <center>
    <figure><img src="public/images/twitter_confusion_matrices.png" alt="Figure: Prediction results in condusion matrices" style="width:500px"><figcaption align="center"><br>Figure: Prediction results in condusion matrices</figcaption></figure>
    <figure><img src="public/images/twitter_overall_performance.png" alt="Figure: Overall Performance of 4 methods" style="width:500px"><figcaption align="center"><br>Figure: Overall Performance of 4 methods</figcaption></figure>
    </center>
    

<br>

## **2. Deep Convolutional Neural Network Applications -** [Click to explore](/2.%20Deep%20Convolutional%20Neural%20Network%20Applications/)

a) Number Classification
- Designed a **Neural Network** following VGG style to regconise numbers from Street View House Numbers dataset. This project removes a large part of the data to simulate data shortage.
- 3 approaches were used to train, test and evaluate the model:
    - Train from scratch
    - Apply **data augmentation** to combat data shortage
    - **Transfer learning** + Data augmentation
    <center>
    <figure><img src="public/images/dcnn_identifying_numbers.png" alt="Figure: Performance of 3 approaches, plus a non-deep method - SVM" style="width:600px"><figcaption align="center"><br>Figure: Performance of 3 approaches, plus a non-deep method (SVM). This helps us compare the pros and cons of utilising different models to do the same tasks. </figcaption></figure>
    </center>
    <!-- <details>
    <summary><b>View cool figures! 👈</b></summary>
    </details> -->

b) Person Re-identification
- Implemented **Siamese model** and **Triplet Loss Function** for its advantage in focusing on increasing/decreasing distance between matching/mismatching data to re-identify person.
- Top-10 Accuracy reached 64% (i.e. subject is matched in the 10 most closest results with 64% success rate) and Top-1 Accuracy reached 25%. This is an impressive result given the simple model and low-quality dataset.

    <center>
    <figure><img src="public/images/dcnn_cmc_curve.png" alt="Figure: Cumulative Matching Characteristics (CMC) curve of the Siamese Model" style="width:400px"><figcaption align="center"><br>Figure: Cumulative Matching Characteristics (CMC) curve of the Siamese Model. Based on this curve, we can see if the model is good (the more top left it bends to, the better) </figcaption></figure>
    </center>

<br>

## **3. Movie Recommendation -** [Click to explore](/3.%20Movie%20recommendation%20-%20Clustering/)
- Uses Gaussian Mixture Model (GMM) out of other methods like HAC, DBScan, K-Means because of its scalability and probability assignment.
- The number of K cluster is chosen based on Bayesian Information Criterion which also takes model complexity into consideration
- For the result, suggested movies are closely related to user's watched ones and no obviously unrelated recommendation is detected

<br>

## **4. Regression & Classification -** [Click to explore](/4.%20Regression%20%26%20Classification/)

a) Regression
- Predicting crime percentage per population in a suburb by using (Linear Regression, Ridge, Lasso). 
- Analysis from correlation heat map **below** shows many variables are redundent and measuring similar subjects. Data are standardised before fitting the model to set all columns' values to the same scale. Thus enhance accuracy and avoid bias.

    <center>
    <figure><img src="public/images/regression_corr_heatmap.png" alt="igure: Data correlation heatmap" style="width:500px"><figcaption align="center"><br>Figure: Data correlation heatmap</figcaption></figure>
    <br>
    <figure><img src="public/images/regression_linear_regression.png" alt="Figure: Linear Regression result and its prediction" style="width:500px"><figcaption align="center"><br>Figure: Linear Regression result and its prediction</figcaption></figure>
    </center>



b) Classification
- Applied and Compare performance of 3 different classifiers (SVM, CKNN, Random Forest) in identifying 4 different forest types.
- Process also includes data split, finding the best params from Validation data using GridSearchCV and finally evaluate result on Test data.

    <center>
    <figure><img src="public/images/classification_corr_heatmap.png" alt="Figure: Data correlation heatmap" style="width:500px"><figcaption align="center"><br>Figure: Data correlation heatmap. Similar to the one above, it helps us to identify redundent/unhelpful predictors and from here, we formulate a suitable solution.</figcaption></figure>
    <br>
    <figure><img src="public/images/classification_svm_best.png" alt="Figure: Confusion matrices of SVM prediction" style="width:500px"><figcaption align="center"><br>Figure: Confusion matrices of SVM prediction. Good for model performance evaluation for individual class.</figcaption></figure>
    </center>

