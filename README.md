## Lyft Surge Predictor

### 1. Executive Summary

Our project was on predicting whether or not there’s a surge for Lyft rides. Ride companies such as Lyft have become an extremely popular way for transportation in places around the world. There is little known about surge for rides such as Lyft, so this prediction project helps us figure out paramaters/variables that the surge relies on. Comparing all the ROC curves on three graphs, we’ve found that the highest AUC to be 0.72 being Logistic Regression, second highest is 0.719 being Linear Regression , and the lowest is 0.53 being KNN. The method with the highest accuracy is Boosting at 0.9 and the lowest accuracy is 0.39 for Classification. This project would help both riders and drivers with picking optimal times to ride/drive knowing if there will be a surge.

### 2. Data Description

-	**Project Overview** <br>
We aim to identify clues to the underlying mechanism employed by ride-hailing services such as Uber and Lyft to suggest a price to the customer. Therefore, we will create classification models that will better predict whether or not there is a surge for a ride-hailing trip based on different factors and provide the option that doesn’t have a surge. We also aim to look at which factors influence the surge, as well as compare, and note differences between the services.

-	**Data Description** <br>
Our data is from Kaggle which includes data on cab information and weather. Here is the link to the original dataset https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma?select=rideshare_kaggle.csv. 
<br> The data includes information on Uber and Lyft rides.

-- **Dependent Variables**
