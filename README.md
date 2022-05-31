## Lyft Surge Predictor

### 1. Executive Summary

Our project was on predicting whether or not there’s a surge for Lyft rides. Ride companies such as Lyft have become an extremely popular way for transportation in places around the world. There is little known about surge for rides such as Lyft, so this prediction project helps us figure out paramaters/variables that the surge relies on. Comparing all the ROC curves on three graphs, we’ve found that the highest AUC to be 0.72 being Logistic Regression, second highest is 0.719 being Linear Regression , and the lowest is 0.53 being KNN. The method with the highest accuracy is Boosting at 0.9 and the lowest accuracy is 0.39 for Classification. This project would help both riders and drivers with picking optimal times to ride/drive knowing if there will be a surge.

### 2. Data Description

-	**Project Overview** <br>
We aim to identify clues to the underlying mechanism employed by ride-hailing services such as Uber and Lyft to suggest a price to the customer. Therefore, we will create classification models that will better predict whether or not there is a surge for a ride-hailing trip based on different factors and provide the option that doesn’t have a surge. We also aim to look at which factors influence the surge, as well as compare, and note differences between the services.

-	**Data Description** <br>
Our data is from Kaggle which includes data on cab information and weather. Here is the link to the original dataset https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma?select=rideshare_kaggle.csv. 
<br> The data includes information on Uber and Lyft rides.
    - Dependent Variables <br>
We use all the variables as predictors except surge, id, timestamp, time zone, hour, day, month, name, product id, and some of the weather variables. The predictors we hypothesize that can have an influence on the surge are weekday dummy, rush hour dummy, source of ride , destination of ride, type of ride (XL, shared, etc), distance and weather-related variables (temperature and precipitation intensity). The weekday and rush hour dummy variables were created from day and hour variables on if it’s a weekday versus weekend and if the ride is during rush hour. <br>
We have also added a binary independent variable based on the given data to facilitate the development of our classification models.

    - Independent Variables <br>
    We used the surge multiplier as the independent variable. We created a new binary variable called new surge multiplier to see if the surge multiplier is greater than 1 for the ride. If the surge is greater than 1 than there is a surge and if it’s equal to 1 than there isn’t a surge.
    
### 3. Exploratory Data Analysis

-	**Issues in the data set** <br>
When data is broken down to “product” (Uber or Lyft), Uber does not have any surge rides. As our project objective is to predict a surge in the fare, we have decided to filter out Uber data and focus on Lyft data.

![](https://ppt.cc/fJIYwx@.png)

As for the Lyft data, there is a serious imbalance between the ratio of surged and non-surged rides in the Lyft data. The surged rides account for less than 2% of the total Lyft data. This imbalance means that there is not a large enough or representative sample of observations from the minority class (surged rides). We will need to address this issue before training our classification models. <br>

From this point on, our report will only discuss the Lyft data set. <br>

Another issue is the short time frame that the Lyft data covers. Throughout the span of three weeks, there is a lack of data on nine days in December. This affects the accuracy of how we analyze demand at different hours of the day and on different days of the week.

![](https://ppt.cc/f5hWZx@.png)

-	**Correlations Among Variables** <br>
After the data processing step, there are 18 variables in the Lyft data set. Firstly, we need to explore the correlations among variables to identify any strong relationships among them and have initial hypotheses about the factors that decide the surge and the price of rides.

-	**Price** <br>
Based on the heatmap, “price” is strongly correlated with “base fare”, “cost per minute”, and “cost per minimum fare”. This is predictable as those variables are the main elements of the price of a ride. Unsurprisingly, “price” is also positively related to “distance” and “surge multiplier”.

![](https://ppt.cc/f8Bgbx@.png)

-	**Cab Type** <br>
The boxplot shows that Lyft Shared has the lowest median price, followed by Lyft. Lyft Lux Black XL has the highest number of outliers.

![](https://ppt.cc/fJ1iUx@.png)

-	**Ride time** <br>
From 9 am to 6 pm or on Wednesday and Thursday, there are more surge rides than average.

![](https://ppt.cc/fTttOx@.png)
![](https://ppt.cc/faSFix@.png)

-	**Weather** <br>
When it is raining or going to rain, there are more surge rides than average.

![](https://ppt.cc/fNEiHx@.png)

### 4. Research Question
Ride-sourcing companies like Lyft have become an extremely popular mode of transportation worldwide in the 21st century. With the COVID-19 pandemic, Lyft has still been providing a safe transportation option to those who want to avoid public transportation. However, while public transportation systems have a fixed pricing strategy, these companies do not. Lyft uses surging in real time to help balance the supply and demand. We decided to predict if there would be a surge and analyze it using classification models. <br>

This dataset was chosen because it had data on many different variables that we thought could help in predicting if a surge was evident. Though the popularity in Lyft is constantly increasing, there isn’t much known about their surge and surge pricing. The goal of our project is to identify ways incorporated by Lyft in identifying whether or not there's a surge to the customers. Something to keep in mind with our dataset is that we had to oversample our data due to an imbalance in the Surge Multiplier. By doing so, we were able to build our classification models.

### 5. Methodology
For our research question on predicting the Lyft surge, we decided to convert the Surge Multiplier to a dummy variable called New Surge Multiplier. For this variable, if the surge is over 1 then the New Surge Multiplier is 1 meaning that there is a surge and if it’s 1 then the New Surge Multiplier is 0 meaning that there isn’t a surge. Since we created this dummy variable and decided to focus on predicting this variable, we wanted to use classification methods. <br>

The classification methods we used include ensemble methods, bagging, boosting, K nearest neighbors, naive bayes, classification tree, linear probability method, and logistic regression. We wanted to compare different classification models to find the ones with the best accuracies, AUC, sensitivity, and specificity that produce the best prediction results. 

### 6. Results & Findings

-	**Linear Probability Model** <br>
Linear Probability Model is usually not a best choice for classification because it could result in the probability greater than 1 or less than 0, and is often affected by the outliers. However, as the basic model, it is a good choice to treat such model as the baseline. When we run this model, the result shows that the coefficient of “destination” is NA and the predict function has the warning "prediction from a rank-deficient fit may be misleading". These imply that our data has the perfect multicollinearity even though we have checked the correlation between features before. Therefore, we exclude the feature “destination” and then everything works well. <br>

To assure that there are no other high correlated features in the data, we compute the Variance Inflation Factor (VIF).  

![](https://ppt.cc/fNK11x@.png)

Due to the existence of categorical variables, R computes the GVIF ^ (1 / (2 * DF)) instead of VIF. Typically, if the value is greater than 10, there would be problems. As the table above shows, all the variables except precipProbability have very low value of GVIF ^ (1 / (2 * DF)). Even the value of precipProbability is lower than 10, so we keep all the variables. 
