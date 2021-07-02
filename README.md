# Churn Prediction for Telecom Customers

## **Problem Statement :** 
In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal.
To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
In this project, we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

 ![alt text](https://nextommerce.com/wp-content/uploads/2019/04/pasted-image-0.png)

**Understanding and Defining Churn**

There are two main models of payment in the telecom industry 

- Postpaid (customers pay a monthly/annual bill after using the services)
- Prepaid (customers pay/recharge with a certain amount in advance and then use the services).

 ![alt text](https://keydifferences.com/wp-content/uploads/2015/11/prepaid-vs-postpaid-thumbnail.jpg)

In the postpaid model, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and you directly know that this is an instance of churn.

However, in the prepaid model, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).

Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘churn’ should be defined carefully.  Also, prepaid is the most common model in India and southeast Asia, while postpaid is more common in Europe and North America.

This project is based on the Indian and Southeast Asian market.

**Definitions of Churn**
There are various ways to define churn, such as:

**Revenue-based churn:**
 Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’. 

The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.

**Usage-based churn:** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time. 

A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.

In this project, you will use the usage-based definition to define churn.

**High-value Churn**
In the Indian and the southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.

In this project, you will define high-value customers based on a certain metric (mentioned later below) and predict churn only on high-value customers.

**Understanding the Business Objective and the Data**
The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. 

The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behavior during churn will be helpful.

**Understanding Customer Behaviour During Churn**
Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :

1. The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.

2. The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

3. The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.
 

In this case, since we are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.

### **Python Libraries Used** 

Here is a list of Python Libraries used along with their version numbers :

 |Library                         |   Version                         |
|-------------------------------|-----------------------------|
|`Numpy`            |1.18.1           |
|`pandas`|1.0.1|
|`matplotlib`|3.1.3|
|`seaborn`|0.10.0|
|`sklearn`|0.3.6|

### **Implementation in Python Notebook** 
### **Understanding & Cleaning  the Input Data** 

The dataset contain 99999 rows and 226 columns. 
1. We will delete 'mobile_number' and 'circle_id', since they dont contribute to information that the model could use.
2. We will  delete the date columns. We cannot derive any meaningful information from the date columns and hence dropping them from the model.
There are certain columns where the mean and all the other values are 0. They offer no information to the dataset. Hence we will remove them.
The number of columns have reduced from 226 to 201.
We will impute the missing values using the appropriate methods.
We will pick only the data that have greater than or equal to 70 percentile of the average rechange over the first two months. These will be the High Value Customers that we would be focusing on.
We will calculate the Churn ratio for the filtered dataset. The churn ratio is 9.25 %.

![bike3](https://lh5.googleusercontent.com/7LcytAcc9c_9ZvtRCMjnH8dS49t6payEIb1j3cP5nqlmqIu6PmdKaaeZLiSpiA1omYcVJZ0n77_24nWxs3lIwjahP62tJmJLkdrjLmgCaTMwEIsR1ybDtvYqlYj0TLF-b048958q)

 We will try to predict the results of the 9th month from the data from 6th, 7th and the 8th month.

For this it is important that we create some derived features.

**Feature Engineering** 

 Calculate the trend - 8th month data minus average of 6th and 7th month for both calls and 3G data

### **Visualising the Data** 

Understanding the correlation in the data :

![churn1](https://user-images.githubusercontent.com/10894854/124294072-ef280a00-db74-11eb-8210-a156f273414f.JPG))

We can see that the correlated variables are present in the dataset:

- Total reacharge amount trend and revenue per unit trend
- arpu 3g trend and arpu 2g trend
- avg recharge amount with arpu 2g and arpu 3g trend etc.

![churn2](https://user-images.githubusercontent.com/10894854/124294460-58a81880-db75-11eb-86c8-3d7a9b2abe76.JPG)

We can see from the above 3 plots that for the churned customers,

- Average Revenue per user drops significantly for the churn customers in the "action" phase (month Aug in our case) just before they leave the operator
- There is a sharper drop in the outgoing calls and incoming calls in the churn customers during the action phase than non-churn customers. It also appears that the drop is more significant in the outgoing calls than the incoming calls
- There is significant drop in Total Recharge Amount in the Action phase compared to Good phase
- The churned customers had less number of association year with the operator. So newer customers are churning more
- The number of voice calls (same network or outside) by the churn customers tend to be higher in the "Good" phase (more than the non-churn customers) and then drop significantly in the "action" phase. This might indicate they join the network due to a promo or short term deal and leave when that promo period expires
- Both the 2g and 3g services are less utilized by the churn customers in the action month. The churn customers prefer to use 3g packets (less than month validity) than monthly service schemes. Its the opposite for non churn customers who prefer monthly service schemes than packets

### **Model Building**

**Handling Class Imbalance**

Since this is a imbalanced class dataset with only 9.25% of the customers churning, we will use resample from Sklearn to make uniform distribution of churn records across samples.

We have built different kinds of models and here is a summary of the results :
![churn3](https://user-images.githubusercontent.com/10894854/124294923-e08e2280-db75-11eb-95bf-c7e5c583a434.JPG)

Plotting the output from few of the models :

*Decision Tree with Depth=3*

![churn5](https://user-images.githubusercontent.com/10894854/124296216-5646be00-db77-11eb-808e-57763b3178d7.JPG)

*Logitstic Regression with RFE*

![churn4](https://user-images.githubusercontent.com/10894854/124295341-5c886a80-db76-11eb-84ba-f0c9b711b1ed.JPG)

*Logistic Regression with PCA*

![churn6](https://user-images.githubusercontent.com/10894854/124296486-a32a9480-db77-11eb-876f-fc5d8193138e.JPG)

**Conclusions**
1. Among all the models, Random Forest provides the best recall rates (0.79). Since we are interested in increasing the True Positives and reducing the False Negatives, Recall is the metric we are focused on
2. The accuracy and the AUC scores of all models are very close to each other in the range of 0.83-0.91 so one model doesn't standout compared to others
3. The ROC-AUC score of all models are also very close to each other in the range of (0.56-0.64) so one model doesn't standout compared to others
4. We don't see a big difference in the final outcomes of the models where RFE was used for parameter reduction vs. PCA being used to reduce the dimensions/variables.
5. Hyper-parameter tuning on the non-interpret-able models did not yield a significant jump in the recall scores

The Logistic Regression model with RFE had the highest recall rate so we will use the final output of that model to make our recommendations

### **Recommendations**
I. Identify ways to detect a probable customer is going to churn during the action phase

- Any decline (measured by usage in current month minus average usage in prior 2 months) in the total outgoing minutes of a customer (total_og_mou) has a large contribution towards the odds of the customer churning next month
- Age on network (aon) is the second biggest contributor to the odds of the customer churning. This means that newer customers have a higher propensity to churn than the old customers
- A sharp decline (measured by current month incoming call usage minus average of prior 2 months) in the local incoming call usage (loc_ic_mou) is the third biggest contributor to the odds of the customer churning.

II. Incentivise the identified probable churn customers in the action month so that they dont churn

- Offer targeted incentives to these customers to continue with the operator. Since the outgoing and incoming call usage decline is one of the indicators of churn, these customers could be offered call related incentives than data related incentives
- Since the newer customers have a higher propensity to churn, target them with monthly recharges or postpaid deals so that they stick for a longer period
- The timing of the offer could be towards the later part of the action month which will ensure they will not get into churn phase

III. Continuous monitoring and model updates

- Continuously monitor the outcome of the incentives on the churn behavior and input the same into the model. If needed re-tune the hyper-parameters to make better predictions.
- In case of any significant event impacting churn, redevelop the model factoring the parameters describing the significant event.
