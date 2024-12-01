# ML System Design Interview 

### Build a machine learning model to predict if an _ad will be clicked_.

Refer to, [Ad Click Prediction | Machine learning system design](https://medium.com/@mumbaiyachori/ad-click-prediction-machine-learning-system-design-6e553d7ccc1c)

### Design food delivery ETA prediction ML Model for Uber Eats/Doordash.

[ML System Design — ETA Prediction](https://mecha-mind.medium.com/ml-system-design-eta-prediction-9dc8000fd86b)

### Design a ML System for language translation of post/comments for Facebook/LinkedIn news feed.

[ML System Design — Language Translation](https://mecha-mind.medium.com/ml-system-design-language-translation-290eac2fb650)

### How to Build a Customer Churn Prediction Model in Python? 

- [How to Build a Customer Churn Prediction Model in Python? ](https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/)

###  Design a feed recommendation system

- [Recommender Systems — A Complete Guide to Machine Learning Models](https://towardsdatascience.com/recommender-systems-a-complete-guide-to-machine-learning-models-96d3f94ea748)

### Design Youtube(Google)
### Design Google contact ranking(Google)
### Design an item replacement recommendation(Instacart)
### Design an ML System to optimize coupon distribution with a set budget(Netflix)

### Detecting unusual spend with AWS Cost Anomaly Detection

AWS Cost Anomaly Detection is an AWS Cost Management feature. This feature uses machine learning models to detect and alert on anomalous spend patterns in your deployed AWS services. WS Cost Anomaly Detection runs approximately three times a day in order to monitor for anomalies in your net unblended cost data.

https://docs.aws.amazon.com/cost-management/latest/userguide/manage-ad.html

### What are the metrics for search ranking?

## Build a movie/show ML recommendation system.

[ML system design interview-Recommendation System](https://yunrui-li.medium.com/ml-system-design-interview-recommendation-system-637df3a31eb0)

## Imagine you’re building a system to recommend users items similar to those they’ve bought. How would you go about building this?

- Item-item similarity matrix: Create an item-item similarity matrix to measure the similarity between pairs of items. You can use cosine similarity or Jaccard similarity methods to compute the similarity scores.
- Item-based recommendation: Once the item-item similarity matrix is built, recommend items to users based on the items they have bought. For each item a user has purchased, find the most similar items and recommend those to the user.
- User-item interaction matrix: Alternatively, you can build a user-item interaction matrix that reflects the relationship between users and items. The matrix can contain information such as whether a user has bought an item, viewed it, or added it to their cart.
- User-based recommendation: Based on the user-item interaction matrix, you can recommend items to users by finding similar users and recommending items that similar users have purchased.
- Hybrid recommendation: Combine item- and user-based recommendations to create a hybrid recommendation system. This can provide better recommendations by considering both the items a user has bought and the behavior of similar users.
- Model evaluation: Evaluate the performance of the recommendation system using metrics such as accuracy, precision, recall, and F1-score. Iteratively improve the system by trying different algorithms, adjusting parameters, and incorporating user feedback.

- Content-based recommendations: Using the features of the items the user has interacted with in the past, such as genre, keywords, or other metadata, to make recommendations.
- Popularity-based recommendations: Recommend the most popular items in the product catalog, regardless of the user’s preferences.
- Hybrid approaches: Combine the outputs of multiple recommendation models to provide recommendations.

For that recommender, how would you handle a new user who hasn’t made any past purchases?

For a new user who hasn’t made any past purchases, there are several ways to handle them in a recommender system:

- Cold start: One approach is to collect more information about the new user, such as demographic information, browsing history, or preferences. This information can then be used to generate recommendations.
- Popularity-based recommendations: Another approach is to make recommendations based on popular items in the system, as this information is easily accessible and doesn’t require user-specific data.
- Clustering: Another approach is to cluster similar users based on demographic information or browsing history and use the cluster to make recommendations.
- Matrix Factorization: This approach decomposes the user-item matrix into two matrices, one representing users and the other items. This can be used to make recommendations based on similar users or items.

### Stock Price Prediction and Time Series Analysis

Stock prices fluctuate over time, understanding their patterns and trends becomes crucial for making informed investment decisions. We will explore various techniques, such as moving averages, linear regression, seasonal plots, and lag embeddings, to model and predict stock prices.

### What is a Marketing Attribution Platform?

a Marketing Attribution Platform is a tool that allows you to track the effectiveness of your marketing campaigns. It does this by tracking the user’s journey from the first time they visit your site to the point of conversion.
