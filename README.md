Interview Project For WeHome

To predict the price of a new housing source, we need to follow the following four steps..

1) Visualizing data using scatter plot with different varaible vs. price (e.g. location vs. price, # of beds vs. price, etc) to determine the most correlated factors

- I plotted every varaibles and it turns out that # of beds and size of the room are the two most correlated factors to price
- I use airbnb_rent column as the price data since the rent column is mostly empty
- For simplicity, I only attached four plots as demo

2) Preprocessing and cleaning the data (e.g. discard any \\NA and nan values in both dependent and independent variables)

There are many ways to preprocess the data...you can fill them manually..or construct a model to predict those missing values...I discard the missing ones because we still have enough (6311) samples after all the discards

3) Model Construction (Split data into train and test sets)

Originally I was to use SVM to solve this and ROC/AUC curve to measure the performance but then I realized they are designed for non-quantititive classifcation. Therefore I switched to a Linear Regression appraoch. I used Sklearn library's LinearRegression

4) Output the model statistics (e.g. both test and train accuracy)

Both the testing and training accuracy are around 60%. 

5) Model Improvement

Possibly there are other factors that play a role in determining the rent price. Futher research and effort might be needed to improve the accuracy to around 80%.

Please let me know if you have any other questions

