main.py contains functions for

1) Visualizing
2) Preprocessing
3) Model Construction
4) Result Validation

Three sample images are attached to illustrate the result of visualization.
Originally I was to use SVM and ROC/AUC curve but then I realized they are used
to do the labeling and classification task not quantatitive prediction...
So I switched to the linear regression approach. Many factors might
effect the price of housing(e.g. location, beds, size, etc). Based on the
visualization part I use # of beds and room size as indicators.

The test accuracy was around 60%. 
