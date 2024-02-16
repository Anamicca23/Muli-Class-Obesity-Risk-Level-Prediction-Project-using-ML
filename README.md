## **<span style="color:red">COMPREHENSIVE ANALYSIS AND PREDICTION OF OBESITY RISK LEVELS USING MACHINE LEARNING TECHNIQUES WITH - (100)% ACCURACY</span>**
**Author**: **Anamika Kumari**

## Introduction:
<summary><b><span style="color:blue">What is Obesity:</span></b></summary>

<p>Obesity is a complex health condition affecting millions globally, with significant implications for morbidity, mortality, and healthcare costs.Obesity is a global concern, with statistics indicating a significant rise in the number of obese individuals, now accounting for approximately 30% of the global population, triple the figures from 1975. This escalating trend highlights the pressing need to address the multifaceted risks associated with excess weight. Obesity is a major contributor to various health complications, including diabetes, heart disease, osteoarthritis, sleep apnea, strokes, and high blood pressure, thereby significantly reducing life expectancy and increasing mortality rates. Effective prediction of obesity risk is crucial for implementing targeted interventions and promoting public health.</p>

<p>In this project, we undertake a comprehensive analysis to predict obesity risk levels using advanced machine learning techniques.</p>

<img src="https://www.limarp.com/wp-content/uploads/2023/02/obesity-risk-factors.png" alt="Obesity-Risk-Factors" width="1500">



## Table of Contents:
​

<details>
<summary><strong>Section: 1. Introduction</strong></summary>

| No. | Topic                                      |
|-----|--------------------------------------------|
| 1.  | [What is Obesity?](#What-is-Obesity:)      |
| 2.  | [Understanding Obesity and Risk Prediction](#Understanding-Obesity-and-Risk-Prediction:)|
| 3.  | [Dataset Overview](#Dataset-Overview:)     |



</details>

<details>
<summary><strong>Section: 2. Importing Libraries and Dataset</strong></summary>

| No. | Topic                                |
|-----|--------------------------------------|
| 1.  | [Importing Relevant Libraries](#Importing-Relevant-Libraries:) |
| 2.  | [Loading Datasets](#Loading-Datasets:)                        |


</details>

<details>
<summary><strong>Section: 3. Descriptive Analysis</strong></summary>

| No. | Topic                                                               |
|-----|---------------------------------------------------------------------|
| 1.  | [Summary Statistic of dataframe](#1.-Summary-Statistic-of-dataframe:) |
| 2.  | [The unique values present in dataset](#2.-The-unique-values-present-in-dataset:) |
| 3.  | [The count of unique value in the NObeyesdad column](#3.-The-count-of-unique-value-in-the-NObeyesdad-column:) |
| 4.  | [Categorical and numerical Variables Analysis](#4.-Categorical-and-numerical-Variables-Analysis:) |
|     |   - [a. Extracting column names for categorical, numerical, and categorical but cardinal variables](#a.-Extracting-column-names-for-categorical,-numerical,-and-categorical-but-cardinal-variables:) |
|     |   - [b. Summary Of All Categorical Variables](#b.-Summary-Of-All-Categorical-Variables:) |
|     |   - [c. Summary Of All Numerical Variables](#c.-Summary-Of-All-Numerical-Variables:) |


</details>

<details>
<summary><strong>Section: 4. Data Preprocessing</strong></summary>

| No. | Topic                                                      |
|-----|------------------------------------------------------------|
| 1.  | [Typeconversion of dataframe](#1.-Typeconversion-of-dataframe:) |
| 2.  | [Renaming the Columns](#2.-Renaming-the-Columns:)          |
| 3.  | [Detecting Columns with Large or Infinite Values](#3.-Detecting-Columns-with-Large-or-Infinite-Values:) |


</details>

<details>
<summary><strong>Section: 5. Exploratory Data Analysis and Visualization-EDAV</strong></summary>

<details>
<summary><strong>1. Univariate Analysis</strong></summary>
  
| No. | Topic                                                      |
|-----|------------------------------------------------------------|
| a.  | [Countplots for all Variables](#a.-Countplots-for-all-Variables:) |
| b.  | [Analyzing Individual Variables Using Histogram](#b.-Analyzing-Individual-Variables-Using-Histogram:) |
| c.  | [KDE Plots of Numerical Columns](#c.-KDE-Plots-of-Numerical-Columns:) |
| d.  | [Pie Chart and Barplot for categorical variables](#d.-Pie-Chart-and-Barplot-for-categorical-variables:) |
| e.  | [Violin Plot and Box Plot for Numerical variables](#e.-Violin-Plot-and-Box-Plot-for-Numerical-variables:) |


</details>

<details>
<summary><strong>2. Bivariate Analysis</strong></summary>

| No. | Topic                                                                   |
|-----|-------------------------------------------------------------------------|
| a.  | [Scatter plot: AGE V/s Weight with Obesity Level](#a.-Scatter-plot:-AGE-V/s-Weight-with-Obesity-Level:) |
| b.  | [Scatter plot: AGE V/s Height with Obesity Level](#b.-Scatter-plot:-AGE-V/s-Height-with-Obesity-Level:) |
| c.  | [Scatter plot: Height V/s Weight with Obesity Level](#c.-Scatter-plot:-Height-V/s-Weight-with-Obesity-Level:) |
| d.  | [Scatter plot: AGE V/s Weight with Overweighted Family History](#d.-Scatter-plot:-AGE-V/s-Weight-with-Overweighted-Family-History:) |
| e.  | [Scatter plot: AGE V/s height with Overweighted Family History](#e.-Scatter-plot:-AGE-V/s-height-with-Overweighted-Family-History:) |
| f.  | [Scatter plot: Height V/s Weight with Overweighted Family History](#f.-Scatter-plot:-Height-V/s-Weight-with-Overweighted-Family-History:) |
| g.  | [Scatter plot: AGE V/s Weight with Transport use](#g.-Scatter-plot:-AGE-V/s-Weight-with-Transport-use:) |
| h.  | [Scatter plot: AGE V/s Height with Transport use](#h.-Scatter-plot:-AGE-V/s-Height-with-Transport-use:) |
| i.  | [Scatter plot: Height V/s Weight with Transport use](#i.-Scatter-plot:-Height-V/s-Weight-with-Transport-use:) |

</details>

<details>
<summary><strong>3. Multivariate Analysis</strong></summary>

| No. | Topic                                                                   |
|-----|-------------------------------------------------------------------------|
| a.  | [Pair Plot of Variables against Obesity Levels](#a.-Pair-Plot-of-Variables-against-Obesity-Levels:) |
| b.  | [Correlation heatmap for Pearson's correlation coefficient](#b.-Correlation-heatmap-for-Pearson's-correlation-coefficient:) |
| c.  | [Correlation heatmap for Kendall's tau correlation coefficient](#c.-Correlation-heatmap-for-Kendall's-tau-correlation-coefficient:) |
| d.  | [3D Scatter Plot of Numerical Columns against Obesity Level](#d.-3D-Scatter-Plot-of-Numerical-Columns-against-Obesity-Level:) |


<details>
<summary><strong>e. Cluster Analysis</strong></summary>

| No. | Topic                                                                        |
|-----|------------------------------------------------------------------------------|
| I.  | [K-Means Clustering on Obesity level](#I.-K-Means-Clustering-on-Obesity-level:) |
| II. | [PCA Plot of numerical variables against obesity level](#II.-PCA-Plot-of-numerical-variables-against-obesity-level:) |


</details>

</details>

<details>
<summary><strong>4. Outlier Analysis</strong></summary>

<details>
<summary><strong>a. Univariate Outlier Analysis</strong></summary>

| No. | Topic                                                         |
|-----|---------------------------------------------------------------|
| I.  | [Boxplot Outlier Analysis](#I.-Boxplot-Outlier-Analysis:)     |
| II. | [Detecting outliers using Z-Score](#II.-Detecting-outliers-using-Z-Score:) |
| III.| [Detecting outliers using Interquartile Range (IQR)](#III.-Detecting-outliers-using-Interquartile-Range-(IQR):) |

</details>

<details>
<summary><strong>b. Multivariate Outlier Analysis</strong></summary>

| No. | Topic                                                         |
|-----|---------------------------------------------------------------|
| I.   | [Detecting Multivariate Outliers Using Mahalanobis Distance](#I.-Detecting-Multivariate-Outliers-Using-Mahalanobis-Distance:) |
| II.  | [Detecting Multivariate Outliers Using Principal Component Analysis (PCA)](#II.-Detecting-Multivariate-Outliers-Using-Principal-Component-Analysis-(PCA):) |
| III. | [Detecting Cluster-Based Outliers Using KMeans Clustering](#III.-Detecting-Cluster-Based-Outliers-Using-KMeans-Clustering:) |


</details>

</details>

<details>
<summary><strong>5. Feature Engineering:</strong></summary>

| No. | Topic                                                              |
|-----|--------------------------------------------------------------------|
| a.  | [Encoding Categorical to numerical variables](#a.-Encoding-Categorical-to-numerical-variables:) |
| b.  | [BMI(Body Mass Index) Calculation](#b.-BMI(Body-Mass-Index)-Calculation:) |
| c.  | [Total Meal Consumed:](#c.-Total-Meal-Consumed:)                   |
| d.  | [Total Activity Frequency Calculation](#d.-Total-Activity-Frequency-Calculation:) |
| e.  | [Ageing process analysis](#e.-Ageing-process-analysis:)            |


</details>

</details>

<details>
<summary><strong>Section: 6. Analysis & Prediction Using Machine Learning(ML) Model</strong></summary>

| No. | Topic                                                                   |
|-----|-------------------------------------------------------------------------|
| 1.  | [Feature Importance Analysis and Visualization](#1.-Feature-Importance-Analysis-and-Visualization:) |
|     |   a. [Feature Importance Analysis  using Random Forest Classifier](#a.-Feature-Importance-Analysis--using-Random-Forest-Classifier:) |
|     |   b. [Feature Importance Analysis using XGBoost(XGB) Model](#b.-Feature-Importance-Analysis-using-XGBoost(XGB)-Model:) |
|     |   c. [Feature Importance Analysis Using (LightGBM) Classifier Model](#c.-Feature-Importance-Analysis-Using-(LightGBM)-Classifier-Model:) |
| 2.  | [Data visualization after Feature Engineering](#2.-Data-visualization-after-Feature-Engineering:) |
|     |   a. [Bar plot of numerical variables](#a.-Bar-plot-of-numerical-variables:) |
|     |   b. [PairPlot of Numerical Variables](#b.-PairPlot-of-Numerical-Variables:) |
|     |   c. [Correlation Heatmap of Numerical Variables](#c.-Correlation-Heatmap-of-Numerical-Variables:) |


</details>

<details>
<summary><strong>Section: 7. Prediction of Obesity Risk Level Using Machine learning(ML) Models</strong></summary>

| No. | Topic                                                                                                    |
|-----|----------------------------------------------------------------------------------------------------------|
| 1.  | [Machine Learning Model Creation: XGBoost and LightGBM - Powering The Predictions! 🚀](#1.-Machine-Learning-Model-Creation:-XGBoost-and-LightGBM---Powering-The-Predictions!-🚀) |
| 2.  | [Cutting-edge Machine Learning Model Evaluation: XGBoosting and LightGBM 🤖](#2.-Cutting-edge-Machine-Learning-Model-Evaluation:-XGBoosting-and-LightGBM-🤖) |
| 3.  | [Test Data Preprocessing for Prediction](#3.-Test-Data-Preprocessing-for-Prediction:) |
| 4.  | [Showcase Predicted Encdd_Obesity_Level Values on Test Dataset 📊](#4.-Showcase-Predicted-Encdd_Obesity_Level-Values-on-Test-Dataset-📊) |


</details>

<details>
<summary><strong>Section: 8. Conclusion: 📝</strong></summary>

| No. | Topic                                                                                  |
|-----|----------------------------------------------------------------------------------------|
| 1.  | [Conclusion: 📝](#Conclusion:-📝)                                                      |
| 2.  | [It's time to make Submission:](#It's-time-to-make-Submission:)                        |


</details>


# About Obesity Risk Level Prediction-Project:
<details>
<summary><b><span style="color:blue">Understanding Obesity and Risk Prediction:</span></b></summary>

<ul>
  <li><b>Understanding Obesity:</b>
    <ul>
      <li>Obesity stems from excessive body fat accumulation, influenced by genetic, environmental, and behavioral factors.</li>
      <li>Risk prediction involves analyzing demographics, lifestyle habits, and physical activity to classify individuals into obesity risk categories.</li>
    </ul>
  </li>
  <li><b>Global Impact:</b>
    <ul>
      <li>Worldwide obesity rates have tripled since 1975, affecting 30% of the global population.</li>
      <li>Urgent action is needed to develop effective risk prediction and management strategies.</li>
    </ul>
  </li>
  <li><b>Factors Influencing Risk:</b>
    <ul>
      <li>Obesity risk is shaped by demographics, lifestyle habits, diet, physical activity, and medical history.</li>
      <li>Analyzing these factors reveals insights into obesity's mechanisms and identifies high-risk populations.</li>
    </ul>
  </li>
  <li><b>Data-Driven Approach:</b>
    <ul>
      <li>Advanced machine learning and large datasets enable the development of predictive models for stratifying obesity risk.</li>
      <li>These models empower healthcare professionals and policymakers to implement tailored interventions for improved public health outcomes.</li>
    </ul>
  </li>
  <li><b>Proactive Health Initiatives:</b>
    <ul>
      <li>Our proactive approach aims to combat obesity by leveraging data and technology for personalized prevention and management.</li>
      <li>By predicting obesity risk, we aspire to create a future where interventions are precise, impactful, and tailored to individual needs.</li>
    </ul>
  </li>
</ul>

<p><b>Source:</b> <a href="https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight">World Health Organization.</a> (2022). Obesity and overweight.</p>
</details>

<details>
<summary><b><span style="color:blue">Dataset Overview:</span></b></summary>

<p>The dataset contains comprehensive information encompassing eating habits, physical activity, and demographic variables, comprising a total of 17.</p>

<h3>Key Attributes Related to Eating Habits:</h3>
<ul>
  <li><b>Frequent Consumption of High-Caloric Food (FAVC):</b> Indicates the frequency of consuming high-caloric food items.</li>
  <li><b>Frequency of Consumption of Vegetables (FCVC):</b> Measures the frequency of consuming vegetables.</li>
  <li><b>Number of Main Meals (NCP):</b> Represents the count of main meals consumed per day.</li>
  <li><b>Consumption of Food Between Meals (CAEC):</b> Describes the pattern of food consumption between main meals.</li>
  <li><b>Consumption of Water Daily (CH20):</b> Quantifies the daily water intake.</li>
  <li><b>Consumption of Alcohol (CALC):</b> Indicates the frequency of alcohol consumption.</li>
</ul>

<h3>Attributes Related to Physical Condition:</h3>
<ul>
  <li><b>Calories Consumption Monitoring (SCC):</b> Reflects the extent to which individuals monitor their calorie intake.</li>
  <li><b>Physical Activity Frequency (FAF):</b> Measures the frequency of engaging in physical activities.</li>
  <li><b>Time Using Technology Devices (TUE):</b> Indicates the duration spent using technology devices.</li>
  <li><b>Transportation Used (MTRANS):</b> Describes the mode of transportation typically used.</li>
</ul>

<p>Additionally, the dataset includes essential demographic variables such as gender, age, height, and weight, providing a comprehensive overview of individuals' characteristics.</p>

<h3>Target Variable:</h3>
<p>The target variable, NObesity, represents different obesity risk levels, categorized as:</p>
<ul>
  <li>Underweight (BMI < 18.5): 0</li>
  <li>Normal (18.5 <= BMI < 20): 1</li>
  <li>Overweight I (20 <= BMI < 25): 2</li>
  <li>Overweight II (25 <= BMI < 30): 3</li>
  <li>Obesity I (30 <= BMI < 35): 4</li>
  <li>Obesity II (35 <= BMI < 40): 5</li>
  <li>Obesity III (BMI >= 40): 6</li>
</ul>
</details>



