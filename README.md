# Salary Estimator for Uk Data Science roles

###
###



## Project Overview

#### Objective:
To develop a production pipeline able to predict job salaries of subfiels of a give carier domain.
In this instance - a user can input a data science related job title such as data scientist, machine learning engineer, etc... 
As input, the use may provide:

* Role Type - Junior or Senior
* Job title
* Job location
* Company name

#### Results and Insight:
* Created tool that estimates data science salaries (MAE ~ £ 6K) to help data scientists negotiate their income.
* Scraped over 1200 job descriptions from glassdoor using python and selenium.
* Engineered features from job description to quantify the value of python, excel, aws, ect... to companies.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
* Built a client facing API using flask.

###
###



## Prerequisites

* [IDE](https://jupyter.org/install) - Jupyter Lab

* [Python Version:](https://www.python.org/downloads/) - 3.7

* [Web Driver](https://chromedriver.chromium.org/) - Chrome Web Driver

* [Glassdoor Web Scraper:](https://github.com/arapfaik/scraping-glassdoor-selenium) - Github

* [Glassdoor Web Scraper:](https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905) - Article

* [Flask Productionisation:](https://www.python.org/downloads/) - Github

* [Flask Productionisation:](https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2) - Article

* [Requierments Folder:]() - requiments.txt

* Packages: - pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle

For installation, run the following:
```
>>> pip install -r requirements.txt
```

###
###



## Data Collection

The data used in our project is collected from the [Glassdoor](https://www.glassdoor.co.uk/index.htm) website, using a [web scrapper](https://github.com/arapfaik/scraping-glassdoor-selenium), developed with the **Selenium** browser autimated libray. The basic instillation and deployment proceses are discussed in the following [article](https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905).

For the [glassdoor_scraper.py]() we tweek the following parameters:
* URL - Changed to match UK's Glassdoor website address.
* Exception statements - Removed `ElementClickInterseption` and `NoSuchElementExeption`.
* find_element_by_xpth() - Changed input to mach current website div class or tags.

Additionally to **bypass** the "*login*" and "*accept cookies*" prompts,

![alt text][logo]

[logo]: https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/glassdoor_prompts.png "Glassdoor Prompts" 

the following lines were added.

```python
  try:
      #clicking to the X
      driver.find_element_by_css_selector("[alt=Close]").click()
  except NoSuchElementException:
      pass

  if len(jobs) < 1:
      try:
          #clicking on the Accept cookies btn
          driver.find_element_by_id("onetrust-accept-btn-handler").click()
      except:
          pass
```

By specifying the path to the web driver, the key word we are searching for, and the number of search result in the data_coolection.py file we get a dataframe containing the following information:
* Job title
* Salary Estimate
* Job Description
* Rating
* Company
* Location
* Company Headquarters
* Company Size
* Company Founded Date
* Type of Ownership
* Industry
* Sector
* Revenue
* Competitors

```python
import glassdoor_scraper as gs
import pandas as pd

path = "/Users/tobiadeniyi/Documents/Portfolio/Python/ProjectLibrary/glassdoor_proj/chromedriver"
df = gs.get_jobs('data-scientist', 1300, False, path, 5)
df.to_csv('glassdoor_jobs.csv', index = False)
```

This is saved as a csv file name [`glassdoor_jobs.csv`](https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/glassdoor_jobs.csv). 

###
###



## Data Cleaning

During data collection entries without salary data were scraped. After collection, the data is cleaned to allow for the best predictions and accuracy of our model. Additionally new feachurs are parsed from existing fields (columns) and some existing variables are changed as follows:

1. Parsed job seniority from job title
2. Parsed numeric data from:
  * salary
  * rating
  * revenue
  * founded
  * company size
3. Created field containng lists for:
  * companies sector
  * companies industry
  * companies competitors
  * companies ownership type
4. Made new columns for:
  * job location - city
  * company headquaters - city
5. Created new veriable:
  * 1 if "company city" == "HQ city"
  * otherwise 0
6. Parsed important [DS tools](https://data-flair.training/blogs/data-science-tools/) from job discription:
  * Python
  * Excel
  * SQL
  * AWS
  * Spark
  * Hadoop
  * Java
  * Tensorflow
  * MATLAB
  * R

###
###

Here is an example of how this was done with DS tools. By inserting spaces before and after "R", we can recognise if the programming language was present in the job discription.

```python
tools = ["python", "excel", "sql", "aws", "spark", "hadoop", "java ", " r ", "tensorflow", "matlab"]
df["tools"] = df["Job Description"].apply(lambda x: text_presence_in_description(x, tools))
```

###
###



## Exploratory data analysis and Feature Engineering

To analyses the data the appropreate graphs where plotted, along with various tables of the data and the value counts for categorical variables. Below are a few highlights from the pivot tables.

![alt text](https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/EDA_0.png "Box Plot")
![alt text](https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/EDA_1.png "Correlation Plot")
![alt text](https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/EDA_3.png "Bar Plot") 
![alt text](https://github.com/TobiAdeniyi/glassdoor_proj/blob/master/eda.png "Word Cloud")

Additional new features were created, including the number of languages (tools) in a given job and the number of competitors a company has.

###
###



## Model building

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline model to test other approaches.
*	**Lasso Regression** – Due to data sparcity, a normalized regression would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance

The Gradient Boosted, and Lasso Regression models far outperformed the other approaches on the test and validation sets. 

| Model                      | MEA   | R^2   | EV    |
| :------------------------: |------:|------:|------:|
| Gradient Boosted Regressor | ?     | ?     | ?     |
| Lasso Regressor            | ?     | ?     | ?     |
| XGBoosted Regressor        | ?     | ?     | ?     |
| Randome Forest Regressor   | ?     | ?     | ?     |
| Linear Regressor           | ?     | ?     | ?     |
| Ridge Regressor            | ?     | ?     | ?     |

###
###



## Production

In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 

Users enter search

If company info in database --> Auto-fill secondary info
Else --> retur imputed data

Predict based on input

Show user Estimated salary, variable that provides largest increase in pay

###
###



## Looking Foward

###
###



## Authors

* **[Tobiloba Adeniyi](https://github.com/TobiAdeniyi)** - *Initial work* - [Salary Estimator](https://github.com/TobiAdeniyi/glassdoor_proj)



## Acknowledgments

* [Ken Jee](https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg) - Data Science Projects From Scratch.
* [Kaggle](https://www.kaggle.com/) - Courses.
* [Glassdoor](https://www.glassdoor.co.uk/index.htm) - Job data.
* [DS tools](https://data-flair.training/blogs/data-science-tools/) - Essential Data Science Ingredients
