# Salary Estimator for Uk Data Science roles

###
###



## Project Overview

#### Objective:
To develop a production pipeline able to predict job salaries of subfiels of a give carier domain.
In this instance - a user can input a data science related job title such as data scientist, machine learning engineer, etc... 
As input, the use may provide:

* Role Type - Intern, Junior, Senior, Manager, ...
* Job title
* Job location
* Company name

#### Results and Insight:
* Created tool that estimates data science salaries (MAE ~ $ 11K) to help data scientists negotiate their income.
* Scraped over 1000 job descriptions from glassdoor using python and selenium.
* Engineered features from text in job description to quantify the value of python, excel, aws, and spark to companies.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
* Built a client facing API using flask.

###
###



## TOC

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

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2" the following lines were added.

```python
  try:
      driver.find_element_by_css_selector("[alt=Close]").click()  #clicking to the X.
  except NoSuchElementException:
      pass

  if len(jobs) < 1:
      try:
          driver.find_element_by_id("onetrust-accept-btn-handler").click()  #clicking on the Accept cookies btn.
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

This is saved as a csv file name [`glassdoor_jobs.csv`](). 

###
###



## Data Cleaning

After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

Parsed numeric data out of salary
Made columns for employer provided salary and hourly wages
Removed rows without salary
Parsed rating out of company text
Made a new column for company state
Added a column for if the job was at the company’s headquarters
Transformed founded date into age of company
Made columns for if different skills were listed in the job description:
Python
R
Excel
AWS
Spark
Column for simplified job title and Seniority
Column for description length

### How to implement

Explain what these tests test and why

```
Give an example
```

###
###



## Exploratory data analysis

Add additional notes about how to deploy this on a live system

### Analysis

Explain what these tests test and why

```
Give an example
```

### Feature Engineering

Explain what these tests test and why

```
Give an example
```

###
###



## Model building

###
###



## Production

Users enter search

If company info in database --> Auto-fill secondary info
Else --> retur imputed data

Predict based on input

Show user Estimated salary, variable that provides largest increase in pay

###
###



## Authors

* **[Tobiloba Adeniyi](https://github.com/TobiAdeniyi)** - *Initial work* - [Salary Estimator](https://github.com/TobiAdeniyi/glassdoor_proj)



## Acknowledgments

* [Ken Jee](https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg) - Data Science Projects From Scratch.
* [Kaggle](https://www.kaggle.com/) - Courses.
* [Glassdoor](https://www.glassdoor.co.uk/index.htm) - Job data.
