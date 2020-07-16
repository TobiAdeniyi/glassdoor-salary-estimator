#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:16:22 2020

@author: tobiadeniyi
"""

import pandas as pd


df = pd.read_csv('glassdoor_jobs.csv')
df_copy = df.copy()
df_copy = df_copy.astype(str)




# Remove entries without salary or location entries

## Number of entries each column is missing
print("The totale noumber of missing elements in  each column")
for col in df_copy.columns:
    series_col = df_copy[df_copy[col] == '-1']
    print('{}: {}'.format(col, len(series_col)))
    
    
"""
Headquaters: 168
Size: 163
Founded: 346
Types of ownership: 163
Industry: 277
Sector: 277
Revenue: 163
Competitors: 947

-->
"""
# Drop Competitors
df_droped = df_copy.drop(["Competitors"], axis=1)




# Clean Salary estimates

## Remove text, 'K', '$' and '-' and convert values to int
df_droped["salary_estimate"] = df_droped["Salary Estimate"].apply(lambda x: list(map(int, x[1:-18].split("K-£"))))

## Multiply entries by 1000 because we took out  the 'K'
df_droped["salary_estimate"] = df_droped["salary_estimate"].apply(lambda x: [1000 * x[0], 1000 * x[1]])

## Create Aerage, Min and Max Salary columns
df_droped["avg_salary"] = df_droped["salary_estimate"].apply(lambda x: sum(x)/2)
df_droped["min_salary"] = df_droped["salary_estimate"].apply(lambda x: x[0])
df_droped["max_salary"] = df_droped["salary_estimate"].apply(lambda x: x[1])

df_droped = df_droped.drop("Salary Estimate", axis=1)



# Parse Relevant data from Job Description column

## says if text was in job description
def text_presence(job_desc, lan_list):
    """

    Parameters
    ----------
    job_desc : str
        Job Description.
    lan_list : list
        List of languages.

    Returns
    -------
    dic      : dictionary
        Dictionary of if each language was in description.

    """
    dic = {}
    
    for language in lan_list:
        present = language in job_desc.lower()
        dic[language] = int(present)
        
    return dic



## E.g. Refrence to programming languages (Python, Java, etc)
### To enture Java and R are stand alone insert space before and or after
Languages = ["python", "java ", "javascript", "html", "css", "matlab", " r "]
df_droped["languages"] = df_droped["Job Description"].apply(lambda x: text_presence(x, Languages))

## Print Language Frequencies
print("\nLanguage Frequency")
for language in Languages:
    lan_frq = df_droped["languages"].apply(lambda x: x[language]).sum()/1300
    print("{}\t: \t{:.2f}% ".format(language, lan_frq))




# Clean Cumpany name column
## Parse ratings out of comp name
df_droped["company_name"] = df_droped["Company Name"].apply(lambda x: x.split("\n")[0])

df_droped = df_droped.drop("Company Name", axis=1)



# Parse City out of Location Column 
df_droped["location_city"] = df_droped["Location"].apply(lambda x: x.split(", ")[0])

# Parse City out of Headquaters Column 
df_droped["hq_city"] = df_droped["Headquarters"].apply(lambda x: x.split(", ")[0])



# Seperate company size column into uper and lower est

## Parse out 'employees' and '+'
df_droped["Size"] = df_droped["Size"].apply(lambda x: x.replace("Unknown", "-1"))
df_droped["Size"] = df_droped["Size"].apply(lambda x: x.split(" employees")[0])
df_droped["Size"] = df_droped["Size"].apply(lambda x: x.split("+")[0])
df_droped["Size"] = df_droped["Size"].apply(lambda x: list(map(int, x.split(" to "))))

## Max and min employees
df_droped["min_size"] = df_droped["Size"].apply(lambda x: x[0])
df_droped["max_size"] = df_droped["Size"].apply(lambda x: x[-1])




# Clean company type

## Remove 'Company - ' from enttries
df_droped["Type of ownership"] = df_droped["Type of ownership"].apply(lambda x: x.replace("Company - ", ""))


#data = df_droped.drop(["Industry", "Sector", "Revenue"], axis=1)

# Turn industry into list element




# Clean revenu column

## 'Unknown / Non-Applicable' --> 'Nan'
## Remove ' (GDP)'
## Remove '£' symbol
## Seperate into upper and lowwer est
## Convert strings to numbers



# Create seniority field from title and description