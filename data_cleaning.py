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
print("Missing elements")
for col in df_copy.columns:
    series_col = df_copy[df_copy[col] == '-1']
    print('{}\t: \t{}'.format(col, len(series_col)))
print("\n")
    
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

df_droped = df_copy.copy()


# Clean Cumpany name column
## Parse ratings out of comp name
df_droped["company_name"] = df_droped["Company Name"].apply(lambda x: x.split("\n")[0])

# Parse City out of Location Column 
df_droped["location_city"] = df_droped["Location"].apply(lambda x: x.split(", ")[0])

# Parse City out of Headquaters Column 
df_droped["hq_city"] = df_droped["Headquarters"].apply(lambda x: x.split(", ")[0])

# Turn ratings to  a number
df_droped["ratings"] = df_droped["Rating"].apply(lambda x: round(float(x),2))


# Clean Salary estimates
## Remove text, 'K', '$' and '-' and convert values to int
df_droped["salary_estimate"] = df_droped["Salary Estimate"].apply(lambda x: list(map(int, x[1:-18].split("K-£"))))
## Multiply entries by 1000 because we took out  the 'K'
df_droped["salary_estimate"] = df_droped["salary_estimate"].apply(lambda x: [1000 * x[0], 1000 * x[1]])

## Create Aerage, Min and Max Salary columns
df_droped["avg_salary"] = df_droped["salary_estimate"].apply(lambda x: sum(x)/2)
df_droped["min_salary"] = df_droped["salary_estimate"].apply(lambda x: x[0])
df_droped["max_salary"] = df_droped["salary_estimate"].apply(lambda x: x[1])
df_droped = df_droped.drop("salary_estimate", axis=1)


df_droped["revenue"] = df_droped["Revenue"].apply(lambda x: "-1" if x[0] == "U" else x)
df_droped["revenue"] = df_droped["revenue"].apply(lambda x: x.replace(" (GBP)", "").replace("£", "").replace(" to", "").split(" "))

def convert_to_num(x):
    
    if len(x) == 4:
        # Less than n million
        if x[0][0] == 'L':
            y =  [0, int(x[2])*10**6]
        # n million - m billion
        else:
            y = [int(x[0])*10**6, int(x[2])*10**9]
        
    elif len(x) == 3:
        units = x[2]
        # n - m million
        if units == 'million':
            y = [int(x[0])*10**6, int(x[1])*10**6]
        # n - m billion
        elif units == 'billion':
            y = [int(x[0])*10**9, int(x[1])*10**9]
    
    elif len(x) == 2:
        # n+ billion ==> n billion - 1 trillion
        y = [int(x[0][:-1])*10**9, 10**12]
    
    else:
        # -1
        y = [int(x[0])]
        
    return y


## 'Unknown / Non-Applicable' --> '-1'
## Remove ' (GDP)'
## Remove '£' symbol
## Seperate into upper and lowwer est
## Convert strings to numbers
df_droped["revenue"] = df_droped["revenue"].apply(lambda x: convert_to_num(x))
df_droped["min_revenue"] = df_droped["revenue"].apply(lambda x: x[0])
df_droped["max_revenue"] = df_droped["revenue"].apply(lambda x: x[-1])
df_droped["avg_revenue"] = (df_droped["min_revenue"]+df_droped["max_revenue"])/2
df_droped = df_droped.drop("revenue", axis=1)


# Seperate company size column into uper and lower est
## Parse out 'employees' and '+'
df_droped["size"] = df_droped["Size"].apply(lambda x: x.replace("Unknown", "-1"))
df_droped["size"] = df_droped["size"].apply(lambda x: x.split(" employees")[0])
df_droped["size"] = df_droped["size"].apply(lambda x: x.split("+")[0])
df_droped["size"] = df_droped["size"].apply(lambda x: list(map(int, x.split(" to "))))

## Max and min employees
df_droped["min_size"] = df_droped["size"].apply(lambda x: x[0])
df_droped["max_size"] = df_droped["size"].apply(lambda x: x[-1])
df_droped["avg_size"] = (df_droped["min_size"]+df_droped["max_size"])/2
df_droped = df_droped.drop("size", axis=1)



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
Languages = ["python", "excel", "sql", "spark", "hadoop", "java ", " r ", "tensorflow", "matlab"]
df_droped["languages"] = df_droped["Job Description"].apply(lambda x: text_presence(x, Languages))

## Print Language Frequencies
print("Language/Tools Frequency")
for language in Languages:
    lan_frq = (df_droped["languages"].apply(lambda x: x[language]).sum()/1300)*100
    print("{}\t: \t{:.2f}% ".format(language, lan_frq))
print("\n")


# Clean company type
## Remove 'Company - ' from enttries
owner_types = {'Private'                        : 'private', 
               'Public'                         : 'public',
               'College / University'           : 'third', 
               'Unknown'                        : '-1',
               'Non-profit Organisation'        : 'third',
               '-1'                             : '-1',
               'Subsidiary or Business Segment' : 'private',
               'Government'                     : 'public',
               'Contract'                       : 'private',
               'Private Practice / Firm'        : 'private',
               'Hospital'                       : 'public'}
df_droped["ownership"] = df_droped["Type of ownership"].apply(lambda x: x.replace("Company - ", ""))
df_droped["ownership"] = df_droped["ownership"].apply(lambda x: owner_types[x])


# Turn industry into list element
df_droped["industry"] = df_droped["Industry"].apply(lambda x: x.replace(", ", " & "))
df_droped["industry"] = df_droped["industry"].apply(lambda x: x.replace("& &", "&"))
df_droped["industry"] = df_droped["industry"].apply(lambda x: x.split(" & "))


df_droped["sector"] = df_droped["Sector"].apply(lambda x: x.replace(", ", " & "))
df_droped["sector"] = df_droped["sector"].apply(lambda x: x.replace("& &", "&"))
df_droped["sector"] = df_droped["sector"].apply(lambda x: x.split(" & "))


df_droped["competitors"] = df_droped["Competitors"].apply(lambda x: x.split(', '))
df_droped["founded"] = df_droped["Founded"].apply(lambda x: int(x))


data = df_droped.drop(["Job Title",
                       "Salary Estimate",
                       "Rating",
                       "Company Name",
                       "Size",
                       "Founded",
                       "Type of ownership",
                       "Industry",
                       "Sector",
                       "Revenue",
                       "Competitors"], axis=1)
print(data.dtypes)

data.to_csv('glassdoor_jobs_cleaned.csv', index=False)

# Create seniority field from title and description