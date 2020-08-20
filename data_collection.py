#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:33:40 2020

@author: tobiadeniyi
"""

import glassdoor_scraper as gs
import pandas as pd


path = "/Users/tobiadeniyi/Documents/Portfolio/Python/ProjectLibrary/glassdoor_proj/chromedriver"
df = gs.get_jobs('data-scientist', 1300, False, path, 5)
# df.to_csv('glassdoor_jobs.csv', index = False)