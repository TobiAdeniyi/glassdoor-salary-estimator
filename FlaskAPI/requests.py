#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Wed Jul 22 13:53:45 2020@author: tobiadeniyi"""import requestsfrom data_input import data_indata = {"input": data_in}URL = "http://127.0.0.1:5000/predict"headers = {"Content-Type": "application/json"}r = requests.get(URL,headers=headers, json=data) r.json()