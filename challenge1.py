#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
challenge1.py
Last Mod Nov 30 2020

@author: Conrad Borchers
"""

# Setup

import os

# os.chdir("C:/Users/conra/desktop/code_challenge")

from platform import python_version, platform
print(python_version()) # 3.7.0
print(platform())       # Win 10

import pandas as pd
from cdqa.utils.download import download_model
from cdqa.utils.converters import pdf_converter
from cdqa.pipeline import QAPipeline

# Read in all pdf documents in /data and set up bert

os.system("echo 'Collecting all pdf files in data/ and copying them to pdf/'")
os.system("mkdir pdf > nul 2> nul")
os.system("cp data/**/*.pdf pdf")

os.system("echo 'Setting up bert'")
os.system("mkdir bert > nul 2> nul")
download_model(model = "bert-squad_1.1", dir="./bert")

# Read in pdfs, set up and fit QA pipeline

df = pdf_converter(directory_path = "./pdf")
df.head()

pipeline = QAPipeline(reader = "./bert/bert_qa.joblib")

pipeline.fit_retriever(df=df)

# Sample queriers

def print_answer(q, result):
    print(
            "For Question: {}\nGot answer: {}\nIn file: {}.pdf\nAt paragraph:\n{}".format(
         q, result[0][0], result[0][1], result[0][2]
        )  
    )


q = "Does Sokalan have to be stored in a cool environment?"

result = pipeline.predict(q, 1)  # top response, first index of result

print_answer(q, result)

q = "What is texapon?"

result = pipeline.predict(q, 1) 

print_answer(q, result)

