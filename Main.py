#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:51:22 2024

@author: nicholasrenegar
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src import data_processing
#from src import Utils
#import re

# Inputs
date_string = "May 11th, 2024"
keyterms_df = pd.read_csv('keyterms.csv', header=None)
search_terms = ['"' + term + '"' for term in keyterms_df[0].tolist()]
paper_start_date = (datetime.now() - timedelta(days=2)).strftime("%Y/%m/%d")
#paper_start_date = (datetime.now() - timedelta(days=2)).strftime("%Y/%m/%d")

##############################################################################################
##      Get all relevant pubmed papers for the search terms from the last seven days
##############################################################################################

# Set current date and timedelta for paper search
current_date = datetime.now().strftime("%Y/%m/%d")

# Initialize a list to hold DataFrames
dfs = []

# Iterate through each search term
for term in search_terms:
    print(term)
    dfs.append(data_processing.fetch_and_save(term, paper_start_date))

# Combine all DataFrames into a single DataFrame
#directory = "/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/Recent Newsletter/"
combined_df = data_processing.combine_dataframes(dfs)

#Save to github
#combined_df.to_csv('/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/Recent Newsletter/recent_studies.csv', index=False)

# Filter studies based on date criteria
filtered_df = combined_df[combined_df['Completion Date'].apply(
    lambda x: pd.to_datetime(x, errors='coerce') >= datetime.now() - timedelta(
        days=14))]
#filtered_df.to_csv('filtered_studies.csv', index=False)

##############################################################################################
##      Add the journal impact factor
##############################################################################################

#Load new studies
#combined_df=pd.read_csv('all_studies.csv')
#filtered_df=pd.read_csv('filtered_studies.csv')
#filtered_df = filtered_df.sample(n=200)

####### IMPACT FACTOR

#Load impact factors
impact_factors_df = pd.read_csv('./data/scimagojr 2022.csv', delimiter=';')
impact_factors_df.rename(columns={
    'Title': 'Journal',
    'H index': 'ImpactFactor'
},
                         inplace=True)
impact_factors_df = impact_factors_df[['Journal', 'ImpactFactor']]

#Clean journal names before merging and drop duplicate rows from impact factors
filtered_df['Journal'] = filtered_df['Journal'].apply(
    data_processing.clean_journal_names)
impact_factors_df['Journal'] = impact_factors_df['Journal'].apply(
    data_processing.clean_journal_names)
impact_factors_df = impact_factors_df.sort_values(by='ImpactFactor',
                                                  ascending=False)
impact_factors_df = impact_factors_df.drop_duplicates(subset='Journal')

# Merge impact factors to main dataframe, and add missing impact factors @ low value
merged_df = pd.merge(filtered_df, impact_factors_df, on='Journal', how='left')
merged_df['ImpactFactor'] = merged_df['ImpactFactor'].fillna(5.0)

merged_df.to_csv('merged_studies.csv')
