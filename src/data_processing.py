#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:51:22 2024

@author: nicholasrenegar
"""
import pandas as pd
import numpy as np
import re
from src import entrez_utils
#import time
#import os


def fetch_and_save(term, paper_start_date):
    #while True:
    try:
        df_out = summarize_studies(term, paper_start_date)
        return df_out
    except Exception as e:
        print(e)


def combine_dataframes(dfs):
    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return combined_df


def get_abstract_text(article):
    abstract = article.get('Abstract', {})
    abstract_texts = abstract.get('AbstractText', [])
    if abstract_texts:
        # Join all parts of the abstract into one string
        return ' '.join(abstract_texts)
    else:
        return 'No Abstract'


def get_authors(article):
    if 'AuthorList' in article:
        authors = article['AuthorList']
        author_names = [
            f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip(
            ) for author in authors
        ]
        return ', '.join(author_names)
    else:
        return 'NA'


def get_completed_date(medLineInfo):
    if 'DateCompleted' in medLineInfo:
        year = medLineInfo['DateCompleted'].get('Year', 'NA')
        month = medLineInfo['DateCompleted'].get('Month', 'NA')
        day = medLineInfo['DateCompleted'].get('Day', 'NA')
        return f"{year}-{month}-{day}" if year != 'NA' and month != 'NA' and day != 'NA' else 'NA'
    else:
        return 'NA'


def get_edit_date(medLineInfo):
    if 'DateRevised' in medLineInfo:
        year = medLineInfo['DateRevised'].get('Year', 'NA')
        month = medLineInfo['DateRevised'].get('Month', 'NA')
        day = medLineInfo['DateRevised'].get('Day', 'NA')
        return f"{year}-{month}-{day}" if year != 'NA' and month != 'NA' and day != 'NA' else 'NA'
    else:
        return 'NA'


def get_first_author_affiliation(article):
    try:
        authors = article['AuthorList']
        if 'AffiliationInfo' in authors[0] and authors[0]['AffiliationInfo']:
            return authors[0]['AffiliationInfo'][0].get('Affiliation', 'NA')
        else:
            return 'NA'
    except:
        return 'NA'


def summarize_studies(search_term, paper_start_date):
    studies = entrez_utils.search(search_term, 0, paper_start_date)
    total_records = int(studies['Count'])
    webenv = studies['WebEnv']
    query_key = studies['QueryKey']

    # Initialization
    title_list = []
    abstract_list = []
    journal_list = []
    language_list = []
    pubdate_year_list = []
    pubdate_month_list = []
    completion_date_list = []
    edit_date_list = []
    author_list = []
    first_author_affil_list = []
    url_list = []

    #Get paper details
    chunk_size = 10000  # NCBI's max allowed fetch size for PubMed
    #for start in range(0, total_records, chunk_size):
    for start in range(0, 9998, chunk_size):
        studies = entrez_utils.search(search_term, start, paper_start_date)
        total_records = int(studies['Count'])
        webenv = studies['WebEnv']
        query_key = studies['QueryKey']

        print(
            f"Fetching records {start+1} to {start+chunk_size} of {total_records}"
        )
        papers = entrez_utils.fetch_details(None, webenv, query_key, start,
                                            chunk_size)

        # Processing each paper
        for paper in papers['PubmedArticle']:
            #print('paper')
            article = paper['MedlineCitation']['Article']
            title_list.append(article.get('ArticleTitle', 'No Title'))

            # Abstract
            #print('abstract')
            abstract_text = get_abstract_text(article)
            abstract_list.append(abstract_text)

            # Journal info
            #print('journal')
            journal = article.get('Journal', {})
            journal_list.append(journal.get('Title', 'No Journal'))
            language_list.append(article.get('Language', ['No Language'])[0])

            # Publication date
            #print('publication')
            pub_date = journal.get('JournalIssue', {}).get('PubDate', {})
            pubdate_year_list.append(pub_date.get('Year', 'No Data'))
            pubdate_month_list.append(pub_date.get('Month', 'No Data'))

            # Completion and editing dates
            #print('article_date')
            article_date_info = paper['MedlineCitation'].get('ArticleDate', [])
            completed_date = get_completed_date(paper['MedlineCitation'])
            completion_date_list.append(completed_date)
            edited_date = get_edit_date(paper['MedlineCitation'])
            edit_date_list.append(edited_date)

            # Authors
            #print('authors')
            author_list.append(get_authors(article))
            first_author_affil_list.append(
                get_first_author_affiliation(article))

            # Extract PMID and construct URL
            pmid = str(paper['MedlineCitation']['PMID'])
            article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            url_list.append(article_url)

    # Create DataFrame
    df = pd.DataFrame({
        'Title': title_list,
        'Abstract': abstract_list,
        'Journal': journal_list,
        'Language': language_list,
        'Year': pubdate_year_list,
        'Month': pubdate_month_list,
        'Completion Date': completion_date_list,
        'Edit Date': edit_date_list,
        'Authors': author_list,
        'First Author Affiliation': first_author_affil_list,
        'url': url_list
    })

    return df


def clean_journal_names(journal_name):
    # Move ", the" to the front
    journal_name = re.sub(r',\s*the$',
                          'The ',
                          journal_name,
                          flags=re.IGNORECASE)

    # Convert to title case
    return journal_name.title()
