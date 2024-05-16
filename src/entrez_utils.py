#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:51:22 2024

@author: nicholasrenegar
"""
from Bio import Entrez
from datetime import datetime

def search(query, retstart, paper_start_date):
    Entrez.email = 'email@example.com'
    
    # Update the query to include a date range
    current_date = datetime.now().strftime("%Y/%m/%d")
    query_with_date = f"{query} AND ({paper_start_date}[Date - Publication] : {current_date}[Date - Publication])"
        

    handle = Entrez.esearch(db='pubmed',
                            sort='date',
                            retstart=retstart,
                            retmax='10000',
                            retmode='xml',
                            term=query_with_date,
                            usehistory='y')  # Enable history to handle searches larger than 10k
    results = Entrez.read(handle)
    handle.close()
    return results

def fetch_details(id_list, webenv, query_key, retstart=0, retmax=10000):
    Entrez.email = 'email@example.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           rettype='medline',
                           webenv=webenv,
                           query_key=query_key,
                           retstart=retstart,
                           retmax=retmax)
    results = Entrez.read(handle)
    handle.close()
    return results