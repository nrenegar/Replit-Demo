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
from src import Utils
from openai import OpenAI
import re


# Inputs
date_string = "May 11th, 2024"
keyterms_df = pd.read_csv('keyterms.csv', header=None)
search_terms = ['"' + term + '"' for term in keyterms_df[0].tolist()]
paper_start_date = (datetime.now() - timedelta(weeks=2)).strftime("%Y/%m/%d")
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
    data_processing.fetch_and_save(term, paper_start_date)

# Combine all DataFrames into a single DataFrame
directory = "/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/Recent Newsletter/"
combined_df = data_processing.combine_dataframes(directory)

#Save to github
combined_df.to_csv('/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/Recent Newsletter/recent_studies.csv', index=False)

# Filter studies based on date criteria
filtered_df = combined_df[
    combined_df['Completion Date'].apply(
        lambda x: pd.to_datetime(x, errors='coerce') >= datetime.now() - timedelta(days=14)
    )
]
filtered_df.to_csv('filtered_studies.csv', index=False)


##############################################################################################
##      Add the journal impact factor
##############################################################################################

#Load new studies
#combined_df=pd.read_csv('all_studies.csv')
filtered_df=pd.read_csv('filtered_studies.csv')
#filtered_df = filtered_df.sample(n=200)


####### IMPACT FACTOR

#Load impact factors
impact_factors_df = pd.read_csv('./data/scimagojr 2022.csv', delimiter=';')
impact_factors_df.rename(columns={'Title': 'Journal', 'H index': 'ImpactFactor'}, inplace=True)
impact_factors_df = impact_factors_df[['Journal', 'ImpactFactor']]

#Clean journal names before merging and drop duplicate rows from impact factors
filtered_df['Journal'] = filtered_df['Journal'].apply(data_processing.clean_journal_names)
impact_factors_df['Journal'] = impact_factors_df['Journal'].apply(data_processing.clean_journal_names)
impact_factors_df = impact_factors_df.sort_values(by='ImpactFactor', ascending=False)
impact_factors_df = impact_factors_df.drop_duplicates(subset='Journal')

# Merge impact factors to main dataframe, and add missing impact factors @ low value
merged_df = pd.merge(filtered_df, impact_factors_df, on='Journal', how='left')
merged_df['ImpactFactor'] = merged_df['ImpactFactor'].fillna(5.0)

##############################################################################################
##      Use automated methods to filter out irrelevant papers
##############################################################################################

####### Use LLMs to sequentially filter out irrelevant papers
client = OpenAI()

def filter_human_studies(title, abstract):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled researcher with expertise in identifying the focus of scientific papers."},
            {"role": "user", "content": f"""Title: {title}\nAbstract: {abstract}\nDetermine if this study is based on human subjects. Respond with 'Human Study: Yes' if it is, otherwise 'Human Study: No'."""}
        ]
    )

    response = completion.choices[0].message.content
    is_human_study = 'Yes' in response
    return 1 if is_human_study else 0

# Apply the function to each row in merged_df and filter out non-human studies.
merged_df['Human_Study_Indicator'] = None

for index, row in merged_df.iterrows():
    print(index)
    title = row['Title']
    abstract = row['Abstract']
    human_study_indicator = filter_human_studies(title, abstract)

    # Add the indicator to the new column.
    merged_df.at[index, 'Human_Study_Indicator'] = human_study_indicator

    # if human_study_indicator == 1:
    #     print(f"Paper: {title}, Included: Yes")
    # else:
    #     print(f"Paper: {title}, Included: No")

# Filter the DataFrame to keep only papers based on human studies
human_studies_df = merged_df[merged_df['Human_Study_Indicator'] == 1]

##### Filter out papers not relevant to adult men
def filter_adult_male_studies(title, abstract):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in identifying the target population of health research studies."},
            {"role": "user", "content": f"""Title: {title}\nAbstract: {abstract}\nDetermine if this study is relevant to adult males, and not primarily relevant to children or females. Respond with 'Relevant: Yes' if it is, otherwise 'Relevant: No'."""}
        ]
    )

    response = completion.choices[0].message.content
    is_relevant_to_adult_males = 'Yes' in response
    return 1 if is_relevant_to_adult_males else 0

# Apply the function to each row in merged_df and filter out irrelevant studies.
human_studies_df['Adult_Male_Relevance'] = None

for index, row in human_studies_df.iterrows():
    print(index)
    title = row['Title']
    abstract = row['Abstract']
    adult_male_relevance = filter_adult_male_studies(title, abstract)

    # Add the relevance indicator to the new column.
    human_studies_df.at[index, 'Adult_Male_Relevance'] = adult_male_relevance

    # if adult_male_relevance == 1:
    #     print(f"Paper: {title}, Included: Yes")
    # else:
    #     print(f"Paper: {title}, Included: No")

# Filter the DataFrame to keep only papers relevant to adult males
relevant_to_adult_males_df = human_studies_df[human_studies_df['Adult_Male_Relevance'] == 1]

##### filter out papers no relevant to the average man
def filter_general_male_health_studies(title, abstract):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in evaluating the broad applicability of health research studies."},
            {"role": "user", "content": f"""Title: {title}\nAbstract: {abstract}\nDetermine if this study can be used to improve the health of a large number of men. Exclude studies focused on specific conditions affecting only a small percentage of the population. Respond with 'Broadly Applicable: Yes' if it is broadly applicable, otherwise 'Broadly Applicable: No'."""}
        ]
    )

    response = completion.choices[0].message.content
    is_broadly_applicable = 'Yes' in response
    return 1 if is_broadly_applicable else 0

# Apply the function to each row in relevant_to_adult_males_df and filter out narrowly focused studies.
relevant_to_adult_males_df['Broad_Applicability'] = None

for index, row in relevant_to_adult_males_df.iterrows():
    print(index)
    title = row['Title']
    abstract = row['Abstract']
    broad_applicability = filter_general_male_health_studies(title, abstract)

    # Add the broad applicability indicator to the new column.
    relevant_to_adult_males_df.at[index, 'Broad_Applicability'] = broad_applicability

    #if broad_applicability == 1:
        #print(f"Paper: {title}, Included: Yes")
    #else:
        #print(f"Paper: {title}, Included: No")

# Filter the DataFrame to keep only broadly applicable papers
broadly_applicable_df = relevant_to_adult_males_df[relevant_to_adult_males_df['Broad_Applicability'] == 1]

##### filter just to papers relevant to a lifestyle or pharmacalogical intervention
def filter_lifestyle_pharmacological_studies(title, abstract):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in identifying the focus of health intervention studies."},
            {"role": "user", "content": f"""Title: {title}\nAbstract: {abstract}\nDetermine if this study describes a health impact achievable through lifestyle or pharmacological interventions, excluding surgical interventions. Respond with 'Intervention Applicable: Yes' if it meets these criteria, otherwise 'Intervention Applicable: No'."""}
        ]
    )

    response = completion.choices[0].message.content
    is_intervention_applicable = 'Yes' in response
    return 1 if is_intervention_applicable else 0

# Apply the function to each row in broadly_applicable_df and filter.
broadly_applicable_df['Intervention_Applicability'] = None

for index, row in broadly_applicable_df.iterrows():
    print(index)
    title = row['Title']
    abstract = row['Abstract']
    intervention_applicability = filter_lifestyle_pharmacological_studies(title, abstract)

    # Add the intervention applicability indicator to the new column.
    broadly_applicable_df.at[index, 'Intervention_Applicability'] = intervention_applicability

    #if intervention_applicability == 1:
        #print(f"Paper: {title}, Included: Yes")
    #else:
        #print(f"Paper: {title}, Included: No")

# Filter the DataFrame to keep only papers with applicable interventions
intervention_applicable_df = broadly_applicable_df[broadly_applicable_df['Intervention_Applicability'] == 1]


##############################################################################################
##      Create the LLM relevance score for papers that match our filter criteria
##############################################################################################

# Define a function to get the LLM score for a single paper.
def get_llm_score(title, abstract):
    prompt = f"""Title: {title}\nAbstract: {abstract}\n
    Evaluate this paper based on the following criteria:
    1) Practical applicability for the average man,
    2) Clarity of results,
    3) Broad relevance to men's health.
    Consider giving a wide range of scores based on how well each paper meets these criteria. Avoid giving similar scores to all papers. Then, calculate an overall score out of 100 by averaging these three scores.
    Respond with the overall score in the format: 
    'Overall Score: [n]', where [n] is an integer from 0 to 100 based on the individual categories. Give a paper a low score if it is weak in any categories'"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a health researcher."},
            {"role": "user", "content": prompt}
        ]
    )

    response = completion.choices[0].message.content
    match = re.search(r'Overall Score:\s*(\d+)', response)
    print(f"Paper: {title}, Included: {response}")
    if match:
        return int(match.group(1))
    else:
        # Find all two-digit numbers in the response
        all_two_digit_numbers = re.findall(r'\b\d{2}\b', response)
        all_two_digit_numbers = [int(num) for num in all_two_digit_numbers]
        
        if all_two_digit_numbers:
            # Return the lowest two-digit number
            return min(all_two_digit_numbers)
        else:
            return None

# Scoring each paper
intervention_applicable_df['LLM_Score'] = intervention_applicable_df.apply(lambda row: get_llm_score(row['Title'], row['Abstract']), axis=1)


####### RELEVANCE FACTOR RANKINGS and SORTING

# Calculate log of impact factor.
intervention_applicable_df['Log_Impact_Factor'] = np.log(intervention_applicable_df['ImpactFactor'])

# Calculate relevance score.
intervention_applicable_df['Relevance_Score'] = intervention_applicable_df['LLM_Score'] + intervention_applicable_df['Log_Impact_Factor']

intervention_applicable_df.to_csv('ranked_studies.csv', index=False)

##############################################################################################
##      Get similar previous studies for the top 20 papers
##############################################################################################
intervention_applicable_df=pd.read_csv('ranked_studies.csv')

# Sort by relevance score in descending order.
df_sorted = intervention_applicable_df.sort_values(by='Relevance_Score', ascending=False)
top_papers = df_sorted.head(20).reset_index(drop=True)



# Load original data and its embeddings
original_data = Utils.load_data('/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/all_studies.csv')
original_embeddings = pd.read_csv('/Users/nicholasrenegar/Library/CloudStorage/Dropbox/SHAPER Newsletter/Data/embeddings_all_studies_titles.csv').values

# Load new studies data (studies_df)
# Assuming studies_df is already loaded or provided in the script, if not, load it here
# studies_df = pd.read_csv('path_to_new_studies.csv')

# Create embeddings for the new studies
new_studies_embeddings = Utils.create_embeddings(top_papers)

# Find the five nearest neighbors for each study in studies_df
nearest_neighbors_all = {}
for idx, embedding in enumerate(new_studies_embeddings):
    nearest_neighbor_indices = Utils.find_nearest_neighbors(original_embeddings, embedding, n_neighbors=5)
    nearest_neighbors_df = original_data.iloc[nearest_neighbor_indices]
    nearest_neighbors_all[idx] = nearest_neighbors_df



##############################################################################################
##      Write Newsletter 
##############################################################################################


date_string = datetime.now().strftime("%B %d, %Y")  # Formats date as 'Month Day, Year'

# HTML newsletter with logo at the top and a matching background color
newsletter_html = f"""
<html>
<head>
<style>
  body {{
    background-color: #f7f1e8;
    font-family: 'Georgia', serif; /* Example of a serif font */
  }}
  .container {{
    background-color: #f3ebde;
    width: 80%;
    max-width: 800px;
    margin: 20px auto;
    padding: 40px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1); /* subtle shadow for depth */
    border-radius: 8px; /* rounded corners */
  }}
  .header {{
    text-align: center;
    padding-bottom: 20px;
  }}
  h1 {{
    font-family: 'Helvetica', sans-serif; /* Example of a sans-serif font */
    font-size: 48px;
    color: #333333; /* Dark grey for contrast */
  }}
  h2 {{
    font-family: 'Helvetica', sans-serif;
    font-size: 24px;
    color: #555555;
  }}
  p {{
    font-size: 18px;
    color: #666666;
    line-height: 1.6;
  }}
  .journal-authors {{
    font-style: italic;
    margin-bottom: 10px;
  }}
  .abstract, .experts-take {{
    margin-bottom: 20px;
  }}
  .footer {{
    text-align: center;
    padding-top: 20px;
    font-size: 12px;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <img src="images/SHAPER_Logo.jpg" alt="SHAPER Logo" style="max-width: 80%; height: auto;">
      <h1>SHAPER Health</h1>
      <h2 style="font-size: 32px; margin-top: 0; font-weight: normal;">Bringing You the Latest Research in Men's Health, Fitness, and Longevity</h2>
      <p class="date">{date_string}</p>
    </div>
    <!-- Add content here -->
    <div class="footer">
      <p>© {date_string.split(' ')[2]} SHAPER Health. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
"""



# Experts' names
experts = ["SHAPER", "SHAPER", "SHAPER"]
expert_idx = 0 

client = OpenAI()
def get_experts_take(title, abstract, expert, nearest_neighbors_df):
    # Create a summary of the nearest neighbors
    neighbors_summary = "The nearest neighbors of this paper are: "
    for idx, row in nearest_neighbors_df.iterrows():
        neighbors_summary += f"\n- Title: {row['Title']}, Abstract: {row['Abstract']}"

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "We want to give our expert takes on this paper for our newsletter, which aims to improve men's health, quality of life, and longevity."},
            {"role": "user", "content": f"Consider the following paper:\nTitle: {title}\nAbstract: {abstract}\n\n\nBriefly describe in two sentences why this paper is useful for readers and where the results fit in the research. Some past research which may be useful is listed below: \n{neighbors_summary}"}
        ]
    )
    return completion.choices[0].message.content


for index, row in top_papers.iterrows():
    print(index)
    expert_take = get_experts_take(row['Title'], row['Abstract'], experts[expert_idx], nearest_neighbors_all[index])
    
    newsletter_html += f"<h2>{row['Title']}</h2>"
    newsletter_html += f"<p><strong>Journal:</strong> {row['Journal']}</p>"
    newsletter_html += f"<p><strong>Authors:</strong> {row['Authors']}</p>"
    newsletter_html += f"<p><strong>Abstract:</strong> {row['Abstract']}</p>"
    newsletter_html += f"<p><strong>{experts[expert_idx]}'s Take:</strong> {expert_take}</p>"
    newsletter_html += "<hr>"
    
    # Rotate to the next expert
    expert_idx = (expert_idx + 1) % len(experts)


# Write the HTML content to a file
with open('newsletter_preview.html', 'w') as file:
    file.write(newsletter_html)

# Inform the user
print("The newsletter has been saved as 'newsletter_preview.html'. Open this file in a web browser to view the content.")
