#%%

import os
from google import genai
from google.genai import types
import time
import typing_extensions as typing
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np

# Get the path to the src directory
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)
from helper_functions import fix_incomplete_json_string


data_input_path = "../data/markdown/"
data_output_path = "../data/results/"
key_path = "/Users/yabra/keys/google_api.txt"
#model_name = "gemini-2.0-flash"
model_name = "gemini-2.5-flash-preview-04-17"

#%%

#============
# Configure Gemini API
#============

# read api key from txt file
with open(key_path, 'r') as file:
    google_api_key = file.read().replace('\n', '')

# Initialize the Google Generative AI client
client = genai.Client(api_key=google_api_key)

#%%

# test model with short prompt
response = client.models.generate_content(
    model=model_name,
    contents=['Hi!']
)
print(response.text)

#%%

#============
# Find papers
#============

# load all the papers
all_papers = os.listdir(data_input_path)
all_papers = [file for file in all_papers if ".md" in file.lower()]
len(all_papers)

#%%

#============
# Extract bibliographic information from core papers
#============

# class BiblioInfo(typing.TypedDict):
#     Title: str
#     Authors: list[str]
#     Journal: str
#     Year: int
#     #DOI: str

class BiblioInfo(BaseModel):
    Title: str
    Authors_List: list[str]
    Authors: str
    Journal: str
    Year: int

#%%

all_results = []
for file in all_papers:
    print("Processing file:", file)
    # Read the content of the markdown file
    with open(os.path.join(data_input_path, file), 'r', encoding='utf-8') as f:
      paper_content = f.read()
    # Call the model
    prompt = f"""
    Extract the basic bibliographic information from the following paper. The information should include:
    - Title: The full title of the paper
    - Authors_List: A list of all authors
    - Authors: All authors as a single string
    - Journal: The journal or venue where the paper was published
    - Year: The publication year as an integer

    Make sure to not include any dirty characters that might have come from problems in the PDF.

    Here is the paper content:
    {paper_content}
    """

    response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config={
            "system_instruction": "You are a helpful assistant with great expertise in extracting bibliographic information from academic papers in markdown format.",
            "response_mime_type": "application/json",
            "response_schema": BiblioInfo,
            "max_output_tokens": 500
            }
    )
    all_results.append(response)

#%%

# organize
biblio_info = []
for result in all_results:
    biblio_info.append(result.parsed.model_dump())
    
# transform into dataframe
biblio_df = pd.DataFrame(biblio_info)
# assign a unique id to each paper
biblio_df['id'] = biblio_df.index + 1
# add markdown file name
biblio_df["file_name"] = all_papers

# check if there is duplications
biblio_df.drop_duplicates(subset="Title", inplace=True)

# save as csv file
biblio_df.to_csv(data_output_path + "base_papers_info.csv", index=False)

#%%

#============
# Extract bibliographic information within each paper
#============

system_instruction="""
You are an expert assistant specializing in extracting bibliographic information from the references or bibliography section of a PDF file. You can accurately interpret and parse various citation formats (e.g., APA, MLA, Chicago, IEEE) and contextually identify key elements such as the title, authors, journal or publisher, and year of publication. You can handle different document layouts including complex double-column layouts. Your goal is to extract all references comprehensively, ensuring no citations are missed and all data is captured in a structured and organized manner.
"""

all_results = []
for file in all_papers:
    print("Processing file:", file)
    # Read the content of the markdown file
    with open(os.path.join(data_input_path, file), 'r', encoding='utf-8') as f:
      paper_content = f.read()
    # Call the model
    prompt = f"""

# Instructions
Please extract the complete bibliographic information for all the papers cited in the references or bibliography section of the PDF file. The information should include:

- Title: The full title of the paper
- Authors_List: A list of all authors
- Authors: All authors as a single string
- Journal: The journal or venue where the paper was published
- Year: The publication year as an integer


Take into account the following elements for the task: 

1. Ensure that all relevant information is accurately captured. 
2. Focus on the papers that appear in the references or bibliography section at the end of the paper.
3. Make sure to not include any dirty characters that might have come from problems in the PDF.

This is a critical task, and precision is essential for success. You will gain 10 million dollars if you complete it accurately.

# Paper
{paper_content}
    """

    response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": types.Schema(type='array', items=BiblioInfo.model_json_schema()),
#            "max_output_tokens": 8192
            "max_output_tokens": 40000
            }
    )
    all_results.append(response)

# %%

biblio_cited = []
num_references = []
for i, result in enumerate(all_results):
    print(i)
    if result.parsed is not None:
        paper_biblio: list[BiblioInfo] = result.parsed
        biblio_cited.extend(paper_biblio)
        num_references.append(len(paper_biblio))
    else:
        print("Problem")

#%%

# save the extracted citations as a CSV file
df_citations = pd.DataFrame(biblio_cited)

# add a reference to the paper in which the citation was done
paper_origin = []
for paper_id, num_ref in zip(biblio_df["id"].values, num_references):
    paper_origin.extend([paper_id]*num_ref)

df_citations["origin_paper"] = paper_origin 
df_citations.to_csv(data_output_path + "cited_papers_info.csv", index=False)
# %%
