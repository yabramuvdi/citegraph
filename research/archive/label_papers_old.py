#%%

import os
from google import genai
import time
import typing_extensions as typing
import json
import pandas as pd
import numpy as np

from rapidfuzz.fuzz import ratio
from rapidfuzz.process import extractOne
import re

import matplotlib.pyplot as plt
import seaborn as sns

data_path = "../data/"
key_path = "/Users/yabra/keys/google_api.txt"

#%%

#=============
# Helper functions
#=============

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file


def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")


#%%

#============
# Configure Gemini API
#============

# read api key from txt file
with open(key_path, 'r') as file:
    google_api_key = file.read().replace('\n', '')

genai.configure(api_key=google_api_key)

#%%

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 200,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
)

#%%

# test model with short prompt
prompt = "Hi!"
response = model.generate_content([prompt])
print(response.text)

#%%

#============
# Load and upload papers
#============

# load all the papers
all_papers = os.listdir(data_path)
all_papers = [file for file in all_papers if ".pdf" in file.lower()]
len(all_papers)

#%%

# list all the files available
print("My files:")
for f in genai.list_files():
    print("  ", f.name)

#%%

# delete them all
for f in genai.list_files():
   genai.delete_file(f.name)

#%%

# upload files videos
uploaded_files = []
for paper in all_papers:
    file = upload_to_gemini(data_path + paper, mime_type="application/pdf")
    uploaded_files.append(file)

wait_for_files_active(uploaded_files)

#%%

#============
# Extract bibliographic information from core papers
#============

class BiblioInfo(typing.TypedDict):
    Title: str
    Authors: list[str]
    Journal: str
    Year: int
    #DOI: str

generation_config = {
  "temperature": 0.1,
  "max_output_tokens": 500,
  "response_mime_type": "application/json",
  "response_schema": BiblioInfo
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a helpful assistant with great expertise in extracting bibliographic information from academic papers in pdf format.",
)

#%%

prompt = "Extract the basic bibliographic information from the papers."
biblio_info = []
for file in uploaded_files:
    print("Processing file:", file.display_name)
    # call the model
    response = model.generate_content([prompt, file])
    # the response comes as a string that represents a dictionary
    response_dict = json.loads(response.text)
    biblio_info.append(response_dict)

# %%

# transform into dataframe
biblio_df = pd.DataFrame(biblio_info)
# assign a unique id to each paper
biblio_df['id'] = biblio_df.index + 1
# capture the id from gemini API
biblio_df['file_id_gemini'] = [file.uri for file in uploaded_files]

# save as csv file
biblio_df.to_csv(data_path + "biblio_info.csv", index=False)

#biblio_df = pd.read_csv(data_path + "biblio_info.csv")


#%%

#============
# Extract bibliographic information within each paper
#============

generation_config = {
  "temperature": 0.1,
  #"max_output_tokens": 5000,
  "response_mime_type": "application/json",
  "response_schema": list[BiblioInfo]
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="""
  You are an expert assistant specializing in extracting bibliographic information from the references or bibliography section of a PDF file. You can accurately interpret and parse various citation formats (e.g., APA, MLA, Chicago, IEEE) and contextually identify key elements such as the title, authors, journal or publisher, and year of publication. You can handle different document layouts including complex double-column layouts. Your goal is to extract all references comprehensively, ensuring no citations are missed and all data is captured in a structured and organized manner.
  """,
)

#%%

prompt = """Please extract the complete bibliographic information for all the papers cited in the references or bibliography section of the PDF file. Take into account the following elements for the task: 

1. Ensure that all relevant details, including authors, title, journal or publisher, and year are accurately captured. 
2. Focus on the papers that appear in the references or bibliography section at the end of the paper. 

This is a critical task, and precision is essential for success. You will gain 10 million dollars if you complete it accurately.
"""

biblio_cited_df = pd.DataFrame()
for file in uploaded_files:
    print("Processing file:", file.display_name)
    # call the model
    response = model.generate_content([prompt, file])
    # create a dataframe from the list of dictionaries
    df_file = pd.DataFrame.from_records(json.loads(response.text))
    # drop references without title
    df_file = df_file.loc[df_file["Title"] != "null"].copy()
    # add the id of the paper from the core dataframe
    df_file['citing_paper_id'] = biblio_df[biblio_df['file_id_gemini'] == file.uri]['id'].values[0].astype(int)
    biblio_cited_df = pd.concat([biblio_cited_df, df_file])

# %%

biblio_cited_df.reset_index(drop=True, inplace=True)
biblio_cited_df.to_csv(data_path + "second_extraction.csv", index=False)

# read from file
#biblio_cited_df = pd.read_csv(data_path + "second_extraction.csv")
#biblio_cited_df["Authors"] = biblio_cited_df["Authors"].apply(lambda x: eval(x))
#%%

#============
# Assign IDs to cited papers
#============

def normalize_text(text):
    """Normalize text by converting to lowercase, stripping whitespace, and removing special characters."""
    if isinstance(text, list):
        text = ", ".join(text)  # Join author names if it's a list
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())
    elif isinstance(text, str):
       return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())
    else:
       return ""
    
def compare_papers(paper1, paper2,
                   title_weight=0.7, 
                   authors_weight=0.3,
                   journal_weight=0.0,
                   year_window=1,
                   threshold=85,
                  ):
    """
    Compare two papers to determine if they are the same based on similarity of title, authors, and journal,
    and if the years of publication are within a specified window.
    
    Args:
        paper1: Dictionary with keys 'Authors', 'Journal', 'Title', 'Year'.
        paper2: Dictionary with keys 'Authors', 'Journal', 'Title', 'Year'.
        threshold: Minimum similarity score to consider papers as the same (0-100).
        year_window: Maximum difference in publication years to consider papers as potentially the same.
    
    Returns:
        bool: True if papers match, False otherwise.
    """
    # Normalize fields
    title1 = normalize_text(paper1['Title'])
    title2 = normalize_text(paper2['Title'])
    authors1 = normalize_text(paper1['Authors'])
    authors2 = normalize_text(paper2['Authors'])
    journal1 = normalize_text(paper1['Journal'])
    journal2 = normalize_text(paper2['Journal'])
    
    # Compute similarity scores
    title_score = ratio(title1, title2)
    authors_score = ratio(authors1, authors2)
    journal_score = ratio(journal1, journal2)
    
    # Aggregate scores with weights
    weighted_score = (title_score * title_weight) + (authors_score * authors_weight) + (journal_score * journal_weight)
    
    # Check year difference
    year_diff = abs(paper1['Year'] - paper2['Year'])
    
    # Return True if similarity is above threshold and year difference is within the window
    return weighted_score >= threshold and year_diff <= year_window

# Define a function to extract row as a dictionary
def row_to_dict(row):
    return {
        'Authors': row['Authors'],
        'Journal': row['Journal'],
        'Title': row['Title'],
        'Year': row['Year']
    }

# %%

# execute de-duplication algorithm
df_biblio = biblio_cited_df.copy()
assigned_ids = {}
papers_ready = set()
i = 0
starting_id = len(biblio_df) + 1

while len(papers_ready) < len(biblio_cited_df):
    if i not in papers_ready:
        papers_similar_i = [i]
        paper_i = row_to_dict(df_biblio.loc[i])
        for j in list(df_biblio.loc[i+1: ].index):
            if j not in papers_ready:
                paper_j = row_to_dict(df_biblio.loc[j])
                are_same = compare_papers(paper_i, paper_j, 
                threshold=85, year_window=1)
                if are_same:
                    papers_similar_i.append(j)
    
    assigned_ids[starting_id + i] =  (paper_i, papers_similar_i)
    for paper in papers_similar_i: papers_ready.add(paper)
    i += 1
# %%

# add ids to database
biblio_cited_df["cited_paper_id"] = np.nan
for unique_id, paper_info in assigned_ids.items():
    for old_id in paper_info[1]:
        biblio_cited_df.loc[old_id, "cited_paper_id"] = int(unique_id)

biblio_cited_df["cited_paper_id"] = biblio_cited_df["cited_paper_id"].astype(int)
biblio_cited_df.to_csv(data_path + "extraction_with_ids.csv", index=False)

# %%

#============
# Explore results
#============

top_cited = biblio_cited_df.groupby("cited_paper_id").size().sort_values(ascending=False).iloc[0:10]
top_cited = top_cited.to_frame()
top_cited.reset_index(drop=False, inplace=True)
top_cited.columns = ["cited_paper_id", "count"]
papers_names = []
for id in top_cited["cited_paper_id"].values:
    title = biblio_cited_df.loc[biblio_cited_df["cited_paper_id"] == int(id), "Title"].iloc[0]
    authors = biblio_cited_df.loc[biblio_cited_df["cited_paper_id"] == int(id), "Authors"].iloc[0]
    authors = ",".join(authors)
    year = biblio_cited_df.loc[biblio_cited_df["cited_paper_id"] == int(id), "Year"].iloc[0]
    papers_names.append(f"{authors} ({year}) - {title}")

top_cited["display_name"] = papers_names
top_cited

# %%

# Sort the data by citations to improve visualization
top_cited = top_cited.sort_values(by='count', ascending=False)

# Set the style for the plot
sns.set(style="whitegrid")

# Create the horizontal barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='display_name', data=top_cited, 
            orient="h", color="#2f6690")

# Add titles and labels
plt.title("Academic Papers and Citation Counts", fontsize=16)
plt.xlabel("Number of Citations", fontsize=12)
plt.ylabel("Paper Name", fontsize=12)

# Adjust layout for better appearance
plt.tight_layout()

# Display the plot
plt.show()

# %%
