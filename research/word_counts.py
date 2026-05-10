#%%

import os
import re
import yaml 
from langdetect import detect
import pandas as pd

data_input_path = "../../data/markdown/"
data_output_path = "../../data/results/"

# read dictionaries (now lives next to this script in research/)
with open('dicts.yaml', 'r') as file:
    dictionaries = yaml.safe_load(file)

def infer_language(text):
    """
    Detect the language of a given text using langdetect
    Returns ISO 639-1 language code (e.g. 'en', 'es', 'pt')
    """
    try:
        return detect(text)
    except:
        return 'unknown'

#%%

#============
# Simple method
#============


def find_terms_in_text(text, terms_list):
    """
    Find occurrences of terms in a text string, case-insensitive
    Args:
        text (str): Input text to search in
        terms_list (list): List of terms to search for
    Returns:
        list: Terms that were found in the text
    """
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()
    
    # Initialize list for found terms
    found_terms = []
    
    # Search for each term
    for term in terms_list:
        term = term.lower()
        if term in text:
            found_terms.append(term)
            
    return found_terms

#%%

# load the info from the papers
df_papers = pd.read_csv(data_output_path + "base_papers_info.csv")
# %%

# for file in df_papers["file_name"].values:
#     # Read the content of the markdown file
#     with open(os.path.join(data_input_path, file), 'r', encoding='utf-8') as f:
#       paper_content = f.read()
#     # Detect language to decide on the dictionaries
#     lang = infer_language(paper_content)
#     if lang == "en":
#         general_terms = find_terms_in_text(paper_content, dictionaries["en_main"])
#         context_terms = find_terms_in_text(paper_content, dictionaries["en_context"])
#     elif lang == "es":
#         general_terms = find_terms_in_text(paper_content, dictionaries["es_main"])
#         context_terms = find_terms_in_text(paper_content, dictionaries["es_context"])
#     else:
#         print(f"Language not supported: {lang}")
    

# %%

#============
# My library approach
#============

from dictionary_methods.methods import Dictionary
from difflib import SequenceMatcher
from collections import Counter

def find_closest_matches(a, b):
    """
    For each string in list a, find the closest matching string in list b.
    
    Args:
        a (list): List of strings to find matches for
        b (list): List of potential matching strings
        
    Returns:
        list: A list of closest matches from b for each string in a
    """

    
    # Function to calculate similarity ratio between two strings
    def similarity_ratio(str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()
    
    results = []
    
    for item_a in a:
        # If b is empty, we can't find a match
        if not b:
            results.append(None)
            continue
            
        # Find the string in b with the highest similarity ratio
        closest_match = max(b, key=lambda item_b: similarity_ratio(item_a, item_b))
        results.append(closest_match)
    
    return results

#%%

language = []
general_terms_found = []
context_terms_found = []
general_dict_count = []
context_dict_count = []
for file in df_papers["file_name"].values:
    # Read the content of the markdown file
    with open(os.path.join(data_input_path, file), 'r', encoding='utf-8') as f:
      paper_content = f.read()
    # Detect language to decide on the dictionaries
    lang = infer_language(paper_content)
    language.append(lang)
    if lang == "en":
        #### 1. find general terms
        dict_general_terms = Dictionary(
            terms=dictionaries["en_main"],
            part_of_word=None,
            ignore_case=True,
            flexible_multi_word=False,
            search_type="all",
            return_matches=True)
        found, matches = dict_general_terms.tag_text(paper_content)
        general_dict_count.append(len(matches))
        if found:
            matches_terms = [m[0] for m in matches]
            matched_terms = find_closest_matches(matches_terms, dictionaries["en_main"])
            count_matches = Counter(matched_terms)
            general_terms_found.append(count_matches)
        else:
            general_terms_found.append("")
        
        #### 2. find context terms
        dict_context_terms = Dictionary(
            terms=dictionaries["en_context"],
            part_of_word=None,
            ignore_case=True,
            flexible_multi_word=False,
            search_type="all",
            return_matches=True)
        found, matches = dict_context_terms.tag_text(paper_content)
        context_dict_count.append(len(matches))
        if found:
            matches_terms = [m[0] for m in matches]
            matched_terms = find_closest_matches(matches_terms, dictionaries["en_context"])
            count_matches = Counter(matched_terms)
            context_terms_found.append(count_matches)
        else:
            context_terms_found.append("")
                    
    elif lang == "es":
        #### 1. find general terms
        dict_general_terms = Dictionary(
            terms=dictionaries["es_main"],
            part_of_word=None,
            ignore_case=True,
            flexible_multi_word=False,
            search_type="all",
            return_matches=True)
        found, matches = dict_general_terms.tag_text(paper_content)
        general_dict_count.append(len(matches))
        if found:
            matches_terms = [m[0] for m in matches]
            matched_terms = find_closest_matches(matches_terms, dictionaries["es_main"])
            count_matches = Counter(matched_terms)
            general_terms_found.append(count_matches)
        else:
            general_terms_found.append("")
        
        #### 2. find context terms
        dict_context_terms = Dictionary(
            terms=dictionaries["es_context"],
            part_of_word=None,
            ignore_case=True,
            flexible_multi_word=False,
            search_type="all",
            return_matches=True)
        found, matches = dict_context_terms.tag_text(paper_content)
        context_dict_count.append(len(matches))
        if found:
            matches_terms = [m[0] for m in matches]
            matched_terms = find_closest_matches(matches_terms, dictionaries["es_context"])
            count_matches = Counter(matched_terms)
            context_terms_found.append(count_matches)
        else:
            context_terms_found.append("")
    
    else:
        print(f"Language not supported: {lang}")

# %%

# organize dataframe
df_papers["lang"] = language
df_papers["general_dict_count"] = general_dict_count
df_papers["general_dict_terms"] = general_terms_found
df_papers["context_dict_count"] = context_dict_count
df_papers["context_dict_terms"] = context_terms_found
# save data
df_papers.to_csv(data_output_path + "keywords_papers.csv", index=False)

# %%
