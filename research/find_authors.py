#%%

import pandas as pd
import re

data_input_path = "../../data/results/"
data_output_path = "../../data/results/"
ostrom_options = ["Ostrom, E.", "Elinor Ostrom"]

def find_author(authors_string, search_names):
    """
    Search for multiple author names in a string of author names, ignoring case and extra whitespace.
    
    Args:
        authors_string (str): String containing author names
        search_names (list): List of names to search for
        
    Returns:
        bool: True if any name is found, False otherwise
    """
    # Normalize the authors string by removing extra whitespace
    authors_string = ' '.join(authors_string.split())
    
    for search_name in search_names:
        # Normalize the search name
        search_name = ' '.join(search_name.split())
        pattern = re.compile(re.escape(search_name), re.IGNORECASE)
        if pattern.search(authors_string):
            return True
    return False

#%%

# read data
df = pd.read_csv(data_input_path + "cited_papers_info.csv")
df

# %%

# search for Ostrom
df["ostrom"] = df["Authors"].apply(lambda x: find_author(x, ostrom_options))
df.loc[df["ostrom"]].to_csv(data_output_path + "citaciones_ostrom.csv", index=False)

# %%

df.loc[df["ostrom"]]["origin_paper"].unique().shape