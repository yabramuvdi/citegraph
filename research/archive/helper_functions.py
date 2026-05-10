import re
import pandas as pd

def fix_incomplete_json_string(input_string):
    """
    Fix an incomplete JSON string by:
    1. Removing surrounding single quotes if present
    2. Finding the last complete object (ending with "}")
    3. Properly closing the JSON array
    
    Args:
        input_string: The incomplete JSON string
        
    Returns:
        A properly formatted JSON string that can be parsed
    """
    # Remove surrounding single quotes if present
    cleaned_string = input_string
    if cleaned_string.startswith("'"):
        cleaned_string = cleaned_string[1:]
    if cleaned_string.endswith("'"):
        cleaned_string = cleaned_string[:-1]
    
    # Find the last complete object
    last_complete_object_end = cleaned_string.rfind("}")
    
    if last_complete_object_end != -1:
        # Get the substring up to the last complete object
        cleaned_string = cleaned_string[:last_complete_object_end + 1]
        
        # Close the JSON array
        cleaned_string += "\n]"
        
        return cleaned_string
    else:
        print("Could not find a complete object end!")
        return None
    

def transform_date(date_str, current_year=2025):
    """
    Transform dates from any D.M format to DD.MM.YYYY format.
    Handles D.M, D.MM, DD.M, and DD.MM formats, padding as needed.
    
    Parameters:
    -----------
    date_str : str or None
        Date string in any D.M format, or None/NaN for missing values
    current_year : int, optional
        Year to append to the date, defaults to 2025
        
    Returns:
    --------
    str or None
        Date in DD.MM.YYYY format, or None for invalid/missing dates
    """
    # Handle NaN, None or empty string
    if pd.isna(date_str) or date_str == '':
        return None
    
    # Convert to string if it's not already
    if not isinstance(date_str, str):
        date_str = str(date_str)
    
    # Remove additional characters
    if date_str[-1] == ".":
        date_str = date_str[:-1]

    # Check if already has year (D.M.YYYY or DD.MM.YYYY format)
    full_date_match = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', date_str)
    if full_date_match:
        day = full_date_match.group(1).zfill(2)  # Pad day with leading zero if needed
        month = full_date_match.group(2).zfill(2)  # Pad month with leading zero if needed
        year = full_date_match.group(3)
        return f"{day}.{month}.{year}"
    
    # Check for D.M or DD.MM format (without year)
    partial_match = re.match(r'^(\d{1,2})\.(\d{1,2})$', date_str)
    if partial_match:
        day = partial_match.group(1).zfill(2)  # Pad day with leading zero if needed
        month = partial_match.group(2).zfill(2)  # Pad month with leading zero if needed
        return f"{day}.{month}.{current_year}"
    
    return None