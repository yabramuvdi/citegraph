#%%

import fitz  # PyMuPDF
pdf_path = "../../data/Bogliacino et al_crime related exposure.pdf"


# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# text = extract_text_from_pdf(pdf_path)
# with open("output.txt", "w") as file:
#     file.write(text)

# %%


def extract_columns(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc, start=1):
        # Define regions for the two columns
        left_column = fitz.Rect(0, 0, page.rect.width / 2, page.rect.height)
        right_column = fitz.Rect(page.rect.width / 2, 0, page.rect.width, page.rect.height)
        
        # Extract text from each column
        left_text = page.get_text("text", clip=left_column)
        right_text = page.get_text("text", clip=right_column)
        
        all_text.append(f"Page {page_num}\n\nLeft Column:\n{left_text}\n\nRight Column:\n{right_text}")
    
    return "\n".join(all_text)

text = extract_columns(pdf_path)
with open("output.txt", "w") as file:
    file.write(text)
