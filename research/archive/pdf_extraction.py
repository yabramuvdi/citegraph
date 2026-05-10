#%%

from docling.document_converter import DocumentConverter
import os

data_path = "../data/pdfs/"
output_path = "../data/markdown/"

# load all the papers
all_papers = os.listdir(data_path)
all_papers = [file for file in all_papers if ".pdf" in file.lower()]
len(all_papers)

#%%

for file in all_papers:
    print(f"processing file: {file}")
    source = data_path + file
    converter = DocumentConverter()
    result = converter.convert(source)
    output_file = os.path.splitext(file)[0] + ".md"
    with open(output_path + output_file, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())

# %%
