from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import fitz
import re

raw_dir = Path("data/raw")

# function for cleaning the text
def clean_text(text:str) ->str:
    text = text.strip()
    text = re.sub(r"[^\S\n]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    text = re.sub(r"\.{2,}", "...", text)
    return text

# function for extraction of tabular data
def table_to_text(table_data) -> str:
    if not table_data:
        return " "

    header = table_data[0]
    rows = []

    for row in table_data[1:]:
        fields = []

        for header ,  value in zip(header , row):
            header = str(header).strip() if header else ""
            value = str(value).strip() if value else  ""
            if header and value:
                fields.append(f"{header} : {value}")
            elif value:
                fields.appenf(f"{value}")

        if fields:
            rows.append("\n".join(fields))

    return "\n".join(rows)


# main function for document loading
def load_documents():
    documents = []

    pdf_files = list(raw_dir.glob("*.pdf"))

    for pdf_file in pdf_files:
        #scheme name for meta data filtering
        scheme_name = pdf_file.stem
        
        print(pdf_file)
        with fitz.open(pdf_file) as pdf:

            for page_number,page in enumerate(pdf , start =1):
                # cleaning text data purposes
                text = page.get_text("text")
                text = clean_text(text)

                if text:
                    documents.append({
                        "text" : text,
                        "metadata" : {
                            "source" : pdf_file.name,
                            "page" : page_number,
                            "scheme" : scheme_name,
                        }
                    })
    return documents

if __name__ == "__main__":
    document_loaders = load_documents()
    print(f"found {len(document_loaders)} documents")