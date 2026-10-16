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
                fields.append(f"{value}")

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

                tables = page.find_tables()
                #checking if any tables are there
                if tables.tables:
                     # Extract ordinary text
                    text = clean_text(page.get_text("text"))
                    
                    if text:
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": pdf_file.name,
                                "page": page_number,
                                "scheme": scheme_name,
                                "content_type": "text"
                            }
                        })
                    
                    # Extract each table separately
                    #table inde is for many tables can be found in one page
                    for table_index, table in enumerate(tables.tables, start=1):
                        table_data = table.extract()
                    
                        table_text = table_to_text(table_data)
                        table_text = clean_text(table_text)
                    
                        if table_text:
                            documents.append({
                                "text": table_text,
                                "metadata": {
                                    "source": pdf_file.name,
                                    "page": page_number,
                                    "scheme": scheme_name,
                                    "content_type": "table",
                                    "table_index": table_index
                                }
                            })
                #if no tables are found
                else:
                    text = clean_text(page.get_text("text"))
                
                    if text:
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": pdf_file.name,
                                "page": page_number,
                                "scheme": scheme_name,
                                "content_type": "text"
                            }
                        })
    return documents

if __name__ == "__main__":
    document_loaders = load_documents()
    print(f"found {len(document_loaders)} documents")