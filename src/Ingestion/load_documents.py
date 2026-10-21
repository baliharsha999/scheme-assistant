from pathlib import Path
import fitz
import re

raw_dir = Path("data/raw")

def clean_text(text: str) -> str:
    """Clean extracted PDF text without removing meaningful content."""

    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def table_to_text(table_data) -> str:
    """Convert a table into text while preserving column relationships."""

    if not table_data:
        return ""

    headers = table_data[0]
    headers = [
        str(header).strip() if header is not None else ""
        for header in headers
    ]

    rows = []

    for row in table_data[1:]:
        fields = []

        for column_name, value in zip(headers, row):
            column_name = column_name.strip()
            value = str(value).strip() if value is not None else ""

            if not value:
                continue

            if column_name:
                fields.append(f"{column_name}: {value}")
            else:
                fields.append(value)

        if fields:
            rows.append("\n".join(fields))
    if not rows and len(table_data) == 1:
        fields = [
            str(value).strip()
            for value in headers
            if value
        ]

        if fields:
            rows.append("\n".join(fields))

    return "\n\n".join(rows)

def load_documents():
    """Extract ordinary text and tables from PDFs."""

    documents = []

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {raw_dir.resolve()}"
        )
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {raw_dir.resolve()}")
        return documents

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        scheme_name = pdf_file.stem

        print(f"Processing: {pdf_file.name}")

        try:
            with fitz.open(pdf_file) as pdf:

                for page_number, page in enumerate(pdf, start=1):

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
                    try:
                        tables = page.find_tables()

                        for table_index, table in enumerate(
                            tables.tables,
                            start=1
                        ):
                            table_data = table.extract()

                            table_text = clean_text(
                                table_to_text(table_data)
                            )

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

                    except Exception as exc:
                        print(
                            f"Warning: Could not extract tables "
                            f"from {pdf_file.name}, page {page_number}: "
                        )

        except Exception as exc:
            print(
                f"Error processing {pdf_file.name}: {exc}"
            )

    print(f"Total documents extracted: {len(documents)}")

    return documents

def document_loaders():
    """Load all PDF documents for the chunking pipeline."""

    return load_documents()


if __name__ == "__main__":

    documents = document_loaders()
    print(f"Total extracted documents: {len(documents)}")

