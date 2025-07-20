from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader(
    "books/Robust process control.pdf",
    mode = "page",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser(),
)
docs = loader.load()
for doc in docs[30:40]:
    print(doc.page_content)
