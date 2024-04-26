import urllib.request
from io import BytesIO
import PyPDF2


def downloader(url):
    response = urllib.request.urlopen(url)
    pdf = response.read()
    text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf))
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# print(downloader("https://fms.aktu.ac.in/Resources/aktu/pdf/syllabus/Syllabus2324/B.Tech_2nd_Yr_CSE_v3.pdf"))

