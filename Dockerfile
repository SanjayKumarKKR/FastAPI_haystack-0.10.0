FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz && \
    tar -xvf xpdf-tools-linux-4.03.tar.gz && cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

RUN apt-get install libpoppler-cpp-dev pkg-config -y --fix-missing

# Install Tesseract
RUN apt-get install tesseract-ocr libtesseract-dev poppler-utils -y

# copy code
# COPY haystack /home/user/haystack

# install as a package
COPY setup.py requirements.txt README.md /home/user/
RUN pip install farm-haystack==0.10.0
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3
RUN pip install s3fs
RUN pip install fastapi
RUN pip install PyMuPDF==1.16.14
RUN pip install fitz

RUN pip install -e .

# copy saved models
COPY README.md models* /home/user/models/

# Copy REST API code
COPY rest_api /home/user/rest_api

# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data

EXPOSE 8000

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app",  "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
