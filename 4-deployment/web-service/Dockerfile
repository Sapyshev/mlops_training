FROM python:3.9-slim


RUN pip install -U pip
RUN pip install pipenv
RUN pip install spacy
RUN pip install scikit-learn==1.3.0
RUN python -m spacy download en_core_web_lg

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "svm_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]