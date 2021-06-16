FROM python:3.9-slim-buster
WORKDIR /app
EXPOSE 8501

COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /app

CMD ["streamlit","run","app.py","--server.port","8501"]