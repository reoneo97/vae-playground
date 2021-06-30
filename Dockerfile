FROM python:3.8-slim-buster
WORKDIR /app
EXPOSE $PORT

COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /app

CMD streamlit run app.py --server.port $PORT