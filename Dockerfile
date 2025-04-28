FROM python:3.13.3-bookworm

WORKDIR /app

COPY main.py ./main.py
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

CMD [ "python", "./main.py"]
