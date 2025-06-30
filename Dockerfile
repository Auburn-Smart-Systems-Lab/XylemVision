FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install gdown

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /app/weight/SAM /app/weight/YOLO

RUN gdown https://drive.google.com/uc?id=16QARfz1cpumYtwBSf23nlBWtr3hweTQy -O /app/weight/SAM/sam_vit_l_0b3195.pth
RUN gdown https://drive.google.com/uc?id=1VuYeIrlKAVg_2QMs4L00ndToZVjjMeTd -O /app/weight/YOLO/best.pt

COPY . /app/

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "root.wsgi:application", "--bind", "0.0.0.0:8000"]
