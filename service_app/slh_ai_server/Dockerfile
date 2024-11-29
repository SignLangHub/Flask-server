# 베이스 이미지 설정
FROM --platform=linux/amd64 python:3.8-slim-buster as build

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir Flask==2.0.2 Werkzeug==2.0.3 openai==0.28 mysql-connector-python==8.0.33 \
    opencv-python-headless==4.5.5.64 mediapipe==0.10.0 jamo==0.4.1

# 애플리케이션 코드 복사
COPY . .

# 애플리케이션 실행
CMD ["python", "slh_ai_server.py"]