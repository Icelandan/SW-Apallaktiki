FROM python:3.10.11-slim-bullseye
WORKDIR /app
COPY libraries.txt .
RUN pip install -r libraries.txt
EXPOSE 8501
COPY . .
CMD ["streamlit", "run", "app.py"]