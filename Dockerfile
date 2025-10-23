FROM pathwaycom/pathway:latest
 
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pathway_app.py .

CMD ["python", "./pathway_app.py"]