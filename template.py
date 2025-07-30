import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


list_of_files = [
    "api/index.py",
    "vercel.json",
    "requirements.txt"
]


for file_path in list_of_files:
    file_path = Path(file_path)
    filedir, filename = os.path.split(file_path)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
    
    if (not os.path.exists(file_path)) or  (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:
            if filename == "chat.html":
                f.write("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Medical Chatbot</title>\n</head>\n<body>\n    <h1>Welcome to the Medical Chatbot</h1>\n</body>\n</html>")
            else:
                f.write("# Placeholder for " + filename)
        logging.info(f"Created file: {file_path}")