import re
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os

load_dotenv()
KEY = os.getenv('ENCRYPT_KEY').encode()

def sanitize_input(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def encrypt(data: str) -> str:
    f = Fernet(KEY)
    return f.encrypt(data.encode()).decode()

def decrypt(encrypted: str) -> str:
    f = Fernet(KEY)
    return f.decrypt(encrypted.encode()).decode()