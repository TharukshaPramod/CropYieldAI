# utils/security.py
import os
import re
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()

# ----------------------------
# Key Management
# ----------------------------
def _get_key_from_env():
    k = os.getenv("ENCRYPT_KEY")
    return k.encode() if k else None

KEY = _get_key_from_env()
if KEY is None:
    KEY = Fernet.generate_key()
    print("WARNING: ENCRYPT_KEY not found in .env — generated temporary key for this session.")

fernet = Fernet(KEY)

# ----------------------------
# Utility Functions
# ----------------------------
def sanitize_input(text: str) -> str:
    """Clean and normalize user input to prevent injection or invalid chars."""
    if text is None:
        return ""
    text = str(text)
    # Allow basic punctuation, keep "tons/ha"
    text = re.sub(r"[^\w\s\.\,\/\:\%\-\(\)]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encrypt(data: str) -> str:
    """Encrypt plain text with Fernet key."""
    if not data:
        return ""
    return fernet.encrypt(str(data).encode()).decode()

def decrypt(token: str) -> str:
    """Decrypt Fernet token safely."""
    if not token:
        return ""
    try:
        return fernet.decrypt(str(token).encode()).decode()
    except (InvalidToken, Exception):
        return token

def decrypt_embedded_tokens(text: str) -> str:
    """
    Search for any Fernet-encrypted tokens embedded in text and decrypt them.
    Example: 'Predicted yield: gAAAA...==' -> 'Predicted yield: 5.4 tons/ha'
    """
    if not text:
        return ""
    matches = re.findall(r"gAAAA[A-Za-z0-9_\-]+", text)
    for token in matches:
        decrypted = decrypt(token)
        text = text.replace(token, decrypted)
    return text
