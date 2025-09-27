# utils/security.py
import os
import re
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()

def _get_key_from_env():
    k = os.getenv("ENCRYPT_KEY")
    return k.encode() if k else None

KEY = _get_key_from_env()
if KEY is None:
    KEY = Fernet.generate_key()
    print("WARNING: ENCRYPT_KEY not found in .env — generated temporary key for this session.")

fernet = Fernet(KEY)

def sanitize_input(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # allow forward slash so "tons/ha" stays intact (and some common punctuation)
    text = re.sub(r"[^\w\s\.\,\/\:\%\-\(\)]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encrypt(data: str) -> str:
    if data is None:
        return ""
    return fernet.encrypt(str(data).encode()).decode()

def decrypt(token: str) -> str:
    if token is None:
        return ""
    try:
        return fernet.decrypt(str(token).encode()).decode()
    except (InvalidToken, Exception):
        return token

def decrypt_embedded_tokens(text: str) -> str:
    """
    Search for any Fernet-encrypted tokens inside `text` and replace them with decrypted content.
    Example: "Result: gAAAAA..." -> "Result: Decrypted content"
    """
    if not text:
        return ""
    # Simple pattern matching for typical Fernet tokens (start with gAAAA)
    matches = re.findall(r"gAAAA[A-Za-z0-9_\-]+=*", text)
    for token in set(matches):
        try:
            plain = decrypt(token)
            # replace all occurrences of token with decrypted text
            text = text.replace(token, plain)
        except Exception:
            # leave token as-is if decryption fails
            pass
    return text
