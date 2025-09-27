# utils/db.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///crop.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

ENGINE = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)
Base = declarative_base()
SessionLocal = sessionmaker(bind=ENGINE)

# try relative import (works when run as package); fallback to absolute
try:
    from .security import encrypt, decrypt
except Exception:
    from utils.security import encrypt, decrypt

class CropData(Base):
    __tablename__ = "crop_data"
    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String(128))
    location = Column(String(128))
    data = Column(Text)

def init_db():
    Base.metadata.create_all(bind=ENGINE)

def load_sample_data(csv_path: str = "data/crop_yield_data_sampled.csv", clear_table: bool = True):
    init_db()
    session = SessionLocal()
    try:
        if clear_table:
            session.query(CropData).delete()
            session.commit()
        df = pd.read_csv(csv_path)
        inserted = 0
        for _, row in df.iterrows():
            crop = row.get("Crop") or row.get("crop") or "Unknown"
            location = row.get("Region") or row.get("region") or "Unknown"
            data = (
                f"Soil: {row.get('Soil_Type','Unknown')}, "
                f"Rainfall: {row.get('Rainfall_mm','Unknown')}mm, "
                f"Temperature: {row.get('Temperature_Celsius','Unknown')}°C, "
                f"Fertilizer: {row.get('Fertilizer_Used','Unknown')}, "
                f"Irrigation: {row.get('Irrigation_Used','Unknown')}, "
                f"Weather: {row.get('Weather_Condition','Unknown')}, "
                f"Days to Harvest: {row.get('Days_to_Harvest','Unknown')}, "
                f"Yield: {row.get('Yield_tons_per_hectare','Unknown')} tons/ha"
            )
            session.add(CropData(crop=str(crop), location=str(location), data=encrypt(data)))
            inserted += 1
        session.commit()
        print(f"Loaded {inserted} records from {csv_path} into DB.")
    except FileNotFoundError:
        session.rollback()
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_all_decrypted_docs():
    init_db()
    session = SessionLocal()
    try:
        rows = session.query(CropData).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "crop": r.crop,
                "location": r.location,
                "data": decrypt(r.data) if r.data else ""
            })
        return out
    finally:
        session.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crop_yield_data_sampled.csv")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()
    load_sample_data(csv_path=args.csv, clear_table=not args.no_clear)
