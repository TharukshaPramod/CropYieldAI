from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
ENGINE = create_engine(os.getenv('DATABASE_URL', 'sqlite:///crop.db'))
Base = declarative_base()

class CropData(Base):
    __tablename__ = 'crop_data'
    id = Column(Integer, primary_key=True)
    crop = Column(String)
    location = Column(String)
    data = Column(Text)

Base.metadata.create_all(ENGINE)
Session = sessionmaker(bind=ENGINE)

def load_sample_data():
    session = Session()
    try:
        # Clear existing data to avoid duplicates
        session.query(CropData).delete()
        session.commit()

        # Load Kaggle dataset
        df = pd.read_csv('data/crop_yield_data.csv')
        for index, row in df.iterrows():
            crop = row['Crop']
            location = f"{row['State']} {row['District']}"
            data = f"Year: {row['Year']}, Season: {row['Season']}, Area: {row['Area']}, Production: {row['Production']}"
            session.add(CropData(crop=crop, location=location, data=data))
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Load data on import (optional, or call manually)
if __name__ == "__main__":
    load_sample_data()