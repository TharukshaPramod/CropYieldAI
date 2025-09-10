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

        # Load the new Kaggle dataset
        df = pd.read_csv('data/crop_yield_data_sampled.csv')  # Replace with your CSV filename
        for index, row in df.iterrows():
            crop = row['Crop']
            location = row['Region']  # Use Region as location
            data = f"Soil: {row['Soil_Type']}, Rainfall: {row['Rainfall_mm']}mm, Temperature: {row['Temperature_Celsius']}°C, Fertilizer: {row['Fertilizer_Used']}, Irrigation: {row['Irrigation_Used']}, Weather: {row['Weather_Condition']}, Days to Harvest: {row['Days_to_Harvest']}, Yield: {row['Yield_tons_per_hectare']} tons/ha"
            session.add(CropData(crop=crop, location=location, data=data))
        session.commit()
        print(f"Loaded {len(df)} records from the new dataset.")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Load data on import (optional, or call manually)
if __name__ == "__main__":
    load_sample_data()