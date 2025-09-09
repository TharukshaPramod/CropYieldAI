from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

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
        session.add(CropData(crop='wheat', location='California', data='Yield: 5 tons/ha, soil pH 6.5, rain 400mm'))
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()