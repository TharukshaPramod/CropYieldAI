# 🌾 Crop Yield Prediction AI

A comprehensive machine learning system for predicting agricultural crop yields using advanced AI and geospatial data analysis.

## 🚀 Features

- **Predictive Analytics**: Forecast crop yields based on historical data and environmental factors
- **Multi-Model Approach**: Ensemble of machine learning models for accurate predictions
- **Geospatial Integration**: Incorporates satellite imagery and location-based features
- **Real-time Monitoring**: Live data processing and visualization
- **RESTful API**: Fully documented API for integration with other systems
- **Interactive Dashboard**: User-friendly Streamlit interface for data exploration

## 🛠 Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Geospatial**: GeoPandas, Rasterio, Sentinel Hub
- **Data Processing**: Pandas, NumPy, Dask
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Validation**: Pydantic
- **Testing**: Pytest, Hypothesis

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Poetry (dependency management)
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd crop-yield-prediction-ai
```

2. **Install dependencies using Poetry**
```bash
poetry install
```

3. **Download required NLP model**
```bash
poetry run python -m spacy download en_core_web_sm
```

4. **Environment Configuration**
   
Create a `.env` file in the root directory:
```env
# API Keys and External Services
SENTINEL_HUB_API_KEY=your_sentinel_hub_key
WEATHER_API_KEY=your_weather_api_key
DATABASE_URL=your_database_connection_string

# Model Configuration
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

## 🎯 Usage

### Running the Web Interface
```bash
poetry run streamlit run app.py
```
Access the dashboard at: `http://localhost:8501`

### Starting the API Server
```bash
poetry run uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```
API documentation available at: `http://localhost:8000/docs`

### Production Deployment
```bash
# Start with production settings
poetry run uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🧪 Testing

Run the complete test suite:
```bash
poetry run pytest
```

Specific test categories:
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# With coverage report
poetry run pytest --cov=src --cov-report=html

# Performance testing
poetry run pytest tests/performance/ -v
```

## 📁 Project Structure

```
crop-yield-prediction-ai/
├── src/                    # Source code
│   ├── models/            # ML model implementations
│   ├── data/              # Data processing modules
│   ├── api/               # API routes and handlers
│   └── utils/             # Utility functions
├── tests/                 # Test suites
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   ├── processed/         # Processed data
│   └── models/           # Trained model files
├── notebooks/            # Jupyter notebooks for exploration
├── docs/                 # Documentation
└── config/               # Configuration files
```

## 🔧 Configuration

### Model Parameters
Edit `config/model_config.yaml` to adjust:
- Feature engineering parameters
- Model hyperparameters
- Training configurations
- Validation settings

### API Settings
Modify `config/api_config.yaml` for:
- Rate limiting
- CORS settings
- Authentication
- Cache configurations

## 📊 Data Sources

The system integrates multiple data sources:
- **Satellite Imagery**: Sentinel-2, Landsat 8
- **Weather Data**: Historical and forecast data
- **Soil Data**: Composition and quality metrics
- **Agricultural Records**: Historical yield data
- **Economic Factors**: Market prices and trends

## 🔍 API Endpoints

### Core Endpoints
- `POST /api/v1/predict` - Generate yield predictions
- `GET /api/v1/models` - List available models
- `POST /api/v1/train` - Retrain models with new data
- `GET /api/v1/health` - System health check

### Data Endpoints
- `GET /api/v1/data/sources` - Available data sources
- `POST /api/v1/data/upload` - Upload new datasets
- `GET /api/v1/data/statistics` - Dataset statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Pre-commit hooks
poetry run pre-commit install

# Run linter
poetry run black src/ tests/
poetry run isort src/ tests/
```

## 📈 Performance Monitoring

The system includes:
- Real-time performance metrics
- Model drift detection
- Automated retraining pipelines
- Comprehensive logging

## 🚨 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
poetry lock --no-update
poetry install
```

**API Connection Issues**
- Verify `.env` file configuration
- Check API key validity
- Ensure required ports are available

**Model Loading Errors**
- Clear cache: `rm -rf ./models/cache`
- Retrain models using the training endpoint

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Satellite data providers: ESA Sentinel Hub, NASA Landsat
- Weather data sources: OpenWeatherMap, NOAA
- Agricultural research institutions and open data initiatives

---

**Maintainers**: [Your Team/Organization]  
**Support**: [Support Contact/Channel]  
**Version**: 1.0.0