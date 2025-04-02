# Scouting OCR Pipeline

This project implements an OCR (Optical Character Recognition) pipeline using Azure services to process handwritten data in table form. The system provides automated data extraction, error analysis, and data correction capabilities.

## Features

- **OCR Processing**: Utilizes Azure's OCR services to extract text from handwritten tables
- **Error Analysis**: Identifies and analyzes common error patterns in the extracted data
- **Data Correction**: Automatically corrects common mislabeled or misinterpreted data
- **Human Review Flagging**: Flags problematic entries that require human verification
- **Analytics Dashboard**: Provides insights into error patterns and data quality metrics

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Azure credentials:
   - Set up your Azure account and obtain necessary API keys
   - Configure environment variables for Azure service authentication

4. Run the project:
```bash
python main.py
```

## Project Structure

- `main.py`: Main entry point of the application
- `requirements.txt`: Project dependencies
- `venv/`: Virtual environment directory
- `config/`: Configuration files for Azure services and OCR settings
- `utils/`: Utility functions for data processing and error handling
- `analytics/`: Analytics and reporting modules

## Dependencies

The project requires the following main dependencies:
- Azure Computer Vision SDK
- Azure Form Recognizer
- Pandas for data manipulation
- NumPy for numerical operations
- Additional dependencies listed in requirements.txt

## Usage

1. Input your handwritten table images
2. The system will process the images through Azure OCR
3. Review the extracted data and any flagged items
4. Access analytics dashboard for error patterns and quality metrics

## Contributing

Feel free to submit issues and enhancement requests! 