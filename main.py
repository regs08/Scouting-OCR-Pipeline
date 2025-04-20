#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.application_manager import ApplicationManager

def main():
    # Configuration
    input_dir = "/Users/cole/PycharmProjects/Scouting-OCR-Pipeline/input"
    vineyard = "arget_singer"
    
    # Create application manager which internally sets up the setup and session managers
    app_manager = ApplicationManager(
        input_dir=input_dir,
        expected_vineyard=vineyard,
        verbose=True
    )
    
    # Run the complete pipeline
    print(f"Starting application for session {app_manager.session_id}...")
    result = app_manager.run()
    
    print(f"Processing complete. Checkpoint status: {result.get('checkpoint_status', {})}")
    return result

if __name__ == "__main__":
    main() 