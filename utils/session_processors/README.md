# Session Processors

This directory contains the session processors that are used during a processing session. Each processor is responsible for a specific stage of the data processing pipeline.

## Structure

- `base_session_processor.py` - Base class for all session processors, providing common functionality
- `ocr_processor.py` - Processor for OCR (Optical Character Recognition) on image files
- `dimension_comparison_processor.py` - Processor for comparing dimensions between ground truth and OCR output
- `confusion_matrix_processor.py` - Processor for generating confusion matrices between ground truth and checkpoint data

## Usage

Session processors are used by the SessionManager to process data in a sequential pipeline. Each processor inherits from the BaseSessionProcessor class and implements the required `process` method.

Example of how to use a session processor:

```python
from utils.session_processors.ocr_processor import OCRProcessor
from utils.path_manager import PathManager

# Initialize path manager
path_manager = PathManager(base_dir="data")

# Initialize OCR processor
ocr_processor = OCRProcessor(
    path_manager=path_manager,
    session_id="test_session",
    verbose=True
)

# Process data
result = ocr_processor.run({
    'session_dir': 'data/test_session'
})
```

### Using the Confusion Matrix Processor

The ConfusionMatrixSessionProcessor compares data from a previous checkpoint against ground truth files to generate confusion matrices and analysis metrics:

```python
from utils.session_processors.confusion_matrix_processor import ConfusionMatrixSessionProcessor
from utils.path_manager import PathManager

# Initialize path manager
path_manager = PathManager(base_dir="data")

# Initialize confusion matrix processor
confusion_processor = ConfusionMatrixSessionProcessor(
    path_manager=path_manager,
    session_id="test_session",
    checkpoint_name="ckpt2_dimension_comparison",  # Specify which checkpoint data to use
    verbose=True
)

# Process data
result = confusion_processor.run({
    'session_dir': 'data/test_session'
})

# Access metrics
avg_f1_score = result['avg_f1_score']
print(f"Average F1 Score: {avg_f1_score:.2f}")
```

## Adding New Processors

To add a new session processor:

1. Create a new Python file in this directory
2. Inherit from `BaseSessionProcessor`
3. Implement the `process` method to perform the specific processing task
4. Add the new processor to the `__init__.py` file to make it available for import 