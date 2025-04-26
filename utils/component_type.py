from enum import Enum, auto

class ComponentType(Enum):
    """Enum defining different types of pipeline components."""
    
    # High-level components
    MANAGER = auto()       # High-level pipeline managers (SetupManager, SessionManager, etc.)
    
    # Processing components
    PROCESSOR = auto()     # General data processors
    SETUP = auto()        # Setup-specific processors
    VALIDATOR = auto()    # Data validation processors
    CLEANER = auto()      # Data cleaning processors
    
    # Utility components
    UTILITY = auto()      # Utility components (PathManager, DirectoryManager, etc.)
    LOGGER = auto()       # Logging-specific components 