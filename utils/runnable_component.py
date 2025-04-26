from abc import ABC, abstractmethod
from typing import Any, Dict


class RunnableComponent(ABC):
    """Abstract interface for components that can be executed in a pipeline."""
    
    def __init__(self, **kwargs):
        """Initialize the component, accepting any kwargs."""
        super().__init__()
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the component's main function.
        
        Args:
            input_data: Dictionary containing input data for the component
            
        Returns:
            Dictionary containing output data from the component
        """
        pass 