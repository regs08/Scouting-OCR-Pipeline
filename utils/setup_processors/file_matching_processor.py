from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
import re
import shutil
from .base_setup_processor import BaseSetupProcessor

class FileMatchingProcessor(BaseSetupProcessor):
    """Processor for ensuring matching files between raw data and ground truth."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rp_pattern = r'R\d+P\d+_R\d+P\d+'  # Pattern to match R#P#_R#P# sequences
        
    def _extract_sequence(self, filename: str) -> Optional[str]:
        """
        Extract the R#P#_R#P# sequence from a filename.
        
        Args:
            filename: The filename to extract from
            
        Returns:
            The sequence if found, None otherwise
        """
        match = re.search(self.rp_pattern, filename)
        return match.group(0) if match else None
        
    def _find_matching_files(self, 
                           raw_files: List[Path], 
                           gt_files: List[Path]) -> Tuple[Dict[str, List[Path]], Set[str]]:
        """
        Find matching files between raw data and ground truth.
        
        Args:
            raw_files: List of raw data files
            gt_files: List of ground truth files
            
        Returns:
            Tuple of (matching_files, unmatched_sequences)
            matching_files: Dictionary mapping sequences to lists of matching files
            unmatched_sequences: Set of sequences that don't have matches
        """
        # Extract sequences from all files
        raw_sequences = {self._extract_sequence(f.stem): f for f in raw_files if self._extract_sequence(f.stem)}
        gt_sequences = {self._extract_sequence(f.stem): f for f in gt_files if self._extract_sequence(f.stem)}
        
        # Find all unique sequences
        all_sequences = set(raw_sequences.keys()) | set(gt_sequences.keys())
        
        # Find matching and unmatched files
        matching_files = {}
        unmatched_sequences = set()
        
        for sequence in all_sequences:
            raw_file = raw_sequences.get(sequence)
            gt_file = gt_sequences.get(sequence)
            
            if raw_file and gt_file:
                matching_files[sequence] = [raw_file, gt_file]
            else:
                unmatched_sequences.add(sequence)
                # Move unmatched files to flagged directory
                flagged_dir = self.path_manager.get_flagged_dir(self.session_id)
                flagged_dir.mkdir(parents=True, exist_ok=True)
                
                if raw_file and not gt_file:
                    self.log_warning("_find_matching_files", 
                                   f"No ground truth match for raw file: {raw_file.name}")
                    shutil.copy2(str(raw_file), str(flagged_dir / raw_file.name))
                elif gt_file and not raw_file:
                    self.log_warning("_find_matching_files", 
                                   f"No raw data match for ground truth file: {gt_file.name}")
                    shutil.copy2(str(gt_file), str(flagged_dir / gt_file.name))
                    
        return matching_files, unmatched_sequences
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure matching files between raw data and ground truth.
        
        Args:
            input_data: Dictionary containing setup data
            
        Returns:
            Updated dictionary with matching file information
        """
        try:
            # Get directories
            session_dir = input_data['session_dir']
            raw_dir = session_dir / "raw" / "original" / "_by_id"
            gt_dir = session_dir / "ground_truth"
            
            # Get all files
            raw_files = list(raw_dir.glob("*.png"))
            gt_files = list(gt_dir.glob("*.csv"))
            
            self.log_info("process", f"Found {len(raw_files)} raw files and {len(gt_files)} ground truth files")
            
            # Find matching files
            matching_files, unmatched_sequences = self._find_matching_files(raw_files, gt_files)
            
            # Log results
            self.log_info("process", f"Found {len(matching_files)} matching file pairs")
            if unmatched_sequences:
                self.log_warning("process", 
                               f"Found {len(unmatched_sequences)} unmatched sequences. Files moved to flagged directory.")
                
            # Update input data
            input_data['matching_files'] = matching_files
            input_data['unmatched_sequences'] = list(unmatched_sequences)
            
            return input_data
            
        except Exception as e:
            self.log_error("process", f"Error matching files: {str(e)}")
            raise 