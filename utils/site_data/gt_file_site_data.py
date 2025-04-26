from .site_data_base import SiteDataBase, FileAnalysisResult
from typing import List, Optional
from datetime import datetime
import re
from pathlib import Path

class GTFileSiteData(SiteDataBase):
    """
    Site data configuration for ground truth (GT) files.
    Expects filenames like: SITE_gt_YYYYMMDD_location.ext
    Can be constructed from an existing SiteDataBase instance.
    """
    def __init__(self, site_data: Optional[SiteDataBase] = None, site_code: Optional[str] = None, collection_date: Optional[str] = None):
        if site_data is not None:
            # Copy relevant attributes from the provided site_data
            data_cols = getattr(site_data, 'data_cols', [])
            index_cols = getattr(site_data, 'index_cols', [])
            site_name = getattr(site_data, 'site_name', 'gt_site') + '_gt'
            site_code = getattr(site_data, 'site_code', site_code)
            location_pattern = getattr(site_data, 'location_pattern', r'.*')
            supported_extensions = getattr(site_data, 'supported_extensions', ['.csv'])
            collection_date = getattr(site_data, 'collection_date', collection_date)
        else:
            data_cols: List[str] = []
            index_cols: List[str] = []
            site_name = f"{site_code}_gt"
            location_pattern = r'.*'
            supported_extensions = ['.csv']
            if collection_date is None:
                collection_date = datetime.now().strftime("%Y%m%d")
        if collection_date is None:
            collection_date = datetime.now().strftime("%Y%m%d")
        # Custom pattern: SITE_gt_YYYYMMDD_location
        file_pattern = rf'^({site_code})_(gt)_(\d{{8}})_(.+?)$'
        super().__init__(
            data_cols=data_cols,
            index_cols=index_cols,
            site_name=site_name,
            site_code=site_code,
            location_pattern=location_pattern,
            supported_extensions=supported_extensions,
            collection_date=collection_date,
            file_pattern=file_pattern
        )

    def get_data_column_indices(self) -> List[int]:
        return []

    def get_index_column_indices(self) -> List[int]:
        return []

    def analyze_filename(self, filename: str) -> FileAnalysisResult:
        base_name = Path(filename).stem
        pattern = self.get_filename_pattern()
        match = re.match(pattern, base_name)
        if not match:
            return FileAnalysisResult(
                is_valid=False,
                error_message=f"Invalid filename format. Expected pattern matching {pattern}"
            )
        groups = match.groups()
        site_code = groups[0]
        # groups[1] is 'gt', groups[2] is the date, groups[3] is location
        date_str = groups[2]
        location_part = groups[3] if len(groups) > 3 else ""
        # Extract row and panel from location part
        row, panel, location_id = self.extract_location_info(location_part)
        if (row is None or panel is None) and location_part:
            return FileAnalysisResult(
                is_valid=False,
                site_code=site_code,
                date=date_str,
                error_message=f"Could not extract row/panel from location: {location_part}"
            )
        return FileAnalysisResult(
            is_valid=True,
            site_code=site_code,
            date=date_str,
            location_id=location_id,
            row=row,
            panel=panel
        ) 