import os
import re

class FileMatcher:
    def __init__(self, working_folder, gt_directory, pred_directory, regex=r'R\d+P\d+'):
        """
        working_folder: which dataset we're working with e.g argetsinger, 
        pred_directory: full path to predictions folder
        regex: regex to match file names with any number of digits for row and panel numbers (e.g., R1P1, R10P17, R100P200)
        """
        self.working_folder = working_folder
        self.gt_directory = os.path.join(gt_directory, working_folder)
        self.pred_directory = os.path.join(pred_directory, working_folder)
        self.regex = r'R[1-9]\d*P[1-9]\d*'  # Matches R followed by any non-zero number and P followed by any non-zero number
        self.gt_files = []
        self.pred_files = []
        self.matches = {}

    def find_files(self):
        # Find ground truth files
        self.gt_files = [f for f in os.listdir(self.gt_directory) if re.search(self.regex, f)]
        # Find predicted files
        self.pred_files = [f for f in os.listdir(self.pred_directory) if re.search(self.regex, f)]

    def match_files(self):
        self.matches = {}
        for gt_file in self.gt_files:
            match = re.search(self.regex, gt_file)
            if match:
                unique_id = match.group()
                full_gt_paths = os.path.join(self.gt_directory, gt_file)
                self.matches[unique_id] = {
                    'gt_path': full_gt_paths,
                    'pred_path': None
                }
        
        for pred_file in self.pred_files:
            match = re.search(self.regex, pred_file)
            if match:
                unique_id = match.group()
                if unique_id in self.matches:
                    full_pred_path = os.path.join(self.pred_directory, pred_file)
                    self.matches[unique_id]['pred_path'] = full_pred_path

        return self.matches

    def unmatched_files(self):
        # Identify unmatched ground truth files
        unmatched_gt = [f for f in self.gt_files if not any(re.search(self.regex, f) and match['gt_path'] == os.path.join(self.gt_directory, f) for match in self.matches.values())]
        
        # Identify unmatched predicted files
        unmatched_pred = [f for f in self.pred_files if not any(re.search(self.regex, f) and match['pred_path'] == os.path.join(self.pred_directory, f) for match in self.matches.values())]
        
        # Identify files in the directories that do not match the regex
        all_gt_files = os.listdir(self.gt_directory)
        all_pred_files = os.listdir(self.pred_directory)

        non_matching_gt = [f for f in all_gt_files if not re.search(self.regex, f)]
        non_matching_pred = [f for f in all_pred_files if not re.search(self.regex, f)]

        return {
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred,
            'non_matching_gt': non_matching_gt,
            'non_matching_pred': non_matching_pred
        }

# Example usage
working_folder = 'argetsinger'
gt_directory = '/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ground_truth/'
pred_directory = '/Users/nr466/Python Projects/Scouting_OCR_Pipeline/data/ocr_predictions/'  # Update this path accordingly

file_matcher = FileMatcher(working_folder, gt_directory, pred_directory)
file_matcher.find_files()
matched_files = file_matcher.match_files()
unmatched_files = file_matcher.unmatched_files()

print("Matched Files:", matched_files)
print("Unmatched Files:", unmatched_files)