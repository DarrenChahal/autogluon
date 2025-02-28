import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Union

from ..constants import COLUMN, IMAGE, POINT_PROMPT, POINT_PROMPT_VALID_NUM
from .collator import ListCollator, PadCollator

logger = logging.getLogger(__name__)

class PointPromptProcessor:
    """
    Process point prompts for the Segment Anything Model (SAM).
    """

    def __init__(
        self,
        model,
        max_points_per_image: int = 10,
        missing_value_strategy: str = "empty",
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        max_points_per_image
            Maximum number of points to process per image.
        missing_value_strategy
            How to handle missing point prompts:
            - "empty": Use empty point prompts (model will use only image features)
            - "random": Generate random points (for data augmentation)
        """
        self.prefix = model.prefix
        self.max_points_per_image = max_points_per_image
        self.missing_value_strategy = missing_value_strategy
        self.image_size = model.image_size

    @property
    def point_prompt_key(self):
        return f"{self.prefix}_{POINT_PROMPT}"

    @property
    def point_prompt_valid_num_key(self):
        return f"{self.prefix}_{POINT_PROMPT_VALID_NUM}"

    @property
    def point_prompt_column_prefix(self):
        return f"{self.point_prompt_key}_{COLUMN}"

    def collate_fn(self, point_column_names: Optional[List] = None) -> Dict:
        """
        Collate point prompts into a batch.
        """
        fn = {
            self.point_prompt_key: PadCollator(pad_val=0),
            self.point_prompt_valid_num_key: PadCollator(pad_val=0),
        }
        return fn

    def process_one_sample(
        self,
        point_features: Dict[str, List[str]],
        feature_modalities: Dict[str, List[str]],
        is_training: bool,
    ) -> Dict:
        """
        Process point prompts for one sample.

        Parameters
        ----------
        point_features
            Dictionary containing point prompt data.
        feature_modalities
            What modality each column belongs to.
        is_training
            Whether in training mode.

        Returns
        -------
        A dictionary containing processed point prompts.
        """
        ret = {}
        
        # Find the point prompt column
        point_column = None
        for column_name, column_modality in feature_modalities.items():
            if column_modality == POINT_PROMPT:
                point_column = column_name
                break
        
        # If no point column found or it's empty
        if point_column is None or point_column not in point_features or not point_features[point_column]:
            if self.missing_value_strategy == "random" and is_training:
                # Generate random points for training
                num_points = np.random.randint(1, self.max_points_per_image + 1)
                points = torch.rand(1, num_points, 2) * self.image_size
                labels = torch.ones(1, num_points)
                
                ret[self.point_prompt_key] = torch.cat([points, labels.unsqueeze(-1)], dim=-1)
                ret[self.point_prompt_valid_num_key] = torch.tensor([num_points])
            else:
                # Empty point prompts
                ret[self.point_prompt_key] = torch.zeros(1, 1, 3)  # [x, y, label]
                ret[self.point_prompt_valid_num_key] = torch.tensor([0])
            return ret
        
        # Process actual point prompts
        point_data = point_features[point_column]
        if isinstance(point_data, str):
            # Parse string format: "x1,y1,label1;x2,y2,label2;..."
            points_list = []
            labels_list = []
            
            point_pairs = point_data.split(';')
            for pair in point_pairs[:self.max_points_per_image]:
                if not pair.strip():
                    continue
                    
                coords = pair.split(',')
                if len(coords) >= 3:
                    x, y, label = float(coords[0]), float(coords[1]), int(coords[2])
                    # Normalize coordinates to [0, image_size]
                    x = min(max(0, x), 1) * self.image_size
                    y = min(max(0, y), 1) * self.image_size
                    points_list.append([x, y])
                    labels_list.append(label)
            
            if not points_list:
                # No valid points found
                ret[self.point_prompt_key] = torch.zeros(1, 1, 3)
                ret[self.point_prompt_valid_num_key] = torch.tensor([0])
                return ret
                
            points = torch.tensor(points_list).unsqueeze(0)  # [1, num_points, 2]
            labels = torch.tensor(labels_list).unsqueeze(0)  # [1, num_points]
            
            # Combine points and labels
            ret[self.point_prompt_key] = torch.cat([points, labels.unsqueeze(-1)], dim=-1)  # [1, num_points, 3]
            ret[self.point_prompt_valid_num_key] = torch.tensor([len(points_list)])
            
        elif isinstance(point_data, (list, np.ndarray)):
            # Handle array format: [[x1, y1, label1], [x2, y2, label2], ...]
            point_data = np.array(point_data)
            if point_data.size == 0:
                ret[self.point_prompt_key] = torch.zeros(1, 1, 3)
                ret[self.point_prompt_valid_num_key] = torch.tensor([0])
                return ret
                
            # Ensure correct shape
            if len(point_data.shape) == 1 and point_data.shape[0] == 3:
                # Single point
                point_data = point_data.reshape(1, 3)
            
            # Limit number of points
            point_data = point_data[:self.max_points_per_image]
            
            # Normalize coordinates to [0, image_size]
            point_data[:, 0] = np.clip(point_data[:, 0], 0, 1) * self.image_size
            point_data[:, 1] = np.clip(point_data[:, 1], 0, 1) * self.image_size
            
            points_tensor = torch.tensor(point_data).float().unsqueeze(0)  # [1, num_points, 3]
            ret[self.point_prompt_key] = points_tensor
            ret[self.point_prompt_valid_num_key] = torch.tensor([len(point_data)])
            
        return ret

    def __call__(
        self,
        point_features: Dict[str, List[str]],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Process point prompts.

        Parameters
        ----------
        point_features
            Dictionary containing point prompt data.
        feature_modalities
            What modality each column belongs to.
        is_training
            Whether in training mode.

        Returns
        -------
        A dictionary containing processed point prompts.
        """
        return self.process_one_sample(point_features, feature_modalities, is_training) 