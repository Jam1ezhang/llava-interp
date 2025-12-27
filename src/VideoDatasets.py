import json
import os
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class VideoQADataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        video_root: str,
        max_samples: Optional[int] = None,
    ):
        self.annotations_path = annotations_path
        self.video_root = video_root
        with open(annotations_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        video_name = item["video_name"]
        video_path = os.path.join(self.video_root, video_name)
        return {
            "video_path": video_path,
            "question": item["question"],
            "candidates": item.get("candidates", []),
            "answer": item.get("answer"),
            "question_id": item.get("question_id"),
            "task_name": item.get("task_name"),
            "start": item.get("start"),
            "end": item.get("end"),
            "frames_with_answer_object": item.get("frames_with_answer_object"),
        }
