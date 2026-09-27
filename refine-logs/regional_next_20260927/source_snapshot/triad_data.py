"""Data adapter for the geometry-aligned Hunan 8-to-16 backbone contract."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


GEOMETRY_CONTRACT = "history_8_then_target_16_v2"


def validate_frame_paths(paths, bjt_start=None):
    """Fail closed unless a row is 24 chronological 15-minute frames."""
    if len(paths) != 24:
        raise ValueError(f"expected 24 chronological AGRI frames, got {len(paths)}")
    times = [datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") for rel in paths]
    step = timedelta(minutes=15)
    if any(b - a != step for a, b in zip(times, times[1:])):
        raise ValueError("manifest AGRI frame paths are not strictly 15-minute chronological")
    if bjt_start is not None:
        first_bjt = times[0] + timedelta(hours=8)
        expected = datetime.strptime(bjt_start, "%Y-%m-%d %H:%M:%S")
        if first_bjt != expected:
            raise ValueError(f"manifest BJT_start {expected} does not match first frame {first_bjt}")
    return times


class ForecastHunanDataset(Dataset):
    """Return 8 history images, 24 ordered geometry maps, and 16 targets.

    Each manifest row contains 24 chronological AGRI frames: indices 0..7 are
    observed history and indices 8..23 are the forecast targets.  Geometry is
    kept for all 24 frames so each forecast lead can use its own target-time
    solar geometry rather than a truncated history/target mixture.
    """

    def __init__(self, source, root, manifest, norm, split, include_index=False):
        sys.path.insert(0, str(source))
        from hunan_data import HunanDataset, load_frame

        self.base = HunanDataset(root, manifest, norm, split)
        self.load_frame = load_frame
        self.root = root
        self.include_index = include_index
        self.mean, self.std = self.base.mean, self.base.std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base.rows[index]
        images, masks, geometry = [], [], []
        paths = row["data_relpaths"].split("|")
        validate_frame_paths(paths, row.get("BJT_start"))
        for rel in paths:
            agri, valid, geom = self.load_frame(self.root, rel)
            images.append(np.where(valid, (agri - self.mean) / self.std, 0).astype(np.float32))
            masks.append(valid)
            geometry.append(geom)
        values = (np.stack(images[:8]), np.stack(geometry),
                  np.stack(images[8:]), np.stack(masks[8:]))
        if self.include_index:
            return (index, *values)
        return values
