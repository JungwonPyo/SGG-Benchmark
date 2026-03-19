"""
Base class for scene-graph relation datasets.

Subclasses (VGDataset, RelationDataset) must set these attributes in __init__:
    * ``self.filenames``         – list of absolute image paths
    * ``self.img_info``          – list of dicts (width, height, image_id)
    * ``self.ind_to_classes``    – object-class names, 0-indexed (background first)
    * ``self.ind_to_predicates`` – predicate names, 0-indexed (background first)

Subclasses must also implement ``_compute_raw_statistics()``.
"""
import json
import os

import numpy as np
import torch
from PIL import Image


class BaseRelationDataset(torch.utils.data.Dataset):
    """Shared interface and logic for all SGG relation datasets."""

    # ------------------------------------------------------------------
    # Length / indexing
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.filenames)

    def get_img_info(self, index):
        return self.img_info[index]

    # ------------------------------------------------------------------
    # Custom image loading (demo / inference on arbitrary images)
    # ------------------------------------------------------------------

    def get_custom_imgs(self, path):
        """Populate ``self.custom_files`` and ``self.img_info`` from a
        directory of images or a JSON file listing image paths."""
        self.custom_files = []
        self.img_info = []
        if not os.path.exists(path):
            return
        if os.path.isdir(path):
            img_exts = ('.jpg', '.jpeg', '.png')
            files = os.listdir(path)
            if not any(f.endswith(img_exts) for f in files):
                return
            for file_name in files:
                if not file_name.endswith(img_exts):
                    continue
                full = os.path.join(path, file_name)
                self.custom_files.append(full)
                img = Image.open(full).convert("RGB")
                self.img_info.append({
                    'width':    int(img.width),
                    'height':   int(img.height),
                    'image_id': str(file_name.split('.')[0]),
                })
        elif os.path.isfile(path):
            file_list = json.load(open(path))
            for file in file_list:
                self.custom_files.append(file)
                img = Image.open(file).convert("RGB")
                self.img_info.append({
                    'width':    int(img.width),
                    'height':   int(img.height),
                    'image_id': str(file.split('/')[-1].split('.')[0]),
                })

    # ------------------------------------------------------------------
    # Statistics (template method — delegates to _compute_raw_statistics)
    # ------------------------------------------------------------------

    def get_statistics(self):
        """Return the statistics dict used to build FrequencyBias and related
        modules.  The ``pred_dist`` field is computed here from the raw
        foreground/background matrices returned by ``_compute_raw_statistics``.
        """
        (fg_matrix, bg_matrix,
         predicate_new_order, predicate_new_order_count,
         pred_freq, triplet_freq, pred_weight) = self._compute_raw_statistics()

        eps = 1e-3
        bg_matrix += 1
        fg_sum = fg_matrix.sum(2)[:, :, None]

        ratio = np.divide(
            fg_matrix, fg_sum,
            out=np.zeros_like(fg_matrix, dtype=float),
            where=fg_sum > 0,
        )
        pred_dist = np.log(np.where(fg_sum > 0, ratio, 1e-10) + eps)

        return {
            'fg_matrix':               torch.from_numpy(fg_matrix),
            'pred_dist':               torch.from_numpy(pred_dist).float(),
            'obj_classes':             self.ind_to_classes,
            'rel_classes':             self.ind_to_predicates,
            'predicate_new_order':     predicate_new_order,
            'predicate_new_order_count': predicate_new_order_count,
            'pred_freq':               pred_freq,
            'triplet_freq':            triplet_freq,
            'pred_weight':             pred_weight,
        }

    def _compute_raw_statistics(self):
        """Return the 7-tuple expected by ``get_statistics``:

        ``(fg_matrix, bg_matrix, predicate_new_order,
           predicate_new_order_count, pred_freq, triplet_freq, pred_weight)``

        Must be implemented by every concrete subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _compute_raw_statistics()"
        )
