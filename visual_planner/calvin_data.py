import ast
from cgitb import text
import functools
import io
import json
import logging
import math
import os
import copy
import random
import sys
import tarfile
from dataclasses import dataclass
from multiprocessing import Value
import zipfile


import torch
import torchvision

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvideotransforms import video_transforms, volume_transforms

from torchvision import transforms as transforms_
import numpy as np
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset


import cv2

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5


import logging
import os
from pathlib import Path
import re
from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch

logger = logging.getLogger(__name__)


def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}


def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}


def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}


def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}


def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang


def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }


def load_dataset_statistics(train_dataset_dir, val_dataset_dir, transforms):
    """
    Tries to load statistics.yaml in every dataset folder in order to update the transforms hardcoded in the
    hydra config file. If no statistics.yaml exists, nothing is changed

    Args:
        train_dataset_dir: path of the training folder
        val_dataset_dir: path of the validation folder
        transforms: transforms loaded from hydra conf

    Returns:
        transforms: potentially updated transforms
    """
    paths = {"train": train_dataset_dir, "val": val_dataset_dir}
    for dataset_type in ["train", "val"]:
        try:
            statistics = OmegaConf.load(Path(paths[dataset_type]) / "statistics.yaml")
            # Hack for maintaining two repositories with transforms
            statistics = OmegaConf.create(OmegaConf.to_yaml(statistics).replace("calvin_models.", ""))
            # this ugly piece of code only exists because OmegaConf actually can't merge ListConfigs.
            # we do not want to override everything, but just the transforms that are specified in both
            # see https://stackoverflow.com/questions/61315623/omegaconf-can-i-influence-how-lists-are-merged
            for modality in transforms[dataset_type]:
                if modality in statistics:
                    conf_transforms = transforms[dataset_type][modality]
                    dataset_transforms = statistics[modality]
                    for dataset_trans in dataset_transforms:
                        exists = False
                        for i, conf_trans in enumerate(conf_transforms):
                            if dataset_trans["_target_"] == conf_trans["_target_"]:
                                exists = True
                                transforms[dataset_type][modality][i] = dataset_trans
                                break
                        if not exists:
                            transforms[dataset_type][modality] = ListConfig([*conf_transforms, dataset_trans])
        except FileNotFoundError:
            logger.warning("Could not load statistics.yaml")
    return transforms


def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


import logging
from pathlib import Path
from typing import Dict, Tuple, Union


import numpy as np
from omegaconf import DictConfig
# import pyhash
import torch
from torch.utils.data import Dataset


# hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static"],
        "depth_obs": ["depth_static"],
        "state_obs": ["robot_obs"],
        "actions": ["actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)




from typing import Any, Dict, List, Tuple, Callable
from itertools import chain
# from calvin_agent.datasets.utils.episode_utils import lookup_naming_pattern
import pickle
import torch.nn as nn
import torch.nn.functional as F



class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        sampling_step = 1,
        image_size = 256,
        with_depth = False,
    ):
        self.observation_space = obs_space
        print(self.observation_space)
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.min_window_size = window_size
        self.max_window_size = window_size
        self.sampling_step = sampling_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
       

        assert (
            "validation" in self.abs_datasets_dir.as_posix()
            or "training" in self.abs_datasets_dir.as_posix()
        )
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        print(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

        self.transform = video_transforms.Compose([
                # video_transforms.CenterCrop((256, 256)),
                video_transforms.Resize(image_size),
                volume_transforms.ClipToTensor()
            ])
        self.transform_depth = video_transforms.Compose([
                # video_transforms.CenterCrop((256, 256)),
                transforms_.Resize(image_size, antialias=True),
            ])
        self.normalize = transforms_.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        self.with_depth = with_depth

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)
        
        # RGB images
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        image_tensors = self.normalize(torch.from_numpy(np.array(self.transform(sequence["rgb_obs"]["rgb_static"]))).transpose(0, 1))


        # Depth images
        np_depth = copy.deepcopy(sequence["depth_obs"]['depth_static'].numpy())
        per_vid_depth_max = np_depth.max() # 6
        per_vid_depth_min = np_depth.min() # 3
        np_depth = ( np_depth - per_vid_depth_min ) / ( per_vid_depth_max - per_vid_depth_min )
        sequence["depth_obs"]['depth_static'] = torch.from_numpy(np_depth[:, None, :, :])
        depth_tensors = self.transform_depth(sequence["depth_obs"]['depth_static']) * 2 - 1

        # Tack instructions
        stacked_language = sequence["lang"]

        state_tensor = sequence['state_info']['robot_obs']
        state_tensor = torch.cat([state_tensor[..., :6], (state_tensor[..., [-1]] + 1) // 2], dim=-1)

        if self.with_depth:
            x_cond = torch.cat([image_tensors[[0]], depth_tensors[[0]]], dim=1)
            x = torch.cat([image_tensors[1:], depth_tensors[1:]], dim=1)
        else:
            x_cond = image_tensors[:1]
            x = image_tensors[1:]

        return x, x_cond, stacked_language, state_tensor

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        # print(seq_acts['actions'].shape)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info



def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format

        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """


        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + self.window_size #* self.sampling_step + self.sampling_step
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")

        
        try:
            episodes = [
                self.load_file(self._get_episode_name(file_idx))
                for file_idx in range(start_idx, end_idx, self.sampling_step)
            ]
        except:
            start_idx += 10
            end_idx += 10
            episodes = [
                self.load_file(self._get_episode_name(file_idx))
                for file_idx in range(start_idx, end_idx, self.sampling_step)
            ]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []

        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            # if self.partial_data:
            #     if (start_idx, end_idx) not in partial_st_ed_list:
            #         continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)












