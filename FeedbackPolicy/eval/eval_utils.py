import argparse
from collections import Counter, defaultdict, namedtuple, deque
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from moviepy.editor import ImageSequenceClip
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
import torch.nn.functional as F  
import torchvision.transforms as transforms
import torchvision
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env


# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['NCCL_BLOCKING_WAIT'] = '0' 
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
logger = logging.getLogger(__name__)

EP_LEN = 360            # Num of actions per task
NUM_SEQUENCES = 1000    # Num of instruction chains

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

class DebugEnv():
    
    def __init__(self) -> None:
        pass
    
    def get_random_obs(self):
        obs = {}
        obs['rgb_obs'] = {}
        obs['rgb_obs']['rgb_static'] = np.ones((200, 200, 3), dtype=np.uint8)
        obs['rgb_obs']['rgb_gripper'] = np.ones((84, 84, 3), dtype=np.uint8)
        obs['robot_obs'] = np.ones(15, dtype=np.float32)
        return obs
    
    def get_obs(self):
        return self.get_random_obs()
    
    def step(self, action):
        return self.get_random_obs()

    def reset(self, **kwargs):
        return

    def get_info(self):
        return


def make_env_debug(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, policy_model, tokenizer, text_encoder):
        super().__init__()
        self.model = model
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(policy_model.device)
        self.action_hist_queue = []
        self.dt_feat_cache = []

        diffusion_res = 128
        policy_res = 192

        self.transform = transforms.Compose([
                            transforms.Resize((diffusion_res, diffusion_res)),
                            transforms.ToTensor(),
                        ])
        self.transform_depth = transforms.Compose([
                            transforms.Resize((diffusion_res, diffusion_res)),
                            transforms.ToTensor(),
                        ])
        
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.resize_and_norm = transforms.Compose([
                            transforms.Resize((policy_res, policy_res)),
                            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                        ])
        self.preprocess_gripper = transforms.Compose([
                            transforms.Resize((policy_res, policy_res)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                        ])
        self.resize_depth = transforms.Resize((168, 168))
        
        self.cosine_sim_window = []
        self.cached_frames_pred = None
        self.tgt_features = None
        self.future_frame_idx = 0   ### indicating the sub-goal
        self.is_first = True

        self.counter = 0
        self.guidance_weight = 4
        self.subgoal_horizon = 8

        

    
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 32).to('cuda')
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed

    def reset(self, subtask_i):
        """
        This is called
        """
        self.future_frame_idx = 0
        self.is_first = True


    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        # Preprocess image
        image = Image.fromarray(obs["rgb_obs"]['rgb_static']) # ["rgb_gripper"]
        image_gripper = Image.fromarray(obs["rgb_obs"]['rgb_gripper']) 

        # Rescale from [0, 1] to [-1, 1]
        image_x = self.transform(image) * 2 - 1         
        image_gripper = self.preprocess_gripper(image_gripper) 

        # Normalize depth to [0, 1]    
        depth_tensors = self.transform_depth(Image.fromarray(obs["depth_obs"]['depth_static']))
        per_vid_depth_max =  depth_tensors.max()
        per_vid_depth_min =  depth_tensors.min()
        depth_tensors = ( depth_tensors - per_vid_depth_min ) / ( per_vid_depth_max - per_vid_depth_min )     
        depth_tensors = depth_tensors * 2 - 1

        # Encode task description
        text_x = self.encode_batch_text([goal])


        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            depth_tensors = depth_tensors.to(device)
            text_x = text_x.to(device)
            image_rgbd = torch.cat([image_x, depth_tensors], dim=0).unsqueeze(0)


            # Run out of predicted frames (sub-states) or fisrt time roll-out
            if self.future_frame_idx > (self.subgoal_horizon - 1) or self.is_first: 
                pred_frames = self.model.module.ema_model.sample(batch_size=1, x_cond=image_rgbd, task_embed=text_x, frames=self.subgoal_horizon, guidance_weight=self.guidance_weight)

                self.is_first = False
                self.future_frame_idx = 0  
                self.cached_frames_pred = pred_frames

                # Re-Normalization
                input_states = self.cached_frames_pred * 0.5 + 0.5
                input_depth = input_states[0, :, 3:] * ( per_vid_depth_max - per_vid_depth_min ) + per_vid_depth_min
                input_depth = ( input_depth - 3 ) / 3

                self.tgt_features = self.policy_model.module.get_pred_features(  vision_x = self.resize_and_norm(input_states[0, :, :3]).unsqueeze(0), 
                                                                                 vision_depth = self.resize_depth(input_depth).unsqueeze(0))
                
                # Compute distance between generated goals
                cos_distance_all = []
                for i in range(len(self.tgt_features) - 1):
                    cos_distance_tmp = 1 - F.cosine_similarity(F.normalize(self.tgt_features[i]), F.normalize(self.tgt_features[i + 1]))
                    cos_distance_all.append(cos_distance_tmp.item())

                # Replanning
                if max(cos_distance_all) > 1:
                    self.is_first = True


            ### Formulate policy inputs (current state + goal state)
            # Value range: [-1, 1] -> [0, 1]
            input_states = torch.cat([image_x.unsqueeze(0), self.cached_frames_pred[0, [self.future_frame_idx], :3]], dim=0) * 0.5 + 0.5
            input_depth = torch.cat([depth_tensors.unsqueeze(0), self.cached_frames_pred[0, [self.future_frame_idx], 3:]], dim=0) * 0.5 + 0.5

            # min-max normalized relative depth
            input_depth = input_depth * ( per_vid_depth_max - per_vid_depth_min ) + per_vid_depth_min
            input_depth = ( input_depth - 3 ) / 3

            # Run policy model
            stacked_velo_pred, stacked_grip_pred, cos_distance = \
            self.policy_model(  vision_x = self.resize_and_norm(input_states).unsqueeze(0), 
                                vision_depth = self.resize_depth(input_depth).unsqueeze(0),)
            self.cosine_sim_window.append(cos_distance)
            action = torch.cat([stacked_velo_pred, stacked_grip_pred], dim=-1)

            # Sub-goal transition
            if cos_distance.item() < 0.02 or len(self.cosine_sim_window) >= 10:
                self.cosine_sim_window = []
                self.future_frame_idx += 1

            action[..., -1] =  action[..., -1].sigmoid() > 0.4
            action[..., -1] = (action[..., -1] - 0.5) * 2  
            action = action[0,-1] 
            action = action.cpu().detach().to(dtype=torch.float16).numpy()

        return action



def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, reset=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, robot_obs=robot_obs, scene_obs=scene_obs, diverse_inst=diverse_inst)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset(subtask_i)
    start_info = env.get_info()

    # if debug:
    img_queue = []

    for step in range(EP_LEN):

        action = model.step(obs, lang_annotation)

        ee_pos = action[:3]
        ee_angle = action[3:6]
        gripper = action[6:]
        # action = (ee_pos, ee_angle, gripper) # Abs-action

        obs, _, _, current_info = env.step(action)
        
        img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
        img_queue.append(img_copy)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif( f'rollout-results/{sequence_i}-{subtask_i}-{subtask}-succ.gif', fps=30)
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(f'rollout-results/{sequence_i}-{subtask_i}-{subtask}-fail.gif', fps=30)
    return False



def eval_one_epoch_calvin_ddp(args, model, policy_model, dataset_path, text_encoder, tokenizer, eval_log_dir=None, debug=False, reset=False, diverse_inst=False):
    env = make_env(dataset_path)
    wrapped_model = ModelWrapper(model, policy_model, tokenizer, text_encoder)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)






