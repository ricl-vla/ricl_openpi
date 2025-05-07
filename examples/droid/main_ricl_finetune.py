# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()


@dataclasses.dataclass
class Args:
    # Hardware parameters
    top_camera_id: str = "26368109" # other camera is 25455306
    right_camera_id: str = "23007103" #  
    wrist_camera_id: str = "14436910"  # 

    # Rollout parameters
    max_timesteps: int = 2000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 3

    # Remote server parameters
    remote_host: str = "158.130.52.14"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Evaluation parameters
    eval_name: str = "default"  # Name for this evaluation session


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    og_open_loop_horizon = args.open_loop_horizon
    assert og_open_loop_horizon in [3, 8]
    print("Entered main!")
    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    print("Connecting to the policy server...")
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print("Connected to the policy server!")

    # Initialize
    date = datetime.datetime.now().strftime("%m%d")

    while True:
        # Get text inputs
        main_category = input("Enter the prefix for this video filename: ")
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        # Create a filename-safe version of the instruction
        safe_instruction = instruction.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]  # limit length
        top_video = []
        right_video = []
        wrist_video = []  # New list for wrist camera frames
        #joint_positions = []
        #action_state = []


        bar = tqdm.tqdm(range(args.max_timesteps))
        switch_time = None
        second_switch_time = None
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            try:

                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )

                # Save both camera views
                top_video.append(curr_obs[f"top_image"])
                right_video.append(curr_obs[f"right_image"])
                wrist_video.append(curr_obs["wrist_image"])


                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "query_right_image": image_tools.resize_with_pad(
                            curr_obs[f"right_image"], 224, 224
                        ),
                        "query_top_image": image_tools.resize_with_pad(
                            curr_obs[f"top_image"], 224, 224
                        ),
                        "query_wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "query_state": np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"]]),
                        "query_prompt": instruction,
                        "prefix": main_category,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["query_actions"]
                    assert pred_action_chunk.shape in [(10, 8), (15, 8)]

                    if args.open_loop_horizon == 3 and switch_time is None and pred_action_chunk[-1, -1] > 0.9:
                        print(f'switch time is {t_step}')
                        switch_time = t_step

                    if args.open_loop_horizon == 3 and switch_time is not None and second_switch_time is None and t_step - switch_time > 10:
                        print(f"Switching to open loop horizon of 5 for grasping at second_switch_time of {t_step}")
                        args.open_loop_horizon = 5 # for grasping
                        second_switch_time = t_step

                    # if args.open_loop_horizon == 8 and second_switch_time is not None and t_step - second_switch_time > 10:
                    #     print(f"Switching back to open loop horizon of 3 for moving at time {t_step}")
                    #     args.open_loop_horizon = 3 # back to 3 for moving
                    #     switch_time = None
                    #     second_switch_time = None
                    
                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                #action_state.append(action)
                #joint_positions.append(curr_obs["joint_position"])

                env.step(action)
            except KeyboardInterrupt:
                break

        # Stack videos side by side
        top_video = np.stack(top_video)
        right_video = np.stack(right_video)
        wrist_video = np.stack(wrist_video)
        #action_csv = np.stack(action_state)
        #joint_csv = np.stack(joint_positions)
        #combined_action_csv = np.concatenate([action_csv, joint_csv], axis=1)
        
        # Ensure both videos have the same height for side-by-side display
        target_height = min(top_video.shape[1], wrist_video.shape[1], right_video.shape[1])
        target_width = min(top_video.shape[2], wrist_video.shape[2], right_video.shape[2])
        
        # Resize both videos to the same dimensions
        top_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in top_video])
        right_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in right_video])
        wrist_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in wrist_video])
        
        # Stack videos horizontally
        combined_video = np.concatenate([top_video_resized, right_video_resized, wrist_video_resized], axis=2)

        date = datetime.datetime.now().strftime("%m%d")
        save_dir = f"results_ricl/videos/{date}"
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{main_category.replace(' ', '_')}_{safe_instruction}_{timestamp}.mp4")
        
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename, codec="libx264")

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()
        if args.open_loop_horizon != og_open_loop_horizon:
            args.open_loop_horizon = og_open_loop_horizon

def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    top_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the stereo_choice = "left" below refers to one of the two cameras in the stereo pair.
        # The model is only trained on this one of the stereo cams, so we only feed that one.
        stereo_choice = "left"
        if args.top_camera_id in key and stereo_choice in key:
            top_image = image_observations[key]
        elif args.right_camera_id in key and stereo_choice in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and stereo_choice in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    top_image = top_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    top_image = top_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([top_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "top_image": top_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
