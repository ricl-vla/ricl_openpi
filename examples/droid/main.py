# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import json  # Added for JSON export
import threading  # Added for non-blocking visualization
import time

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import matplotlib.pyplot as plt  # Added for visualization
import matplotlib.animation as animation  # Added for dynamic updates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

faulthandler.enable()


# Define a custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "25455306" # e.g., "24259877"
    right_camera_id: str = "23007103" # fix: "27085680"  move: # "26368109"  
    wrist_camera_id: str = "14436910"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        "left"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 800
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

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


def save_visualization_snapshot(fig, save_path):
    """Save a snapshot of the current visualization figure."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization snapshot saved to {save_path}")
    except Exception as e:
        print(f"Failed to save visualization snapshot: {e}")


def main(args: Args):
    print("Entered main!")
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Initialize DataFrame and prepare markdown logging
    df = pd.DataFrame(columns=["success", "duration", "video_filename", "instruction", "comment"])
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    date = datetime.datetime.now().strftime("%m%d")
    # Get main category for this evaluation session
    main_category = input("Enter main category for this evaluation session: ")
    os.makedirs(f"results/log/{date}", exist_ok=True)
    markdown_file = f"results/log/{date}/eval_{main_category}.md"
    os.makedirs(f"results/log/{date}/action_plot", exist_ok=True)
    os.makedirs(f"results/log/{date}/action_json", exist_ok=True)


    # Create markdown header
    with open(markdown_file, "a") as f:
        f.write(f"# Pi0-FAST Evaluation: {main_category}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        joint_position_file = f"results/log/{date}/eval_{main_category}_{timestamp}_joints.csv"
        # Create a filename-safe version of the instruction
        safe_instruction = instruction.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]  # limit length
        video = []
        wrist_video = []  # New list for wrist camera frames
        
        # Add data storage for plotting and JSON history
        joint_positions = []
        action_history = []
        action_chunks_history = []  # To store action trunk history
        chunk_timestamps = []  # When each chunk was received
        current_chunk_index = -1  # Track which chunk we're in
        
        # Setup visualization
        plt.ion()  # Turn on interactive mode for live updates
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 subplots instead of 2
        fig.suptitle(f"Instruction: {instruction}", fontsize=12)
        
        # First subplot for joint positions
        joint_pos_lines = []
        for i in range(7):  # 7 joint positions
            line, = axs[0].plot([], [], label=f'Joint {i+1}')
            joint_pos_lines.append(line)
        axs[0].set_title('Joint Positions')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('Position')
        axs[0].legend(loc='upper left', fontsize='small')
        
        # Second subplot for velocity actions
        action_lines = []
        for i in range(7):  # 7 joint velocity actions
            line, = axs[1].plot([], [], label=f'Joint {i+1} Velocity')
            action_lines.append(line)
        axs[1].set_title('Joint Velocity Actions')
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Action Value')
        axs[1].legend(loc='upper left', fontsize='small')
        
        # Third subplot for gripper state
        gripper_action_line, = axs[2].plot([], [], 'r-', linewidth=2, label='Gripper Action')
        gripper_position_line, = axs[2].plot([], [], 'b-', linewidth=2, label='Gripper Position')
        axs[2].set_title('Gripper State')
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylabel('Value [0-1]')
        axs[2].set_ylim(-0.1, 1.1)  # Gripper values are between 0 and 1
        axs[2].legend(loc='upper left', fontsize='small')
        
        # Text area for real-time statistics
        status_text = axs[0].text(0.78, 0.95, '', transform=axs[0].transAxes, 
                                 verticalalignment='top', horizontalalignment='right',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add vertical lines to mark chunk boundaries (will be added during execution)
        chunk_boundary_lines = []
        
        # Initialize plot with empty data
        xdata = []
        joint_ydata = [[] for _ in range(7)]
        action_ydata = [[] for _ in range(8)]  # 8th is gripper
        gripper_action_ydata = []
        gripper_position_ydata = []
        early_stop_markers = []  # Store early stop timesteps
        
        # Track early stopping state
        is_early_stopped = False  # Current state
        early_stop_regions = []  # List of (start, end) tuples
        early_stop_start = None  # Start of current early stop region
        consecutive_stops = 0  # Counter for consecutive stopped frames
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Start time for execution metrics
        start_time = time.time()

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )
                # Save both camera views
                video.append(curr_obs[f"{args.external_camera}_image"])
                wrist_video.append(curr_obs["wrist_image"])
                
                # Store joint positions for visualization
                joint_positions.append(curr_obs["joint_position"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    current_chunk_index += 1

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    server_request_start = time.time()
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    server_request_duration = time.time() - server_request_start
                    assert pred_action_chunk.shape == (10, 8)
                    
                    # Record the action chunk in history with timestep and timing information
                    action_chunks_history.append({
                        "chunk_index": current_chunk_index,
                        "timestep": t_step,
                        "time": time.time() - start_time,
                        "server_request_duration": server_request_duration,
                        "chunk": pred_action_chunk.tolist(),  # Convert numpy array to list for JSON serialization
                    })
                    chunk_timestamps.append(t_step)
                    
                    # Add vertical line to mark chunk boundary on the plot
                    for ax in axs:
                        # Add a vertical line at this timestep to show when a new chunk was received
                        line = ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5)
                        chunk_boundary_lines.append(line)

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
                
                # Check for "early stopping" pattern - when all joint velocities are zero
                joint_velocities = action[:-1]  # All except gripper
                is_current_stopped = bool(np.all(np.abs(joint_velocities) < 1e-2))  # Convert numpy.bool_ to Python bool
                
                if is_current_stopped:
                    consecutive_stops += 1
                    early_stop_markers.append(t_step)
                    
                    # If this is the start of a new stopping region
                    if not is_early_stopped and consecutive_stops == 1:
                        is_early_stopped = True
                        early_stop_start = t_step
                        print(f"Early stopping began at timestep {early_stop_start}")
                else:
                    # If we were in an early stopped state but now there's movement
                    if is_early_stopped:
                        is_early_stopped = False
                        early_stop_end = t_step - 1
                        early_stop_regions.append((early_stop_start, early_stop_end))
                        print(f"Early stopping ended at timestep {early_stop_end}, duration: {early_stop_end - early_stop_start + 1} steps")
                    consecutive_stops = 0
                
                # Store action for history and visualization
                action_record = {
                    "timestep": t_step,
                    "time": time.time() - start_time,
                    "chunk_index": current_chunk_index,
                    "action_index_in_chunk": actions_from_chunk_completed - 1,
                    "action": action.tolist(),
                    "joint_position": curr_obs["joint_position"].tolist(),
                    "gripper_position": curr_obs["gripper_position"].tolist(),
                    "is_early_stop": is_current_stopped
                }
                action_history.append(action_record)
                
                # Update visualization data
                xdata.append(t_step)
                
                # Joint positions data
                for i in range(7):
                    joint_ydata[i].append(curr_obs["joint_position"][i])
                
                # Action data (joint velocities and gripper)
                for i in range(7):  # Only the 7 joint velocity actions
                    action_ydata[i].append(action[i])
                action_ydata[7].append(action[7])  # Gripper action
                
                # Update gripper data
                gripper_action_ydata.append(action[7])
                gripper_position_ydata.append(curr_obs["gripper_position"][0])  # Use actual gripper position
                
                # Update plot every 5 timesteps to avoid slowing down execution
                if t_step % 5 == 0:
                    # Update joint position lines
                    for i, line in enumerate(joint_pos_lines):
                        line.set_data(xdata, joint_ydata[i])
                    
                    # Update action lines (only joint velocities)
                    for i, line in enumerate(action_lines):
                        if i < len(action_ydata) - 1:  # Skip gripper in action lines
                            line.set_data(xdata, action_ydata[i])
                    
                    # Update gripper lines
                    gripper_action_line.set_data(xdata, gripper_action_ydata)
                    gripper_position_line.set_data(xdata, gripper_position_ydata)
                    
                    # Update early stop markers
                    for ax in axs:
                        # Clear any existing early stop markers
                        for artist in ax.findobj(match=lambda x: hasattr(x, 'early_stop_marker')):
                            artist.remove()
                    
                    # Add new markers for early stops
                    for stop_time in early_stop_markers:
                        for ax in axs:
                            marker = ax.axvline(x=stop_time, color='green', linestyle='-.', linewidth=1, alpha=0.2)
                            marker.early_stop_marker = True  # Tag it for later removal
                    
                    # Add shaded regions for completed early stopping periods
                    for region in early_stop_regions:
                        start, end = region
                        for ax in axs:
                            rect = ax.axvspan(start, end, color='green', alpha=0.1)
                            rect.early_stop_marker = True
                    
                    # Add shaded region for current early stopping period (if any)
                    if is_early_stopped and early_stop_start is not None:
                        for ax in axs:
                            rect = ax.axvspan(early_stop_start, t_step, color='green', alpha=0.1)
                            rect.early_stop_marker = True
                    
                    # Update status text
                    current_time = time.time() - start_time
                    status_text.set_text(f"Timestep: {t_step}\n"
                                        f"Time: {current_time:.2f}s\n"
                                        f"Chunk: {current_chunk_index}\n"
                                        f"Action in chunk: {actions_from_chunk_completed-1}/8\n"
                                        f"Early stops: {len(early_stop_markers)}\n"
                                        f"Stop regions: {len(early_stop_regions)}" +
                                        (f"\nCurrent stop: {consecutive_stops}" if is_early_stopped else ""))
                    
                    # Rescale axes
                    for ax in axs:
                        ax.relim()
                        ax.autoscale_view()
                    
                    # Draw and refresh plot
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                env.step(action) # droid actually apply the action
            except KeyboardInterrupt:
                break

        # Final execution time
        total_execution_time = time.time() - start_time
        
        # Complete any ongoing early stopping period
        if is_early_stopped and early_stop_start is not None:
            early_stop_regions.append((early_stop_start, t_step))
            print(f"Final early stopping period ended at the end of experiment, duration: {t_step - early_stop_start + 1} steps")
        
        # Analyze early stopping patterns
        if early_stop_markers:
            # Complete any ongoing early stopping period
            if is_early_stopped and early_stop_start is not None:
                early_stop_regions.append((early_stop_start, t_step))
                print(f"Final early stopping period ended at the end of experiment, duration: {t_step - early_stop_start + 1} steps")
            
            # Process early stopping regions data
            region_durations = [end - start + 1 for start, end in early_stop_regions]
            total_stopped_frames = sum(region_durations)
            
            # Count early stops at the beginning of chunks
            beginning_chunk_stops = sum(1 for t in early_stop_markers 
                                      if t in chunk_timestamps)
            
            # Count early stops at the end of chunks
            chunk_end_stops = 0
            for i, chunk_start in enumerate(chunk_timestamps):
                if i < len(chunk_timestamps)-1:
                    # Check if stop happened just before next chunk
                    chunk_end = chunk_timestamps[i+1] - 1
                    chunk_end_stops += sum(1 for t in early_stop_markers 
                                         if t == chunk_end)
                else:
                    # Last chunk
                    pass
            
            print("\nEarly stopping analysis:")
            print(f"  Total early stopped frames: {len(early_stop_markers)}")
            print(f"  Number of continuous stopping regions: {len(early_stop_regions)}")
            if early_stop_regions:
                avg_duration = total_stopped_frames / len(early_stop_regions)
                max_duration = max(region_durations)
                min_duration = min(region_durations)
                print(f"  Average region duration: {avg_duration:.1f} frames")
                print(f"  Longest region: {max_duration} frames")
                print(f"  Shortest region: {min_duration} frames")
                
                # Find the longest stopping region
                longest_idx = region_durations.index(max_duration)
                longest_start, longest_end = early_stop_regions[longest_idx]
                print(f"  Longest stopping region: frames {longest_start}-{longest_end}")
                
                # Calculate percentage of experiment spent in stopping state
                stopping_percentage = (total_stopped_frames / (t_step + 1)) * 100
                print(f"  Experiment time in stopped state: {stopping_percentage:.1f}%")
            
            print(f"  Stops at chunk beginnings: {beginning_chunk_stops} frames")
            print(f"  Stops at chunk ends: {chunk_end_stops} frames")
            
            # Look at positions within chunks
            chunk_positions = {}
            for t in early_stop_markers:
                # Find which chunk this belongs to
                chunk_idx = -1
                pos_in_chunk = -1
                for i, chunk_start in enumerate(chunk_timestamps):
                    if i < len(chunk_timestamps)-1:
                        if chunk_start <= t < chunk_timestamps[i+1]:
                            chunk_idx = i
                            pos_in_chunk = t - chunk_start
                            break
                    else:
                        # Last chunk
                        if chunk_start <= t:
                            chunk_idx = i
                            pos_in_chunk = t - chunk_start
                
                if pos_in_chunk != -1:
                    if pos_in_chunk not in chunk_positions:
                        chunk_positions[pos_in_chunk] = 0
                    chunk_positions[pos_in_chunk] += 1
            
            print("\nPositions within chunks where early stops occur:")
            for pos, count in sorted(chunk_positions.items()):
                print(f"  Position {pos}: {count} frames ({count/len(early_stop_markers)*100:.1f}%)")
        
        # Save a snapshot of the visualization before closing
        vis_snapshot_path = f"results/log/{date}/action_plot/eval_{main_category}_{instruction}_visualization.png"
        save_visualization_snapshot(fig, vis_snapshot_path)
        
        # Close the plot after the rollout
        plt.close(fig)
        
        # Save action trunk history to JSON
        action_history_file = f"results/log/{date}/action_json/eval_{main_category}_{instruction}_action_history.json"
        
        # Include all relevant metadata
        action_history_data = {
            "metadata": {
                "instruction": instruction,
                "timestamp": timestamp,
                "date": date,
                "category": main_category,
                "total_timesteps": t_step + 1,
                "total_execution_time": total_execution_time,
                "open_loop_horizon": args.open_loop_horizon,
                "external_camera": args.external_camera,
                "left_camera_id": args.left_camera_id,
                "right_camera_id": args.right_camera_id,
                "wrist_camera_id": args.wrist_camera_id,
                "remote_host": args.remote_host,
                "remote_port": args.remote_port
            },
            "action_chunks": action_chunks_history,
            "chunk_timestamps": [int(t) for t in chunk_timestamps],
            "early_stops": {
                "count": len(early_stop_markers),
                "timesteps": [int(t) for t in early_stop_markers],  # Ensure these are native Python ints
                "regions": [{"start": int(start), "end": int(end), "duration": int(end - start + 1)} 
                          for start, end in early_stop_regions],
                "total_regions": len(early_stop_regions),
                "total_stopped_frames": sum(end - start + 1 for start, end in early_stop_regions) if early_stop_regions else 0,
                "threshold": float(1e-2)  # Ensure this is a native Python float
            },
            "detailed_actions": action_history
        }
        
        with open(action_history_file, 'w') as f:
            json.dump(action_history_data, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"Action trunk history saved to {action_history_file}")

        # Stack videos side by side
        video = np.stack(video)
        wrist_video = np.stack(wrist_video)
        
        # Ensure both videos have the same height for side-by-side display
        target_height = min(video.shape[1], wrist_video.shape[1])
        target_width = min(video.shape[2], wrist_video.shape[2])
        
        # Resize both videos to the same dimensions
        video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in video])
        wrist_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in wrist_video])
        
        # Stack videos horizontally
        combined_video = np.concatenate([video_resized, wrist_video_resized], axis=2)

        date = datetime.datetime.now().strftime("%m%d")
        save_dir = f"results/videos/{date}"
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{args.external_camera }_{safe_instruction}_{timestamp}.mp4")
  
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        # Get success value
        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec: "
            )
            if success == "y" or success == "1":
                success = 1.0
            elif success == "n" or success == "0":
                success = 0.0
            elif success == "-1":
                success = -1  #\
            else:
                try:
                    success = float(success) / 100
                    if not (0 <= success <= 1):
                        print(f"Success must be a number in [0, 100] but got: {success * 100}")
                        success = None
                except ValueError:
                    print("Invalid input. Please enter y, n, or a number between 0-100")
                    success = None

        # Get comment about the result
        comment = input("Enter comment about this trial: ")

        # Append to markdown file
        with open(markdown_file, "a") as f:
            f.write(f"### Trial {len(df) + 1}: {instruction}\n")
            f.write(f"- Success: {success * 100}%\n")
            f.write(f"- Duration: {t_step} steps\n")
            f.write(f"- Video: [{os.path.basename(save_filename)}]({save_filename})\n")
            f.write(f"- Comment: {comment}\n\n")


        # Update DataFrame
        df = pd.concat([df, pd.DataFrame([{
            "success": success,
            "duration": t_step,
            "video_filename": save_filename,
            "instruction": instruction,
            "comment": comment
        }])], ignore_index=True)

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    # Save CSV alongside markdown
    csv_filename = markdown_file.replace(".md", ".csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {markdown_file} and {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
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
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
