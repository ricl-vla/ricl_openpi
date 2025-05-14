# RICL: Re-training (a VLA) for In-Context Learning
A RICL version of the openpi repository focused on RICL-Pi0-FAST-DROID.

## Installation
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai
```

## Quickstart
Collect retrieval data for a new task as detailed [under Collecting data > retrieval data for testing the VLA in a new task](#retrieval-data-for-testing-the-vla-in-a-new-task).

Then download our checkpoint and serve it with the above retrieval data as detailed [under Serving RICL-Pi0-FAST-DROID in a new task > Serve the downloaded checkpoint](#serve-the-downloaded-checkpoint)

## Collecting data
Collect demos on your franka droid robot. The demos must be setup in the following directory structures for priming (at training time) and retrieval (at testing time). Please use the [original droid code](https://droid-dataset.github.io/droid/) to collect the demos.

### Priming data for re-training a VLA for in-context learning
```bash
# The following example structure is expected for the collected demos:
# preprocessing/
# ├── collected_demos_training/
# │   ├── {YYYY-MM-DD}_{task_1_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
# │   │── {YYYY-MM-DD}_{task_2_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
```
Please note that the folder containing many demos for a task must have the aboev specified name format. Ensure that the task prompt uses underscores to separate words. The folders inside each task folder can be named anything. But note that each folder inside a task folder, which has all the information for one collected demo, must atleast contain the following:
```bash
# │   │   ├── demo_0
# │   │   │   ├── traj.h5
# │   │   │   ├── recordings/
# |   │   │   │   ├── frames/
# |   │   │   │   │   ├── hand_camera/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_1/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_2/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
```
where the number of jpg files (in each camera folder) is equal to the number of timesteps where the controller is active. You need to extract these from the svo files created by the droid code. The traj.h5 file is the one saved by the original droid code containing the proprioceptive data and actions. These are the only files we use. We ignore everything else in the demo folder.

### Retrieval data for testing the VLA in a new task
Similarly, for retrieval data in a new task at test time, we expect the following structure:
```bash
# preprocessing/
# ├── collected_demos/
# │   ├── {YYYY-MM-DD}_{new_task_prompt}/
# │   │   ├── demo_0
# |   │   ├── demo_1
# |   │   ├── ...
```

## Downloading our datasets
Priming (training) data: `git clone https://huggingface.co/datasets/ricl-vla/collected_demos_training ./preprocessing/collected_demos_training`

Retrieval (testing) data in many new tasks: `git clone https://huggingface.co/datasets/ricl-vla/collected_demos ./preprocessing/collected_demos`

Both of the above can also be found at [this huggingface link](https://huggingface.co/ricl-vla).

## Preprocessing
* First cd into the folder
```bash
cd preprocessing
```

* Process the priming demos for re-training a VLA for in-context learning
```bash
python process_collected_demos.py --dir_of_dirs=collected_demos_training
python retrieve_within_collected_demo_groups.py
```

* Process the retrieval demos for testing the VLA in a new task
```bash
python process_collected_demos.py --dir_of_dirs=collected_demos
```

## Re-training for in-context learning (RICL)
* Compute norm stats after processing the priming demos as follows:
```bash
python scripts/setup_norm_states_for_ricl.py
```

* Create RICL-Pi0-FAST-DROID
```bash
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl --exp-name={YOUR_EXPERIMENT_NAME_HERE} --overwrite
```

## Serving RICL-Pi0-FAST-DROID in a new task
### Serve RICL-Pi0-FAST-DROID's checkpoint after three epochs of training 
The step corresponding to this for our dataset is 5400.
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl --policy.dir=checkpoints/pi0_fast_droid_ricl/{YOUR_EXPERIMENT_NAME_HERE}/5400 --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

### Serve the downloaded checkpoint
Download our checkpoint: `git clone https://huggingface.co/ricl-vla/pi0_fast_droid_ricl_checkpoint`

You can serve it as follows:
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl --policy.dir=pi0_fast_droid_ricl_checkpoint --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

Our checkpoint can also be found at [this huggingface link](https://huggingface.co/ricl-vla).

### On the laptop connected to the franka droid robot
You also have to run the client in `examples/droid/main_ricl.py` on the laptop connexted to the franka droid robot following the original repository's instructions.

## Finetuning like you pretrain to create RICL-Pi0-FAST-DROID-Finetuned
* This requeires a bit more preprocessing of the retrieval demos in the new task
```bash
python retrieve_within_collected_demo_groups.py --folder_name=collected_demos
```

* Create RICL-Pi0-FAST-DROID-Finetuned
```bash
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl___finetune_on_new_task --exp-name={YOUR_EXPERIMENT_NAME_HERE} --overwrite
```
Please make sure the config of `pi0_fast_droid_ricl___finetune_on_new_task` in `src/training/config.py` points to the correct retrieval demos folder that you would like to finetune on.

* Serve RICL-Pi0-FAST-DROID-Finetuned
```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_droid_ricl___finetune_on_new_task --policy.dir=checkpoints/pi0_fast_droid_ricl___finetune_on_new_task/{YOUR_EXPERIMENT_NAME_HERE}/999 --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{new_task_prompt}
```

Run the client on the laptop connected to the franka droid robot to test the finetuned policy.

## Credits
This repository is based on the [openpi](https://github.com/openai/openpi) repository, without which this work would not have been possible.