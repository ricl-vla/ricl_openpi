import json 
import os
import numpy as np 

def compute_and_save_simple_norm_stats_for_ricl(num_retrieved):
    norm_stats_basic_file_save_loc = "assets/norm_stats_simple.json"
    max_distance_file_save_loc = "assets/max_distance.json"

    outer_dir = "ricl_droid_preprocessing/collected_demos_training"
    task_dirs = [f"{outer_dir}/{task_dir}" for task_dir in os.listdir(outer_dir) if os.path.isdir(f"{outer_dir}/{task_dir}")]
    demo_dirs = [f"{task_dir}/{demo_dir}" for task_dir in task_dirs for demo_dir in os.listdir(task_dir) if os.path.isdir(f"{task_dir}/{demo_dir}")]
    all_states = []
    all_actions = []
    all_distances = []
    for demo_dir in demo_dirs:
        demo_data = np.load(f"{demo_dir}/processed_demo.npz")
        indices_and_dists = np.load(f"{demo_dir}/indices_and_distances.npz")
        all_states.append(demo_data["state"])
        all_actions.append(demo_data["actions"])
        all_distances.append(np.concatenate((indices_and_dists["distances"][:, :num_retrieved], indices_and_dists["distances"][:, -1:]), axis=1))
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_distances = np.concatenate(all_distances, axis=0)
    assert all_states.shape == all_actions.shape
    assert all_distances.shape == (all_states.shape[0], num_retrieved + 1)

    s_mean = np.mean(all_states, axis=0).tolist()   
    s_std = np.std(all_states, axis=0).tolist()
    s_q01 = np.percentile(all_states, 1, axis=0).tolist()
    s_q99 = np.percentile(all_states, 99, axis=0).tolist()

    a_mean = np.mean(all_actions, axis=0).tolist()
    a_std = np.std(all_actions, axis=0).tolist()
    a_q01 = np.percentile(all_actions, 1, axis=0).tolist()
    a_q99 = np.percentile(all_actions, 99, axis=0).tolist()

    d_max = np.max(all_distances)

    dict_to_save = {f"norm_stats": 
                    {"state": {"mean": s_mean, "std": s_std, "q01": s_q01, "q99": s_q99},
                     "actions": {"mean": a_mean, "std": a_std, "q01": a_q01, "q99": a_q99},
                    }
    }
    with open(norm_stats_basic_file_save_loc, 'w') as f:
        print(f'writing {norm_stats_basic_file_save_loc}')
        json.dump(dict_to_save, f, indent=2)
        print()

    another_dict_to_save = {"distances": {"max": d_max}}
    with open(max_distance_file_save_loc, 'w') as f:
        print(f'writing {max_distance_file_save_loc}')
        json.dump(another_dict_to_save, f, indent=2)
        print()
    
    return norm_stats_basic_file_save_loc

def convert_simple_norm_stats_to_retrieved_and_query_norm_stats(norm_stats_basic_file, num_retrieved):

    norm_stats_basic = json.load(open(norm_stats_basic_file, 'r'))

    new_norm_stats = {"norm_stats": {}}
    for key in ["state", "actions"]:
        for i in range(num_retrieved):
            prefix = f"retrieved_{i}_"
            new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]
        prefix = f"query_"
        new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]

    for file in ["assets/pi0_fast_droid_ricl/droid/norm_stats.json",
                ]:
        if not os.path.exists(file):
            os.makedirs(file.split("/norm_stats.json")[0], exist_ok=True)
            print(f'writing {file}')
            json.dump(new_norm_stats, open(file, 'w'), indent=2)


if __name__ == "__main__":
    num_retrieved = 4 # consequnce is distances
    output_file_name = compute_and_save_simple_norm_stats_for_ricl(num_retrieved = num_retrieved)
    convert_simple_norm_stats_to_retrieved_and_query_norm_stats(norm_stats_basic_file = output_file_name, num_retrieved = num_retrieved)