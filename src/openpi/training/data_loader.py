from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.models.pi0_fast_ricl as _pi0_fast_ricl
import openpi.training.config as _config
import openpi.transforms as _transforms
import json

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples
    

def get_action_chunk(action_joint_vels, action_gripper_pos, step_idx, action_horizon):
    num_steps = len(action_joint_vels)
    assert action_joint_vels.shape == (num_steps, 7) and action_gripper_pos.shape == (num_steps, 1)
    action_chunk = []
    for i in range(action_horizon):
        if step_idx+i < num_steps:
            action_chunk.append(np.concatenate([action_joint_vels[step_idx+i], action_gripper_pos[step_idx+i]], axis=0))
        else:
            action_chunk.append(np.concatenate([np.zeros(action_joint_vels.shape[-1], dtype=np.float32), action_gripper_pos[-1]], axis=0))
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, 8), f"{action_chunk.shape=}"
    return action_chunk


class Pi0FastDroidFinetuneDataset(Dataset):
    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        assert finetuning_collected_demos_dir is not None
        collected_demos_infos = {k: json.load(open(f"{finetuning_collected_demos_dir}/{k}.json")) for k in ['ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_fols', 'groups_to_ep_idxs']}
        
        # files from the collected demos for training
        indices_files = [] 
        for group_name, ep_fols in collected_demos_infos["groups_to_ep_fols"].items():
            for ep_fol in ep_fols:
                indices_files.append(f"ricl_droid_preprocessing/{ep_fol}/indices_and_distances.npz")
        
        # actual loading...
        count_collected_demos = 0
        all_query_indices = []
        for file_idx, file_path in enumerate(indices_files):
            indices_and_dists = np.load(file_path)
            query_indices = indices_and_dists["query_indices"]
            num_steps = query_indices.shape[0]
            assert query_indices.shape == (num_steps, 2) and query_indices.dtype == np.int32
            expected_query_indices = np.array([[100000+file_idx, i] for i in range(num_steps)], dtype=np.int32)
            assert np.allclose(query_indices, expected_query_indices), f"{query_indices=}, {expected_query_indices=}"
            all_query_indices.append(query_indices)
            count_collected_demos += num_steps
        print(f"num states in collected demos given by count_collected_demos: {count_collected_demos}")
        all_query_indices = np.concatenate(all_query_indices, axis=0)
        len_dataset = all_query_indices.shape[0]
        print(f"len_dataset: {len_dataset}")
        assert len_dataset == count_collected_demos
        assert all_query_indices.shape == (len_dataset, 2) and all_query_indices.dtype == np.int32

        # load all data paths 
        all_ep_idxs = list(np.unique(all_query_indices[:, 0]))
        all_ep_data_paths = {ep_idx: 
                                    f"ricl_droid_preprocessing/{collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)]}/processed_demo.npz"
                            for ep_idx in all_ep_idxs}
        common_prompt = " ".join(collected_demos_infos['ep_idxs_to_fol']['100000'].split("/")[1].split("_")[1:])
        print(f'num episodes: {len(all_ep_idxs)}')
        print(f"common_prompt: {common_prompt}")

        # save
        self.len_dataset = len_dataset
        self.all_ep_data_paths = all_ep_data_paths
        self.common_prompt = common_prompt
        self.all_query_indices = all_query_indices
        self.action_horizon = model_config.action_horizon

    def __getitem__(self, index: SupportsIndex) -> dict:
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]
        ep_data = np.load(self.all_ep_data_paths[query_ep_idx])
        data = {'observation/exterior_image_1_left': ep_data['right_image'][query_step_idx],
                'observation/wrist_image_left': ep_data['wrist_image'][query_step_idx],
                'observation/joint_position': ep_data['state'][query_step_idx][:-1],
                'observation/gripper_position': ep_data['state'][query_step_idx][-1:],
                'actions': get_action_chunk(ep_data['actions'][:, :-1], ep_data['actions'][:, -1:], query_step_idx, self.action_horizon),
                'prompt': self.common_prompt}
        return data

    def __len__(self) -> int:
        return self.len_dataset


class RiclDroidDataset(Dataset):
    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        # setup
        num_retrieved_observations = model_config.num_retrieved_observations
        knn_k = 100
        assert num_retrieved_observations <= knn_k
        embedding_type = "embeddings__wrist_image_left" # retrieval based on embeddings of wrist images
        indices_and_dists_fol = f"ricl_droid_preprocessing/droid_new_broken_up_indices_and_distances/chosenIDscene_id_numepisodes20_embtype{embedding_type}_knnk100"
        outer_dir = "ricl_droid_preprocessing/collected_demos_training" if finetuning_collected_demos_dir is None else finetuning_collected_demos_dir
        collected_demos_infos = {k: json.load(open(f"{outer_dir}/{k}.json")) for k in ['ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_fols', 'groups_to_ep_idxs']}
        # load indices_and_dists
        all_retrieved_indices = []
        all_query_indices = []
        all_distances = []
        
        ## files from the droid dataset
        # indices_files = os.listdir(indices_and_dists_fol)
        # indices_files = [os.path.join(indices_and_dists_fol, f) for f in indices_files]
        indices_files = [] ## no files from droid dataset

        # files from the collected demos for training
        for group_name, ep_fols in collected_demos_infos["groups_to_ep_fols"].items():
            for ep_fol in ep_fols:
                indices_files.append(f"ricl_droid_preprocessing/{ep_fol}/indices_and_distances.npz")
        # actual loading...
        count_droid = 0
        count_collected_demos = 0
        for file_path in indices_files:
            indices_and_dists = np.load(file_path)
            query_indices, retrieved_indices = indices_and_dists["query_indices"], indices_and_dists["retrieved_indices"][:, :num_retrieved_observations, :]
            distances = np.concatenate((indices_and_dists["distances"][:, :num_retrieved_observations], indices_and_dists["distances"][:, -1:]), axis=1)
            num_steps = query_indices.shape[0]
            assert retrieved_indices.shape == (num_steps, num_retrieved_observations, 2) and retrieved_indices.dtype == np.int32 
            assert query_indices.shape == (num_steps, 2) and query_indices.dtype == np.int32
            all_retrieved_indices.append(retrieved_indices)
            all_query_indices.append(query_indices)
            all_distances.append(distances)
            if "collected_demos_training" in file_path or "collected_demos" in file_path:
                count_collected_demos += num_steps
            else:
                count_droid += num_steps
        print(f"count_droid: {count_droid}, count_collected_demos: {count_collected_demos}")
        all_retrieved_indices = np.concatenate(all_retrieved_indices, axis=0)
        all_query_indices = np.concatenate(all_query_indices, axis=0)
        all_distances = np.concatenate(all_distances, axis=0)
        len_dataset = all_retrieved_indices.shape[0]
        print(f"len_dataset: {len_dataset}")
        assert len_dataset == count_droid + count_collected_demos
        assert all_retrieved_indices.shape == (len_dataset, num_retrieved_observations, 2) and all_retrieved_indices.dtype == np.int32
        assert all_query_indices.shape == (len_dataset, 2) and all_query_indices.dtype == np.int32
        assert all_distances.shape == (len_dataset, num_retrieved_observations + 1) and all_distances.dtype == np.float64
        
        # normalize all_distances and convert to float32
        max_dist_value = json.load(open(f"assets/max_distance.json", 'r'))['distances']['max']
        if finetuning_collected_demos_dir is None:
            assert max_dist_value == np.max(all_distances), f"{max_dist_value=} from norm stats time does not match {np.max(all_distances)=} from dataset"
            print(f'max distance value: {max_dist_value}')
        all_distances = all_distances / max_dist_value
        all_distances = all_distances.astype(np.float32)

        # load all data paths 
        ds_name = f"droid_new"
        ds_fol = f"ricl_droid_preprocessing/{ds_name}_broken_up"
        all_ep_idxs = list(np.unique(all_retrieved_indices[:, :, 0])) + list(np.unique(all_query_indices[:, 0]))
        all_ep_data_paths = {ep_idx: 
                                    f"{ds_fol}/episode_{ep_idx}.npz" 
                                    if ep_idx < 100000 else 
                                    f"ricl_droid_preprocessing/{collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)]}/processed_demo.npz"
                            for ep_idx in all_ep_idxs}
        all_ep_prompts = {ep_idx: 
                                    json.load(open(f"{ds_fol}/episode_{ep_idx}.json"))["language_instruction"]  
                                    if ep_idx < 100000 else 
                                    " ".join(collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)].split("/")[1].split("_")[1:])
                            for ep_idx in all_ep_idxs}
        
        # if all episode prompts are the same, print the first prompt
        if all(all_ep_prompts[ep_idx] == all_ep_prompts[list(all_ep_prompts.keys())[0]] for ep_idx in all_ep_prompts):
            print(f"all {len(all_ep_prompts)} episode prompts are the same: {all_ep_prompts[list(all_ep_prompts.keys())[0]]}")

        # save
        self.len_dataset = len_dataset
        self.all_ep_data_paths = all_ep_data_paths
        self.all_ep_prompts = all_ep_prompts
        self.all_retrieved_indices = all_retrieved_indices
        self.all_query_indices = all_query_indices
        self.all_distances = all_distances
        self.use_action_interpolation = model_config.use_action_interpolation
        self.lamda = model_config.lamda
        self.action_horizon = model_config.action_horizon

    def __getitem__(self, index: SupportsIndex) -> dict:
        retrieved_indices = self.all_retrieved_indices[index, :, :]
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]
        
        ep_idxs = list(np.unique(retrieved_indices[:, 0])) + [query_ep_idx]
        ep_data = {ep_idx: np.load(self.all_ep_data_paths[ep_idx]) for ep_idx in ep_idxs}
        data = {}
        random_ext_img = np.random.choice(["left", "right"])
        for ct, (ep_idx, step_idx) in enumerate(retrieved_indices):
            prefix = f"retrieved_{ct}_"
            if ep_idx < 100000:
                data[f"{prefix}top_image"] = ep_data[ep_idx]["observation__exterior_image_1_left"][step_idx]
                data[f"{prefix}right_image"] = ep_data[ep_idx]["observation__exterior_image_2_left"][step_idx]
                data[f"{prefix}wrist_image"] = ep_data[ep_idx]["observation__wrist_image_left"][step_idx]
                data[f"{prefix}state"] = np.concatenate([ep_data[ep_idx]["observation__joint_position"][step_idx], ep_data[ep_idx]["observation__gripper_position"][step_idx]], axis=0)
                data[f"{prefix}actions"] = get_action_chunk(ep_data[ep_idx]["action_dict__joint_velocity"], ep_data[ep_idx]["action_dict__gripper_position"], step_idx, self.action_horizon)
            else:
                data[f"{prefix}top_image"] = ep_data[ep_idx]["top_image"][step_idx]
                data[f"{prefix}right_image"] = ep_data[ep_idx]["right_image"][step_idx]
                data[f"{prefix}wrist_image"] = ep_data[ep_idx]["wrist_image"][step_idx]
                data[f"{prefix}state"] = ep_data[ep_idx]["state"][step_idx]
                data[f"{prefix}actions"] = get_action_chunk(ep_data[ep_idx]["actions"][:, :-1], ep_data[ep_idx]["actions"][:, -1:], step_idx, self.action_horizon)
            data[f"{prefix}prompt"] = self.all_ep_prompts[ep_idx]
        
        prefix = "query_"
        if query_ep_idx < 100000:
            data[f"{prefix}top_image"] = ep_data[query_ep_idx]["observation__exterior_image_1_left"][query_step_idx]
            data[f"{prefix}right_image"] = ep_data[query_ep_idx]["observation__exterior_image_2_left"][query_step_idx]
            data[f"{prefix}wrist_image"] = ep_data[query_ep_idx]["observation__wrist_image_left"][query_step_idx]
            data[f"{prefix}state"] = np.concatenate([ep_data[query_ep_idx]["observation__joint_position"][query_step_idx], ep_data[query_ep_idx]["observation__gripper_position"][query_step_idx]], axis=0)
            data[f"{prefix}actions"] = get_action_chunk(ep_data[query_ep_idx]["action_dict__joint_velocity"], ep_data[query_ep_idx]["action_dict__gripper_position"], query_step_idx, self.action_horizon)
        else:
            data[f"{prefix}top_image"] = ep_data[query_ep_idx]["top_image"][query_step_idx]
            data[f"{prefix}right_image"] = ep_data[query_ep_idx]["right_image"][query_step_idx]
            data[f"{prefix}wrist_image"] = ep_data[query_ep_idx]["wrist_image"][query_step_idx]
            data[f"{prefix}state"] = ep_data[query_ep_idx]["state"][query_step_idx]
            data[f"{prefix}actions"] = get_action_chunk(ep_data[query_ep_idx]["actions"][:, :-1], ep_data[query_ep_idx]["actions"][:, -1:], query_step_idx, self.action_horizon)
        data[f"{prefix}prompt"] = self.all_ep_prompts[query_ep_idx]

        if self.use_action_interpolation:
            # read distances
            distances = self.all_distances[index, :]
            # then compute exp(-lamda * distances)
            data["exp_lamda_distances"] = np.exp(-self.lamda * distances).reshape(-1, 1)

        return data

    def __len__(self) -> int:
        return self.len_dataset


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    if "ricl" in config.name:
        dataset = RiclDroidDataset(config.model, config.finetuning_collected_demos_dir)
    elif "pi0_fast_droid___finetune_on_" in config.name:
        dataset = Pi0FastDroidFinetuneDataset(config.model, config.finetuning_collected_demos_dir)
    else:
        dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                if "ricl" in config.name:
                    yield _model.RiclObservation.from_dict(batch, config.model.num_retrieved_observations), batch["query_actions"]
                else:
                    yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
