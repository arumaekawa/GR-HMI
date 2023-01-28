import os
import warnings

import faiss
import numpy as np
import torch

from .selective_modules import HippocampusBase


class HippocampusPQBase(HippocampusBase):
    def __init__(
        self,
        feature_dim,
        fp32=False,
        max_memory_size=None,
        n_split=16,
        num_codebook=256,
    ):
        super().__init__(feature_dim, fp32=fp32, max_memory_size=max_memory_size)
        nbits = int(np.log2(num_codebook))
        self.pq = faiss.ProductQuantizer(feature_dim, n_split, nbits)

    def feature_to_code(self, feature):
        return self.pq.compute_codes(feature.numpy().astype("float32"))

    def code_to_feature(self, code):
        dtype = torch.float32 if self.fp32 else torch.float16
        return torch.tensor(self.pq.decode(code), dtype=dtype)

    def reform_feature(self, feature):
        return self.code_to_feature(self.feature_to_code(feature))

    def retrieve(self, task, num):
        code = super().retrieve(task, num)
        return self.code_to_feature(code)

    def quantize_memory(self):
        # decode all features
        for task, memory_index in self.memory.items():
            if not isinstance(memory_index, torch.Tensor):
                self.memory[task] = self.code_to_feature(memory_index)
        # train codebook (faiss-cpu only support float32)
        self.pq.train(torch.cat(list(self.memory.values())).numpy().astype("float32"))
        # quantize features
        self.memory = {
            task: self.feature_to_code(feature) for task, feature in self.memory.items()
        }

    def save_pretrained(self, save_dir_path):
        # save codebook
        codebook = faiss.vector_to_array(self.pq.centroids).astype(
            "float32" if self.fp32 else "float16"
        )
        np.save(os.path.join(save_dir_path, "codebook.npy"), codebook)
        return super().save_pretrained(save_dir_path)

    def from_pretrained(self, save_dir_path):
        # load codebook
        codebook = np.load(os.path.join(save_dir_path, "codebook.npy"))
        faiss.copy_array_to_vector(codebook.astype("float32"), self.pq.centroids)
        return super().from_pretrained(save_dir_path)

    @property
    def memory_byte_size(self):
        memory_byte_size = {}
        for task, memory_index in self.memory.items():
            if isinstance(memory_index, torch.Tensor):
                memory_byte_size[task] = (
                    memory_index.element_size() * memory_index.nelement()
                )
            elif isinstance(memory_index, np.ndarray):
                memory_byte_size[task] = memory_index.nbytes
            else:
                raise AssertionError
        # add codebook size
        memory_byte_size["codebook"] = (
            faiss.vector_to_array(self.pq.centroids)
            .astype("float32" if self.fp32 else "float16")
            .nbytes
        )
        return memory_byte_size


class HippocampusPQRandom(HippocampusPQBase):
    """
    store: Random
    retrieve: Random
    quantize: True
    """

    def select_random(self, task, num, replace=True):
        if replace:
            indices = torch.randint(len(self.memory[task]), (num,))
        else:
            if len(self.memory[task]) < num:
                warnings.warn("required size > memory size, So sample with replacement")
                indices = self.select_retrieve_indices(task, num, replace=True)
            else:
                indices = torch.randperm(len(self.memory[task]))[:num]
        return indices

    def select_retrieve_indices(self, task, num):
        return self.select_random(task, num, replace=True)

    def select_keep_indices(self, task, num):
        return self.select_random(task, num, replace=False)


class HippocampusPQNearestKmeans(HippocampusPQBase):
    """
    store: Nearest neighbor of k-means cluster centers
    retrieve: Nearest neighbor of k-means cluster centers
    quantize: No
    """

    def get_nearest_kmeans_indices(self, task, num):
        if len(self.memory[task]) < num:
            warnings.warn(
                "required size > memory size, So sample randomly with replacement"
            )
            return torch.randint(len(self.memory[task]), (num,))

        # decode memory from code_ids

        memory_as_features = (
            self.code_to_feature(self.memory[task]).numpy().astype("float32")
        )

        # k-means clustering
        k_means = faiss.Kmeans(d=self.feature_dim, k=num)
        k_means.train(memory_as_features)

        # nearest neighbor search
        _, nearest_kmeans_indices = faiss.knn(k_means.centroids, memory_as_features, 1)
        return nearest_kmeans_indices.squeeze()

    def select_retrieve_indices(self, task, num):
        return self.get_nearest_kmeans_indices(task, num)

    def select_keep_indices(self, task, num):
        return self.get_nearest_kmeans_indices(task, num)


class HippocampusPQLossDifference(HippocampusPQBase):
    """
    store: Loss difference between before and after training
    retrieve: Loss difference between before and after training
    quantize: No
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_loss = {}
        self.memory_is_sorted = {}

    def reset_new_task(self, task):
        super().reset_new_task(task)
        self.memory_loss[task] = {"before": None, "after": None}
        self.memory_is_sorted[task] = False

    def add_before_loss(self, before_losses):
        assert len(self.memory[self.current_task]) == len(before_losses)
        assert self.memory_loss[self.current_task]["before"] is None

        self.memory_loss[self.current_task]["before"] = torch.tensor(before_losses)

    def add_after_loss(self, after_losses):
        assert len(self.memory[self.current_task]) == len(after_losses)
        assert self.memory_loss[self.current_task]["after"] is None
        assert len(self.memory_loss[self.current_task]["before"]) == len(after_losses)

        self.memory_loss[self.current_task]["after"] = torch.tensor(after_losses)

    def sort_memory_by_loss_difference(self, task):
        assert not self.memory_is_sorted[task]
        sorted_indices = torch.argsort(
            self.memory_loss[self.current_task]["before"]
            - self.memory_loss[self.current_task]["after"],
            descending=True,
        )
        self.memory[task] = self.memory[task][sorted_indices]
        self.memory_loss[task]["before"] = self.memory_loss[task]["before"][
            sorted_indices
        ]
        self.memory_loss[task]["after"] = self.memory_loss[task]["after"][
            sorted_indices
        ]
        self.memory_is_sorted[task] = True

    def select_keep_indices(self, task, num):
        assert len(self.memory[task]) >= num
        # sort memory by loss difference
        if not self.memory_is_sorted[task]:
            self.sort_memory_by_loss_difference(task)

        return torch.arange(num)

    def select_retrieve_indices(self, task, num):
        if len(self.memory[task]) < num:
            warnings.warn(
                "required size > memory size, So sample randomly with replacement"
            )
            return torch.randint(len(self.memory[task]), (num,))
        # sort memory by loss difference
        if not self.memory_is_sorted[task]:
            self.sort_memory_by_loss_difference(task)

        return torch.arange(num)

    def prune_memory(self):
        if self.max_memory_size is not None:
            # retain the same number of features for each task
            single_task_memory_size = self.max_memory_size // len(self.memory)
            for task in self.memory.keys():
                if self.memory_size[task] > single_task_memory_size:
                    indices = self.select_keep_indices(task, single_task_memory_size)
                    self.memory[task] = self.memory[task][indices]
                    self.memory_loss[task]["before"] = self.memory_loss[task]["before"][
                        indices
                    ]
                    self.memory_loss[task]["after"] = self.memory_loss[task]["after"][
                        indices
                    ]

    def save_pretrained(self, save_dir_path):
        torch.save(self.memory_loss, os.path.join(save_dir_path, "memory_loss"))
        return super().save_pretrained(save_dir_path)

    def from_pretrained(self, save_dir_path):
        self.memory_loss = torch.load(os.path.join(save_dir_path, "memory_loss"))
        return super().from_pretrained(save_dir_path)


class HippocampusPQLowPerplexity(HippocampusPQBase):
    """
    store: Loss difference between before and after training
    retrieve: Loss difference between before and after training
    quantize: No
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_loss = {}
        self.memory_is_sorted = {}

    def reset_new_task(self, task):
        super().reset_new_task(task)
        self.memory_loss[task] = None
        self.memory_is_sorted[task] = False

    def add_gen_loss(self, gen_losses):
        assert len(self.memory[self.current_task]) == len(gen_losses)
        assert self.memory_loss[self.current_task] is None

        self.memory_loss[self.current_task] = torch.tensor(gen_losses)

    def sort_memory_by_loss_difference(self, task):
        assert not self.memory_is_sorted[task]
        sorted_indices = torch.argsort(
            self.memory_loss[self.current_task],
            descending=False,
        )
        self.memory[task] = self.memory[task][sorted_indices]
        self.memory_loss[task] = self.memory_loss[task][sorted_indices]
        self.memory_is_sorted[task] = True

    def select_keep_indices(self, task, num):
        assert len(self.memory[task]) >= num
        # sort memory by loss difference
        if not self.memory_is_sorted[task]:
            self.sort_memory_by_loss_difference(task)

        return torch.arange(num)

    def select_retrieve_indices(self, task, num):
        if len(self.memory[task]) < num:
            warnings.warn(
                "required size > memory size, So sample randomly with replacement"
            )
            return torch.randint(len(self.memory[task]), (num,))
        # sort memory by loss difference
        if not self.memory_is_sorted[task]:
            self.sort_memory_by_loss_difference(task)

        return torch.arange(num)

    def prune_memory(self):
        if self.max_memory_size is not None:
            # retain the same number of features for each task
            single_task_memory_size = self.max_memory_size // len(self.memory)
            for task in self.memory.keys():
                if self.memory_size[task] > single_task_memory_size:
                    indices = self.select_keep_indices(task, single_task_memory_size)
                    self.memory[task] = self.memory[task][indices]
                    self.memory_loss[task] = self.memory_loss[task][indices]

    def save_pretrained(self, save_dir_path):
        torch.save(self.memory_loss, os.path.join(save_dir_path, "memory_loss"))
        return super().save_pretrained(save_dir_path)

    def from_pretrained(self, save_dir_path):
        self.memory_loss = torch.load(os.path.join(save_dir_path, "memory_loss"))
        return super().from_pretrained(save_dir_path)


class HippocampusPQRandomReal(HippocampusPQRandom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_samples = {}

    def reset_new_task(self, task):
        super().reset_new_task(task)
        self.real_samples[task] = []

    def memorize_real_sample(self, new_sample_list):
        self.real_samples[self.current_task] += new_sample_list

    def retrieve(self, task, num):
        indices = self.select_retrieve_indices(task, num)
        real_samples = [self.real_samples[task][id] for id in indices]
        return real_samples

    def prune_memory(self):
        if self.max_memory_size is not None:
            # retain the same number of features for each task
            single_task_memory_size = self.max_memory_size // len(self.memory)
            for task in self.memory.keys():
                if self.memory_size[task] > single_task_memory_size:
                    indices = self.select_keep_indices(task, single_task_memory_size)
                    self.memory[task] = self.memory[task][indices]
                    self.real_samples[task] = [
                        self.real_samples[task][id] for id in indices
                    ]

    def save_pretrained(self, save_dir_path):
        torch.save(self.real_samples, os.path.join(save_dir_path, "real_samples"))
        return super().save_pretrained(save_dir_path)

    def from_pretrained(self, save_dir_path):
        self.real_samples = torch.load(os.path.join(save_dir_path, "real_samples"))
        return super().from_pretrained(save_dir_path)
