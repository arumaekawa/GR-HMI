import os
import warnings

import faiss
import torch


class HippocampusBase(object):
    def __init__(self, feature_dim, fp32=False, max_memory_size=None):
        self.memory = {}
        self.feature_dim = feature_dim
        self.fp32 = fp32
        self.max_memory_size = max_memory_size

    def reset_new_task(self, task):
        self.current_task = task
        if self.fp32:
            self.memory[task] = torch.empty(0, self.feature_dim)
        else:
            self.memory[task] = torch.empty(0, self.feature_dim).half()

    def memorize(self, new_features):
        self.memory[self.current_task] = torch.cat(
            (self.memory[self.current_task], new_features.detach().cpu())
        )

    def prune_memory(self):
        if self.max_memory_size is not None:
            # retain the same number of features for each task
            single_task_memory_size = self.max_memory_size // len(self.memory)
            for task in self.memory.keys():
                if self.memory_size[task] > single_task_memory_size:
                    indices = self.select_keep_indices(task, single_task_memory_size)
                    self.memory[task] = self.memory[task][indices]

    def retrieve(self, task, num):
        indices = self.select_retrieve_indices(task, num)
        return self.memory[task][indices]

    def select_keep_indices(self, task, num):
        raise NotImplementedError

    def select_retrieve_indices(self, task, num):
        raise NotImplementedError

    def save_pretrained(self, save_dir_path):
        # save feature vectors
        torch.save(self.memory, os.path.join(save_dir_path, "features"))

    def from_pretrained(self, save_dir_path):
        # load feature vectors
        self.memory = torch.load(os.path.join(save_dir_path, "features"))

    @property
    def memory_size(self):
        return {task: len(fea) for task, fea in self.memory.items()}

    @property
    def memory_byte_size(self):
        memory_size = {}
        for task, feature in self.memory.items():
            memory_size[task] = feature.element_size() * feature.nelement()
        return memory_size


class HippocampusRandom(HippocampusBase):
    """
    store: Random
    retrieve: Random
    quantize: No
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


class HippocampusNearestKmeans(HippocampusBase):
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

        # k-means clustering
        k_means = faiss.Kmeans(d=self.feature_dim, k=num)
        k_means.train(self.memory[task].numpy())

        # nearest neighbor search
        _, nearest_kmeans_indices = faiss.knn(
            k_means.centroids, self.memory[task].numpy(), 1
        )
        return nearest_kmeans_indices.squeeze()

    def select_retrieve_indices(self, task, num):
        return self.get_nearest_kmeans_indices(task, num)

    def select_keep_indices(self, task, num):
        return self.get_nearest_kmeans_indices(task, num)


class HippocampusLossDifference(HippocampusBase):
    """
    store: Loss difference between before and after training
    retrieve: Loss difference between before and after training
    quantize: No
    """

    def __init__(self, feature_dim, fp32=False, max_memory_size=None):
        super().__init__(feature_dim, fp32=fp32, max_memory_size=max_memory_size)
        self.memory_loss = {}
        self.memory_is_sorted = {}

    def reset_new_task(self, task):
        super().reset_new_task(task)
        self.memory_loss[task] = {"before": None, "after": None}
        self.memory_is_sorted[task] = False

    def add_before_loss(self, before_losses):
        assert len(self.memory[self.current_task]) == len(before_losses)
        assert len(self.memory_loss[self.current_task]["before"]) == 0

        self.memory_loss[self.current_task]["before"] = torch.tensor(before_losses)

    def add_after_loss(self, after_losses):
        assert len(self.memory[self.current_task]) == len(after_losses)
        assert len(self.memory_loss[self.current_task]["after"]) == 0
        assert len(self.memory_loss[self.current_task]["before"]) == len(after_losses)

        self.memory_loss[self.current_task]["after"] = torch.tensor(after_losses)

    def sort_memory_by_loss_difference(self, task):
        assert not self.memory_is_sorted[task]
        sorted_indices = torch.argsort(
            self.memory_loss[self.current_task]["before"]
            - self.memory_loss[self.current_task]["after"]
        )
        self.memory[task] = self.memory[task][sorted_indices]
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


class HippocampusLowPerplexity(HippocampusBase):
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
