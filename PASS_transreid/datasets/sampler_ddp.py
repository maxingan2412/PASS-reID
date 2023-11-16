from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None
import torch
import pickle

def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class RandomIdentitySampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        #self.world_size = dist.get_world_size()
        self.length //= self.world_size

    def __iter__(self):
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        #final_idxs = final_idxs[self.rank * length:(self.rank + 1) * length]
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)


    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.mini_batch_size * self.rank + self.mini_batch_size * i, min(self.mini_batch_size * self.rank + self.mini_batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs


    def sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length

class RandomIdentitySampler_mars(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid). img_path是图片的路径，pid是图片的id，camid是摄像头的id
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    #DataLoader(train_set, batch_size=batchsize,sampler=RandomIdentitySampler(dataset.train, batchsize,4),num_workers=4, collate_fn=train_collate_fn)
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances # bs 32 ni 4  结果是8  所以 一个batch里8个pid，32个tracklets 128张图片
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)  #最后的self.index_dic是一个字典，key是pid，value是这个pid对应的所有图片的index
        self.pids = list(self.index_dic.keys()) # 0-624 list

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list) # 在你提供的代码段中，batch_idxs_dict 是一个使用 defaultdict 创建的字典，其中的默认值是一个空列表。这意味着当你访问 batch_idxs_dict 中不存在的键时，会自动创建一个空列表作为默认值。
        # 最后拿到的 batch_idxs_dict 是一个 有625个key的字典，每个key对应一个list，list的长度都不一样，list list里面还是list 长为4， 其中第一层的list的个数也就是告诉你这个id下有多少个tracklets。
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid]) #copy.deepcopy 是 Python 标准库中的一个函数，用于创建对象的深拷贝。深拷贝是指创建一个新的对象，将原始对象的内容递归复制到新对象中，而不是仅仅复制引用。这意味着新对象与原始对象是独立的，对新对象的修改不会影响原始对象。
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs) # 打乱idxs
            batch_idxs = []
            for idx in idxs:  # 这些代码碎按照 num_instances 的长度 构造出batch_idxs_dict
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        #batch_idxs_dict是一个625的dict 里面key是pid 对应的是4个一组的图片的索引，顺序是打乱的 比如 原来pid0对应的186张图片 原来这个dict对应的就是186张图的编号，现在变成了46组4个张图。46是list
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch) #625的list里面随机选8个
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0) # pop(0): 这是列表的一个方法，用于从列表中弹出（取出）指定索引位置的元素。在这里，参数 0 表示取出列表的第一个元素
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs) # iter() 是一个内置函数，它用于创建一个迭代器对象，可以用于遍历可迭代对象（如列表、元组、字典等）的元素。迭代器对象可以通过 next() 函数逐个返回元素，直到没有元素可遍历为止。

    def __len__(self):
        return self.length