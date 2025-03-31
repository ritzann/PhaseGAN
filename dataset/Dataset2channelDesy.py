import h5py
import torch
from torch.utils import data
import numpy as np

def find_valid_indices(file_path):
    """Find indices of valid data points (no NaN or inf values)."""
    valid_indices = []
    with h5py.File(file_path, 'r') as h5_file:
        holograms = h5_file['images/hologram'][()]
        phantoms = h5_file['images/phantom'][()]

        for idx in range(len(holograms)):
            hologram = holograms[idx]
            phantom = phantoms[idx]

            # Check for NaN or inf values in either hologram or phantom
            if not (np.isnan(hologram).any() or np.isinf(hologram).any() or
                    np.isnan(phantom).any() or np.isinf(phantom).any()):
                valid_indices.append(idx)

    return valid_indices

class Dataset2channel(data.Dataset):
    """Dataloader for h5 files. It is based on hdf5 dataset by B. Holländer"""
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.file_path = file_path

        # Precompute valid indices
        self.valid_indices = find_valid_indices(file_path)

        # Store the total number of valid indices
        self.num_valid_indices = len(self.valid_indices)

        # Load the dataset to get shapes
        with h5py.File(self.file_path, 'r') as h5_file:
            self.hologram_shape = h5_file['images/hologram'].shape
            self.phantom_shape = h5_file['images/phantom'].shape

    def __len__(self):
        """Return the number of valid indices."""
        return self.num_valid_indices

    def __getitem__(self, index):
        """Get a valid data point using the precomputed valid indices."""
        # Map the index to the corresponding valid index
        valid_index = self.valid_indices[index]

        if 'hologram' not in self.data_cache:
            with h5py.File(self.file_path, 'r') as h5_file:
                self.data_cache['hologram'] = h5_file['images/hologram'][()]
                self.data_cache['phantom'] = h5_file['images/phantom'][()]

        # Load hologram and phantom using the valid index
        hologram = self.data_cache['hologram'][valid_index]
        phantom = self.data_cache['phantom'][valid_index]

        # Convert to PyTorch tensors and permute hologram
        hologram = np.moveaxis(hologram, -1, 0)
        phantom = torch.from_numpy(phantom)

        # Extract phase and absorption from phantom
        phase = phantom[:, :, 0]
        absorption = phantom[:, :, 1]

        # Compute real and imaginary components
        real = np.exp(-absorption) * np.cos(phase)
        imaginary = np.exp(-absorption) * np.sin(phase)

        return hologram, real, imaginary

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = -1
                    if load_data:
                        idx = self._add_to_cache(ds.value, file_path)
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        with h5py.File(file_path,'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    idx = self._add_to_cache(ds[()], file_path)
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)
                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]