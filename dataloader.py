import h5py
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np


class ECGDatasetH5:
    def __init__(self, path, traces_dset='signals', exam_id_dset='exam_id',
                 ids_dset=None, path_to_chagas=None):
        f = h5py.File(path, 'r')
        traces = f[traces_dset]
        exams = f[exam_id_dset]

        if ids_dset is not None:
            ids = [str(i) for i in f[ids_dset]]
        else:
            ids = [str(i) for i in range(traces.shape[0])]
        # save
        self.f = f
        self.traces = traces
        self.exams = exams
        self.ids = ids  # [0,1,2,...]
        self.id_to_idx = OrderedDict(zip(self.get_ids(), range(len(self))))

        # self.in_chagas: boolean stating indices in the data set that has a chagas diagnos
        chagas_df = pd.read_csv(path_to_chagas)
        self.in_chagas = np.isin(exams[:], chagas_df['exam_id'].to_numpy())

        # pick out the diagnoses
        chagas_df = chagas_df.set_index('exam_id')
        chagas_df = chagas_df.reindex(self.exams)
        self.chagas = chagas_df['chagas'].to_numpy(float)  # chagas diagnos -- missing ones are replaced by 'nan'

        # weights due to unbalance
        self.pos_weights = torch.tensor(np.nansum(1-self.chagas)/np.nansum(self.chagas),
            dtype=torch.float32)

    def get_ids(self):
        return self.ids

    def get_weights(self):
        return self.pos_weights

    def getbatch(self, start=0, end=None, attr_only=False):
        if end is None:
            end = len(self)

        # exclude indices that do not have a chagas diagnos
        indices = np.arange(start, end)
        indices = indices[self.in_chagas[start:end]]

        if attr_only:
            return self.chagas[indices]
        else:
            return torch.tensor(self.traces[indices], dtype=torch.float32).transpose(-1, -2), \
                   torch.tensor(self.chagas[indices], dtype=torch.float32).unsqueeze(1)

    def __del__(self):
        self.f.close()

    def __len__(self):
        return len(self.ids)


class ECGDataloaderH5:
    def __init__(self, dset, batch_size, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(dset)
        self.dset = dset
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start = start_idx

    def getfullbatch(self, attr_only=False):
        return self.dset.getbatch(self.start_idx, self.end_idx, attr_only=attr_only)

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        b = self.dset.getbatch(self.start, end)
        self.start = end
        return b

    def __iter__(self):
        self.start = self.start_idx
        return self

    def __len__(self):
        return self.end_idx-self.start_idx


if __name__ == "__main__":
    dset = ECGDatasetH5(
        path='../../data/code15/code15_virtual.hdf5',
        traces_dset='tracings',
        ids_dset=None,
        path_to_chagas='../../data/chagas.csv'
        )
    loader = ECGDataloaderH5(dset, 32)
