import numpy as np
import os
from utils.skeleton import SkeletonAMASS
from base import BaseDataLoader, BaseMultiAgentDataset
import pandas as pd
import hashlib
from scipy.spatial.transform import Rotation as R
import zarr

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class AMASSDataset(BaseMultiAgentDataset):
    def __init__(self, annotations_folder, datasets, file_idces, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', 
                da_mirroring=0.0, da_rotations=0.0): # data augmentation strategies

        assert (datasets is not None and file_idces is not None) or segments_path is not None
        self.annotations_folder = annotations_folder
        self.segments_path = segments_path
        self.datasets, self.file_idces = datasets, file_idces
        assert self.file_idces == "all", "We only support 'all' files for now"
        self.use_vel = use_vel 
        self.drop_root = drop_root # for comparison against DLow/Smooth4Diverse
        self.dict_indices = {} # dict_indices[dataset][file_idx] indicated idx where dataset-file_idx annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 0 # 0: dataset, 1: filename --> dataset is the class used for metrics computation
        self.idx_to_class = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        self.mean_motion_per_class = [0.004860274970204714, 0.00815901767307159, 0.001774023530090276, 0.004391708416532331, 0.007596136106898701, 0.00575787090703614, 0.008530069935655568]

        assert da_mirroring >= 0.0 and da_mirroring <= 1.0 and da_rotations >= 0.0 and da_rotations <= 1.0, \
            "Data augmentation strategies must be in [0, 1]"
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations
        super().__init__(precomputed_folder, obs_length, pred_length, augmentation=augmentation, stride=stride, normalize_data=normalize_data,
                            normalize_type=normalize_type, dtype=dtype)

    def get_classifier(self, device):
        raise NotImplementedError("We don't have a classifier for the AMASS dataset")
    
    def _get_hash_str(self, use_all=False):
        use_all = [str(self.obs_length), str(self.pred_length), str(self.stride), str(self.augmentation)] if use_all else []
        to_hash = "".join(tuple(self.datasets + list(self.file_idces) + 
                [str(self.drop_root), str(self.use_vel)] + use_all))
        return str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())
            

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.datasets, self.file_idces)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _init_skeleton(self):
        # full list -> https://github.com/vchoutas/smplx/blob/43561ecabd23cfa70ce7b724cb831b6af0133e6e/smplx/joint_names.py#L166
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        left, right = [1, 4, 7, 10, 13, 16, 18, 20], [2, 5, 8, 11, 14, 17, 19, 21]
        self.skeleton = SkeletonAMASS(parents=parents,
                                 joints_left=left,
                                 joints_right=right,)
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(22) if x not in self.removed_joints]) # 22
        self.skeleton.remove_joints(self.removed_joints)

    def _read_all_annotations(self, datasets, file_idces):
        if not os.path.exists(self.precomputed_folder):
            raise NotImplementedError("Preprocessing of AMASS dataset is not implemented yet. Please use the preprocessed data.")


        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0

        print("Loading datasets: ", datasets, file_idces)
        for dataset in datasets:
            self.dict_indices[dataset] = {}

            #print("Loading dataset: ", dataset)
            #print(os.path.join(self.precomputed_folder, dataset))
            z_poses = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses.zarr'), mode='r')
            z_trans = zarr.open(os.path.join(self.precomputed_folder, dataset, 'trans.zarr'), mode='r')
            z_index = zarr.open(os.path.join(self.precomputed_folder, dataset, 'poses_index.zarr'), mode='r')

            # we build the feature vectors for each dataset and file_idx
            #print(z_poses.shape, z_trans.shape, z_index.shape, z_index[-1])
            for file_idx in range(z_index.shape[0]):
                self.dict_indices[dataset][file_idx] = counter

                i0, i = z_index[file_idx]

                seq = z_poses[i0:i]
                #seq = np.concatenate([z_trans[i0:i][:, None], seq], axis=1) # add the root to the sequence
                seq[:, 1:] -= seq[:, :1] # we make them root-relative (root-> joint at first position)
                seq = seq[:, self.kept_joints, :]

                self.dict_indices[dataset][file_idx] = counter
                self.clip_idx_to_metadata.append((dataset, file_idx))
                counter += 1

                anns_all.append(seq[:, None].astype(self.dtype)) # datasets axis expanded

        self._generate_statistics_full(anns_all)
        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        # columns -> dataset,file,file_idx,pred_init,pred_end
        datasets, file_idces = list(df["dataset"].unique()), list(df["file_idx"].unique())
        self.annotations = self._read_all_annotations(datasets, "all")#file_idces)
        
        segments = [(self.dict_indices[row["dataset"]][row["file_idx"]], 
                    row["pred_init"] - self.obs_length, 
                    row["pred_init"] + self.pred_length - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["dataset"], row["file_idx"]) for i, row in df.iterrows()]
                        
        #print(segments)
        #print(self.dict_indices)
        return segments, segment_idx_to_metadata

    def get_custom_segment(self, dataset, file_idx, frame_num):
        counter = self.dict_indices[dataset][file_idx]
        obs, pred = self._get_segment(counter, frame_num, frame_num + self.seg_length - 1)
        return obs, pred

    def recover_landmarks(self, data, rrr=True, fill_root=False):
        if self.normalize_data:
            data = self.denormalize(data)
        # data := (BatchSize, SegmentLength, NumPeople, Landmarks, Dimensions)
        # or data := (BatchSize, NumSamples, DiffusionSteps, SegmentLength, NumPeople, Landmarks, Dimensions)
        # the idea is that it does not matter how many dimensions are before NumPeople, Landmarks, Dimension => always working right
        if rrr:
            assert data.shape[-2] == len(self.kept_joints) or (data.shape[-2] == len(self.kept_joints)-1 and fill_root), "Root was dropped, so original landmarks can not be recovered"
            if data.shape[-2] == len(self.kept_joints)-1 and fill_root:
                # we fill with a 'zero' imaginary root
                size = list(data.shape[:-2]) + [1, data.shape[-1]] # (BatchSize, SegmentLength, NumPeople, 1, Dimensions)
                return np.concatenate((np.zeros(size), data), axis=-2) # same, plus 0 in the root position
            data[..., 1:, :] += data[..., :1, :]
        return data

    def denormalize(self, x):
        if self.drop_root:
            if x.shape[-2] == len(self.kept_joints)-1:
                return super().denormalize(x, idces=list(range(1, len(self.kept_joints))))
            elif x.shape[-2] == len(self.kept_joints):
                return super().denormalize(x, idces=list(range(len(self.kept_joints))))
            else:
                raise Exception(f"'x' can't have shape != {len(self.kept_joints)-1} or {len(self.kept_joints)}")
        return super().denormalize(x)

    def __getitem__(self, idx):
        obs, pred, extra = super(AMASSDataset, self).__getitem__(idx)
        
        if self.drop_root:
            obs, pred = obs[..., 1:, :], pred[..., 1:, :]

        mm_gt = -1
        if self.mm_indces is not None:
            # we need to find the closest (may not be computed for all indices)
            candidates = [int(v) for v in self.mm_indces[str(extra["clip_idx"])].keys()]
            nearest_idx = find_nearest(candidates, extra["init"] + self.obs_length)
            mm_gt_idces = self.mm_indces[str(extra["clip_idx"])][str(nearest_idx)] # mm_gt is indexed with first frame of prediction
            # we get all prediction multimodal GT sequences
            mm_gts = [super(AMASSDataset, self)._get_segment(c_i, c_init - self.obs_length, c_init + self.pred_length - 1)[1][None, ...] for c_i, c_init, score in mm_gt_idces]
            mm_gt = np.concatenate(mm_gts, axis=0)
            mm_gt = mm_gt[..., 1:, :] if self.drop_root else mm_gt

        if self.da_mirroring != 0:
            # apply mirroring with probability 0.5
            mirroring_idces = [0, 1] # 2 is not used because the person would be upside down
            for m in mirroring_idces:
                if np.random.rand() < self.da_mirroring:
                    # make a copy of obs, pred
                    obs, pred = obs.copy(), pred.copy()
                    # invert sign of first coordinate at last axis for obs, pred
                    obs[..., m] *= -1
                    pred[..., m] *= -1

                    if mm_gt != -1:
                        mm_gt = mm_gt.copy()
                        mm_gt[..., m] *= -1
        
        extra["non_rotated_obs"] = obs.copy()
        extra["non_rotated_pred"] = pred.copy()
        if self.da_rotations != 0:
            # apply random rotations with probability 1
            rotation_axes = ['z'] # 'x' and 'y' not used because the person could be upside down
            for a in rotation_axes:
                if np.random.rand() < self.da_rotations:
                    degrees = np.random.randint(0, 360)
                    r = R.from_euler(a, degrees, degrees=True).as_matrix().astype(np.float32)
                    obs = (r @ obs.reshape((-1, 3)).T).T.reshape(obs.shape)
                    pred = (r @ pred.reshape((-1, 3)).T).T.reshape(pred.shape)

                    if mm_gt != -1:
                        mm_gt = (r @ mm_gt.reshape((-1, 3)).T).T.reshape(mm_gt.shape)
        
        extra["mm_gt"] = mm_gt
        #print("metadata_length", len(self.segment_idx_to_metadata), "segments_length", len(self.segments))
        return obs, pred, extra


class AMASSDataLoader(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, datasets=None, file_idces='all', 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0):
                    
        self.dataset = AMASSDataset(annotations_folder, datasets, file_idces, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track)


class AMASSDataset_Auto(AMASSDataset):
    def __init__(self, annotations_folder, datasets, file_idces, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', da_mirroring=0.0, da_rotations=0.0):
        assert pred_length == obs_length, "Autoencoder requires obs_length == pred_length"
        super().__init__(annotations_folder, datasets, file_idces, 
                precomputed_folder, obs_length, pred_length, use_vel=use_vel,
                stride=stride, augmentation=augmentation, segments_path=segments_path, 
                normalize_data=normalize_data, normalize_type=normalize_type,
                drop_root=drop_root, dtype=dtype,
                da_mirroring=da_mirroring, da_rotations=da_rotations)
        self.seg_length = obs_length

    def __getitem__(self, idx):
        obs, pred, extra = super().__getitem__(idx)
        extra["end"] = extra["init"] + self.obs_length
        extra["pred"] = pred
        return obs, obs, extra


class AMASSDataLoader_Auto(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, datasets=None, file_idces='all', 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0):
                    
        self.dataset = AMASSDataset_Auto(annotations_folder, datasets, file_idces, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track)


# python -m data_loader.h36m
if __name__ == '__main__':
    # run this script so that mean/var are precomputed from all training samples
    np.random.seed(0)
    
    training = ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU", "EKUT", "EyesJapanDataset", "KIT", "PosePrior", "TCDHands", "TotalCapture"]
    validation = [ "HumanEva", "HDM05", "SFU", "MoSh" ]
    test = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
    file_idces = "all"

    batch_size = 128
    annotations_folder = "./auxiliar/datasets/AMASS"
    precomputed_folder = "./auxiliar/datasets/AMASS"
    obs_length = 30
    pred_length = 120
    stride = 10
    augmentation = 0

    """
    for datasets in ["training", "validation", "test"]:
        # count number of samples
        stride = 150
        data_loader = AMASSDataLoader(batch_size, annotations_folder, precomputed_folder, 
                    obs_length, pred_length, drop_root=True,
                    datasets=eval(datasets), file_idces=file_idces, drop_last=False,
                    stride=stride, shuffle=False, augmentation=0, normalize_data=False,
                    dtype="float32")
        print("Number of samples in {}: {}".format(datasets, len(data_loader.dataset)))
    """

    data_loader = AMASSDataLoader(batch_size, annotations_folder, precomputed_folder, 
                obs_length, pred_length, drop_root=True, 
                datasets=test, file_idces=file_idces, drop_last=False,
                stride=stride, shuffle=False, augmentation=0, normalize_data=False,
                dtype="float32")

    print("="*50)
    print("Now computing values for CMD.")
    # below to compute the average motion per class for the TEST set, for the CMD computation

    counter = 0
    average_3d = 0
    CLASS_TO_IDX = data_loader.dataset.class_to_idx
    class_average_3d = {k: 0 for k in CLASS_TO_IDX}
    class_counter = {k: 0 for k in CLASS_TO_IDX}
    for batch_idx, batch in enumerate(data_loader):
        data, target, extra = batch
        classes = np.array([CLASS_TO_IDX[c] for c in extra["metadata"][data_loader.dataset.metadata_class_idx]])

        target_3d = target.reshape(-1, 120, 21, 3)
        
        average_3d += (np.linalg.norm(target_3d[:, 1:] - target_3d[:, :-1], axis=-1)).mean()
        counter += 1
        for class_label in CLASS_TO_IDX:
            c = CLASS_TO_IDX[class_label]
            class_mask = classes == c
            target_class_3d = target_3d[class_mask]
            class_average_3d[class_label] += (np.linalg.norm(target_class_3d[:, 1:] - target_class_3d[:, :-1], axis=-1)).mean(axis=-1).mean(axis=-1).sum()
            class_counter[class_label] += target_class_3d.shape[0]

    #print(f"AVERAGE={average_3d/counter:.8f}")

    total_class_counter = sum(class_counter.values())
    list_of_motions_3d = []
    frequencies = []
    for c in class_average_3d:
        #print(f"{c}: {class_average_3d[c]/class_counter[c]:.8f}")
        list_of_motions_3d.append(float((class_average_3d[c]/class_counter[c])))
        frequencies.append(class_counter[c]/total_class_counter)

    print(f"Mean motion values for {len(list_of_motions_3d)} classes (datasets={data_loader.dataset.datasets}):\n", [float(l) for l in list_of_motions_3d])
    #print(frequencies)


