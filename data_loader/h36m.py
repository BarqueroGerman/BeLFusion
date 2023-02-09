import numpy as np
import os
from utils.skeleton import SkeletonH36M
from base import BaseDataLoader, BaseMultiAgentDataset
import pandas as pd
import hashlib
import torch
from scipy.spatial.transform import Rotation as R
from models import ClassifierForFID
import os
from glob import glob
import cdflib
from tqdm import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class H36MDataset(BaseMultiAgentDataset):
    def __init__(self, annotations_folder, subjects, actions, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', 
                da_mirroring=0.0, da_rotations=0.0): # data augmentation strategies

        assert (subjects is not None and actions is not None) or segments_path is not None
        self.annotations_folder = annotations_folder
        self.segments_path = segments_path
        self.subjects, self.actions = subjects, actions
        self.use_vel = use_vel 
        self.drop_root = drop_root # for comparison against DLow/Smooth4Diverse
        self.dict_indices = {} # dict_indices[subject][action] indicated idx where subject-action annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 1 # 0: subject, 1: action --> action is the class used for metrics computation
        self.idx_to_class = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        self.mean_motion_per_class = [0.004533339312024582, 0.005071772030221925, 0.003968115058494981, 0.00592599384929542, 0.003590651675618232, 0.004194935839372698, 0.005625120976387903, 0.0024796492124910586, 0.0035406092427418797, 0.003602172245980421, 0.004347639393585013, 0.004222595821256223, 0.007537553520400006, 0.007066049169369122, 0.006754175094952483]

        assert da_mirroring >= 0.0 and da_mirroring <= 1.0 and da_rotations >= 0.0 and da_rotations <= 1.0, \
            "Data augmentation strategies must be in [0, 1]"
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations
        super().__init__(precomputed_folder, obs_length, pred_length, augmentation=augmentation, stride=stride, normalize_data=normalize_data,
                            normalize_type=normalize_type, dtype=dtype)

    def get_classifier(self, device):
        classifier_for_fid = ClassifierForFID(input_size=48, hidden_size=128, hidden_layer=2,
                                                    output_size=15, device=device, use_noise=None).to(device)
                                                    
        classifier_path = os.path.join("./auxiliar", "h36m_classifier.pth")
        classifier_state = torch.load(classifier_path, map_location=device)
        classifier_for_fid.load_state_dict(classifier_state["model"])
        classifier_for_fid.eval()
        return classifier_for_fid
    
    def _get_hash_str(self, use_all=False):
        use_all = [str(self.obs_length), str(self.pred_length), str(self.stride), str(self.augmentation)] if use_all else []
        to_hash = "".join(tuple(self.subjects + list(self.actions) + 
                [str(self.drop_root), str(self.use_vel)] + use_all))
        return str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())

            

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.subjects, self.actions)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _init_skeleton(self):
        """
        
        """
        self.skeleton = SkeletonH36M(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8

    def _read_all_annotations(self, subjects, actions):
        preprocessed_path = os.path.join(self.precomputed_folder, 'data_3d_h36m.npz')
        if not os.path.exists(preprocessed_path):
            # call function that preprocesses dataset from dataset folder
            preprocess_dataset(self.annotations_folder, output_path=preprocessed_path) # borrowed from VideoPose3D repository

        # we load from already preprocessed dataset
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in subjects, data_o.items()))
        if actions != 'all': # if not all, we only keep the data from the selected actions, for each participant
            for subject in list(data_f.keys()):
                #data_f[key] = dict(filter(lambda x: all([a in x[0] for a in actions]), data_f[key].items())) # OLD and wrong
                data_f[subject] = dict(filter(lambda x: any([a in x[0] for a in actions]), data_f[subject].items()))
                if len(data_f[subject]) == 0: # no actions for subject => delete
                    data_f.pop(subject)
                    print(f"Subject '{subject}' has no actions available from '{actions}'.")
        else:
            print(f"All actions loaded from {subjects}.")

        # we build the feature vectors for each participant and action
        for subject in data_f.keys():
            for action in data_f[subject].keys():
                seq = data_f[subject][action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1] # we make them root-relative (root-> joint at first position)
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1) # shape -> 17+1 (vel only from root joint)
                data_f[subject][action] = seq
        self.data = data_f


        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0
        for subject in self.data:
            self.dict_indices[subject] = {}

            for action in self.data[subject]:
                self.dict_indices[subject][action] = counter
                self.clip_idx_to_metadata.append((subject, action.split(" ")[0]))
                counter += 1

                anns_all.append(self.data[subject][action][:, None].astype(self.dtype)) # participants axis expanded
        
        self._generate_statistics_full(anns_all)

        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        subjects, actions = list(df["subject"].unique()), list(df["action"].unique())
        self.annotations = self._read_all_annotations(subjects, actions)
        
        segments = [(self.dict_indices[row["subject"]][row["action"]], 
                    row["pred_init"] - self.obs_length, 
                    row["pred_init"] + self.pred_length - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["subject"], row["action"].split(" ")[0]) for i, row in df.iterrows()]
                        
        #print(segments)
        #print(self.dict_indices)
        return segments, segment_idx_to_metadata

    def get_custom_segment(self, subject, action, frame_num):
        counter = self.dict_indices[subject][action]
        obs, pred = self._get_segment(counter, frame_num, frame_num + self.seg_length - 1)
        return obs, pred

    def recover_landmarks(self, data, rrr=True, fill_root=False):
        if self.normalize_data:
            data = self.denormalize(data)
        # data := (BatchSize, SegmentLength, NumPeople, Landmarks, Dimensions)
        # or data := (BatchSize, NumSamples, DiffusionSteps, SegmentLength, NumPeople, Landmarks, Dimensions)
        # the idea is that it does not matter how many dimensions are before NumPeople, Landmarks, Dimension => always working right
        if rrr:
            assert data.shape[-2] == 17 or (data.shape[-2] == 16 and fill_root), "Root was dropped, so original landmarks can not be recovered"
            if data.shape[-2] == 16 and fill_root:
                # we fill with a 'zero' imaginary root
                size = list(data.shape[:-2]) + [1, data.shape[-1]] # (BatchSize, SegmentLength, NumPeople, 1, Dimensions)
                return np.concatenate((np.zeros(size), data), axis=-2) # same, plus 0 in the root position
            data[..., 1:, :] += data[..., :1, :]
        return data

    def denormalize(self, x):
        if self.drop_root:
            if x.shape[-2] == 16:
                return super().denormalize(x, idces=list(range(1, 17)))
            elif x.shape[-2] == 17:
                return super().denormalize(x, idces=list(range(17)))
            else:
                raise Exception("'x' can't have shape != 16 or 17")
        return super().denormalize(x)

    def __getitem__(self, idx):
        obs, pred, extra = super(H36MDataset, self).__getitem__(idx)
        
        if self.drop_root:
            obs, pred = obs[..., 1:, :], pred[..., 1:, :]

        mm_gt = -1
        if self.mm_indces is not None:
            # we need to find the closest (may not be computed for all indices)
            candidates = [int(v) for v in self.mm_indces[str(extra["clip_idx"])].keys()]
            nearest_idx = find_nearest(candidates, extra["init"] + self.obs_length)
            mm_gt_idces = self.mm_indces[str(extra["clip_idx"])][str(nearest_idx)] # mm_gt is indexed with first frame of prediction
            # we get all prediction multimodal GT sequences
            mm_gts = [super(H36MDataset, self)._get_segment(c_i, c_init - self.obs_length, c_init + self.pred_length - 1)[1][None, ...] for c_i, c_init, score in mm_gt_idces]
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


class H36MDataLoader(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, subjects=None, actions=None, 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0):
                    
        self.dataset = H36MDataset(annotations_folder, subjects, actions, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track)


class H36MDataset_Auto(H36MDataset):
    def __init__(self, annotations_folder, subjects, actions, 
                precomputed_folder, obs_length, pred_length, use_vel=False,
                stride=1, augmentation=0, segments_path=None, normalize_data=True, normalize_type='standardize',
                drop_root=False, dtype='float64', da_mirroring=0.0, da_rotations=0.0):
        assert pred_length == obs_length, "Autoencoder requires obs_length == pred_length"
        super().__init__(annotations_folder, subjects, actions, 
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


class H36MDataLoader_Auto(BaseDataLoader):
    def __init__(self, batch_size, annotations_folder, precomputed_folder, obs_length, pred_length, validation_split=0.0, subjects=None, actions=None, 
                    stride=1, shuffle=True, num_workers=1, num_workers_dataset=1, augmentation=0, segments_path=None, use_vel=False, seed=0,
                    normalize_data=True, normalize_type='standardize', drop_root=False, drop_last=True, dtype='float64', samples_to_track=None,
                    da_mirroring=0.0, da_rotations=0.0):
                    
        self.dataset = H36MDataset_Auto(annotations_folder, subjects, actions, precomputed_folder, obs_length, pred_length, 
                                            stride=stride, augmentation=augmentation, segments_path=segments_path, use_vel=use_vel,
                                            normalize_data=normalize_data, normalize_type=normalize_type, drop_root=drop_root, dtype=dtype,
                                            da_mirroring=da_mirroring, da_rotations=da_rotations)
    
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, seed, drop_last=drop_last, samples_to_track=samples_to_track)





OUTPUT_3D = 'data_3d_h36m'
SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

def preprocess_dataset(dataset_folder, output_path=OUTPUT_3D, subjects=SUBJECTS):
    
    if os.path.exists(output_path):
        print('The dataset already exists at', output_path)
        exit(0)
        
    print('Converting original Human3.6M dataset from', dataset_folder, '(CDF files)')
    output = {}
    
    for subject in tqdm(subjects):
        output[subject] = {}
        file_list = glob(os.path.join(dataset_folder, f'Poses_D3_Positions_{subject}', subject, 'MyPoseFeatures', 'D3_Positions', '*.cdf'))
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == 'S11' and action == 'Directions':
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                    .replace('WalkingDog', 'WalkDog')
            
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')
    
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)
    print('Done.')



# python -m data_loader.h36m
if __name__ == '__main__':
    np.random.seed(0)
    
    training = ["S1", "S5", "S6", "S7", "S8"]
    test = ["S9", "S11"]
    actions = "all" 

    batch_size = 128
    annotations_folder = "./datasets/Human36M/"
    precomputed_folder = "./auxiliar/datasets/Human36M"
    obs_length = 25
    pred_length = 100
    stride = 10
    augmentation = 0

    print("="*50)
    print("Computing values for CMD.")
    # below to compute the average motion per class for the TEST set, for the CMD computation
    data_loader = H36MDataLoader(batch_size, annotations_folder, precomputed_folder, 
                obs_length, pred_length, drop_root=True, 
                subjects=test, actions=actions, drop_last=False,
                stride=stride, shuffle=True, augmentation=0, normalize_data=False,
                dtype="float32")

    counter = 0
    average_3d = 0
    CLASS_TO_IDX = data_loader.dataset.class_to_idx
    class_average_3d = {k: 0 for k in CLASS_TO_IDX}
    class_counter = {k: 0 for k in CLASS_TO_IDX}
    for batch_idx, batch in enumerate(data_loader):
        data, target, extra = batch
        classes = np.array([CLASS_TO_IDX[c] for c in extra["metadata"][data_loader.dataset.metadata_class_idx]])

        target_3d = target.reshape(-1, 100, 16, 3)
        
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

    print(f"Mean motion values for {len(list_of_motions_3d)} classes (subjects={data_loader.dataset.subjects}):\n", [float(l) for l in list_of_motions_3d])
    #print(frequencies)


