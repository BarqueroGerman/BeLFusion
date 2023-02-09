import os
import argparse
import tarfile
from io import BytesIO

import numpy as np
import torch
import zarr
from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # https://github.com/mmatl/pyrender/issues/13
TARGET_OPEN_GL_MAJOR = 3
TARGET_OPEN_GL_MINOR = 3



def process_data(path, out, target_fps):
    z_poses = zarr.open(os.path.join(out, 'poses.zarr'), mode='w', shape=(0, 22, 3), chunks=(1000, 22, 3), dtype=np.float32)
    z_trans = zarr.open(os.path.join(out, 'trans.zarr'), mode='w', shape=(0, 3), chunks=(1000, 3), dtype=np.float32)
    z_index = zarr.open(os.path.join(out, 'poses_index.zarr'), mode='w', shape=(0, 2), chunks=(1000, 2), dtype=int)
    i = 0
    tar = tarfile.open(path, 'r')
    for member in tqdm(tar):
        file_name = os.path.basename(member.name)
        if file_name.endswith('.npz') and not file_name.startswith('.'):
            try:
                with tar.extractfile(member) as f:
                    array_file = BytesIO()
                    array_file.write(f.read())
                    array_file.seek(0)
                    bdata = np.load(array_file)

                    if 'mocap_framerate' not in bdata and 'mocap_frame_rate' not in bdata:
                        # SOMA -> 'SOMA/soma_subject1/male_stagei.npz' and 'SOMA/soma_subject2/male_stagei.npz' have no mocap_frame_rate
                        # GRAB -> 
                        # 'GRAB/s4/female_stagei.npz', 
                        # 'GRAB/s9/male_stagei.npz', 
                        # 'GRAB/s10/male_stagei.npz',
                        # 'GRAB/s3/female_stagei.npz',
                        # 'GRAB/s2/male_stagei.npz',
                        # 'GRAB/s6/female_stagei.npz',
                        # 'GRAB/s8/male_stagei.npz',
                        # 'GRAB/s7/female_stagei.npz' 
                        # 'GRAB/s5/female_stagei.npz' 
                        # 'GRAB/s1/male_stagei.npz' 
                        # have no mocap_frame_rate --> we skip those files
                        print(f"WARNING: we skip '{member.name}' because it is corrupted (no framerate)")
                        continue
                    else:
                        frame_rate = bdata['mocap_framerate'] if 'mocap_framerate' in bdata else bdata['mocap_frame_rate']
                    gender = str(bdata["gender"])
                    if gender == "b'female'":
                        gender = "female" # this is a common problem in SSM dataset

                    if target_fps == -1:
                        fps = frame_rate
                    else:
                        fps = target_fps

                    #if not frame_rate % target_fps == 0.:
                    #    print(f"Warning: FPS does not match for dataset {path}")
                    frame_multiplier = int(np.round(frame_rate / fps))

                    time_length = len(bdata['trans'])
                    body_parms = {
                        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
                        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
                        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
                        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
                        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
                        #'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
                    }

                    body_pose_hand = bm[gender](**{k:v for k,v in body_parms.items() if k in [
                        'pose_body', 'betas', 'pose_hand', 'root_orient', 'trans'
                        ]})
                    
                    body_joints = c2c(body_pose_hand.Jtr)[:, :22].copy()[::frame_multiplier]
                    body_trans = bdata['trans'][::frame_multiplier]
                    
                    z_poses.append(body_joints, axis=0)
                    z_trans.append(body_trans, axis=0)
                    z_index.append(np.array([[i, i + body_joints.shape[0]]]), axis=0)
                    #print(frame_multiplier, np.array([[i, i + body_pose.shape[0]]]), body_pose.shape, body_trans.shape)
                    i = i + body_joints.shape[0]
            except Exception as e:
                print(e, ". Filename:", file_name)
    """
    z_poses = zarr.open(os.path.join(out, 'poses.zarr'), mode='r')
    z_trans = zarr.open(os.path.join(out, 'trans.zarr'), mode='r')
    z_index = zarr.open(os.path.join(out, 'poses_index.zarr'), mode='r')
    print(z_poses.shape, z_trans.shape, z_index.shape)
    """

parser = argparse.ArgumentParser(description='AMASS Process Raw Data')

parser.add_argument('--fps',
                    type=int,
                    default=60,
                    help='FPS')

parser.add_argument('--datasets',
                    type=str,
                    nargs="+",
                    help='The names of the datasets to process',
                    default=None)

parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')

args = parser.parse_args()


amass_official_splits = {
        'validation': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap']#ACCAD
    }


# python -m data_loader.parsers.amass --gpu
# it will pre-process the AMASS dataset
if __name__ == '__main__':
    fps = args.fps
    datasets = args.datasets

    in_path = "./datasets/AMASS"
    out_path = "./auxiliar/datasets/AMASS"

    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
    models_dir = './auxiliar'

    # initialize all needed resources
    genders = "male", "female"
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    bm, faces = {}, {}
    for gender in genders:
        bm_fname = os.path.join(models_dir, 'body_models/smplh/{}/model.npz'.format(gender))
        dmpl_fname = os.path.join(models_dir, 'body_models/dmpls/{}/model.npz'.format(gender))

        bm[gender] = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
        faces[gender] = c2c(bm[gender].f)

    imw, imh = 1600, 1600
    print("All resources initialized")

    if datasets is None:
        # we process all datasets in 'in_path' folder
        datasets = sorted([p.split(".")[0] for p in os.listdir(in_path)])

    # list all datasets
    print("Datasets to process:")
    print(datasets)
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Processing {dataset}...")
        process_data(os.path.join(in_path, dataset + '.tar.bz2'), os.path.join(out_path, dataset), fps)
