
from base import BaseModel
from models.diffusion.gdunet import GDUnet_Latent
import torch
import os
import numpy as np

from utils import read_json
from parse_config import ConfigParser
import models as module_arch


def compute_idces_order(ref, height, width):
    # function to get a 128-long vector which will be used to preprocess the embeddings
    ref = ref.cpu().numpy().reshape((height, width))
    idces = np.array(range(ref.reshape(-1).shape[0])).reshape([height, width])
    for j in range(ref.shape[1]):
        order = np.argsort(ref[:, j])
        idces[:,j] = idces[order, j]
    for i in range(ref.shape[0]):
        order = np.argsort(ref[i])
        idces[i] = idces[i, order]

    # inverse ordering
    idces_inv = np.array(range(ref.reshape(-1).shape[0]))
    idces_inv = idces_inv[np.argsort(idces.reshape(-1))]
    return idces.reshape(-1), idces_inv

class BaseLatentModel(BaseModel):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, 
                embedder_obs_path, embedder_pred_path, emb_size=None,
                emb_preprocessing="none",
                # others
                freeze_obs_encoder=True,
                freeze_pred_decoder=True,
                load_obs_encoder=True,
                ):
        super(BaseLatentModel, self).__init__(n_features, n_landmarks, obs_length, pred_length)
        
        self.emb_preprocessing = emb_preprocessing
        assert emb_size is not None, "emb_size must be specified."
        self.emb_size = emb_size

        def_dtype = torch.get_default_dtype()

        # LOAD EMBEDDER FOR OBSERVATION AND PREDICTION WINDOWS
        self.embedder_obs_path = embedder_obs_path
        self.embedder_pred_path = embedder_pred_path
        models = []
        models_stats = []
        for path, load_checkpoint in zip((embedder_obs_path, embedder_pred_path), (True, load_obs_encoder)):
            configpath = os.path.join(os.path.dirname(path), "config.json")
            assert os.path.exists(path) and os.path.exists(configpath), f"Missing checkpoint/config file for auxiliary model: '{path}'"
            config = read_json(configpath)
            config = ConfigParser(config, save=False)
            model = config.init_obj('arch', module_arch)
            if load_checkpoint:
                checkpoint = torch.load(path, map_location='cpu')
                assert "statistics" in checkpoint or emb_preprocessing.lower() == "none", "Model statistics are not available in its checkpoint. Can't apply embeddings preprocessing."
                stats = checkpoint["statistics"] if "statistics" in checkpoint else None
                models_stats.append(stats)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                models_stats.append(None)

            models.append(model)

        self.embed_obs, self.embed_obs_stats = models[0], models_stats[0]
        self.embed_pred, self.embed_pred_stats = models[1], models_stats[1]

        if freeze_obs_encoder:
            for para in self.embed_obs.parameters():
                para.requires_grad = False
        if freeze_pred_decoder:
            for para in self.embed_pred.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype) # config loader changes this

        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def to(self, device):
        self.embed_obs.to(device)
        self.embed_pred.to(device)
        if self.embed_obs_stats is not None:
            for key in self.embed_obs_stats:
                self.embed_obs_stats[key] = self.embed_obs_stats[key].to(device)
        if self.embed_pred_stats is not None:
            for key in self.embed_pred_stats:
                self.embed_pred_stats[key] = self.embed_pred_stats[key].to(device)
        super().to(device)
        return self

    def preprocess(self, emb, stats, is_prediction=False):
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")


    def undo_preprocess(self, emb, stats, is_prediction=False):
        if stats is None:
            return emb # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")


    def encode_obs(self, obs):
        return self.preprocess(self.embed_obs.encode(obs), self.embed_obs_stats, is_prediction=False)

    def encode_pred(self, pred, obs):
        # we need to encode the last context_length frames of observation
        pred = torch.cat((obs[:, -self.embed_pred.context_length:], pred), dim=1)
        return self.preprocess(self.embed_pred.encode(pred), self.embed_pred_stats, is_prediction=True)

    def decode_pred(self, obs, pred_emb):
        return self.embed_pred.decode(obs, self.undo_preprocess(pred_emb, self.embed_pred_stats, is_prediction=True))

    def get_emb_size(self):
        return self.emb_size

    def forward(self, pred, timesteps, obs):
        raise NotImplementedError("This is an abstract class.")


class LatentUNetMatcher(BaseLatentModel):
    def __init__(self, n_features, n_landmarks, obs_length, pred_length, 
                embedder_obs_path, embedder_pred_path,
                # others
                freeze_obs_encoder=True,
                freeze_pred_decoder=True,
                load_obs_encoder=True,
                emb_preprocessing="none",
                # GDUNet parameters
                emb_height=None, emb_width=None,
                cond_embed_dim=128,
                model_channels=128,
                num_res_blocks=2,
                attention_resolutions=(1, 2, 3, 4),
                dropout=0,
                channel_mult=(1, 2, 4, 8),
                conv_resample=True,
                dims=2,
                use_checkpoint=False,
                dtype="float32",
                num_heads=1,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
                ):
        super(LatentUNetMatcher, self).__init__(n_features, n_landmarks, obs_length, pred_length, 
                embedder_obs_path, embedder_pred_path,
                emb_size=emb_height * emb_width, emb_preprocessing=emb_preprocessing,
                freeze_obs_encoder=freeze_obs_encoder, freeze_pred_decoder=freeze_pred_decoder, load_obs_encoder=load_obs_encoder,)

        assert emb_height is not None and emb_width is not None, "Embedding height and width must be specified."
        self.emb_height = emb_height
        self.emb_width = emb_width
        # LOAD DIFFUSION UNET (from guided diffusion repo)
        self.unet = GDUnet_Latent(
                in_channels=1,
                out_channels=1,
                cond_embed_dim=cond_embed_dim,
                model_channels=model_channels,
                num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions,
                dropout=dropout,
                channel_mult=channel_mult,
                conv_resample=conv_resample,
                dims=dims,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown,
                use_new_attention_order=use_new_attention_order,
                dtype=dtype)

        self.init_params = {
            "n_features": n_features, 
            "n_landmarks": n_landmarks, 
            "obs_length": obs_length, 
            "pred_length": pred_length, 
            "embedder_obs_path": embedder_obs_path, 
            "embedder_pred_path": embedder_pred_path,
            # others
            "freeze_obs_encoder":freeze_obs_encoder,
            "freeze_pred_decoder":freeze_pred_decoder,
            "load_obs_encoder":load_obs_encoder,
            "emb_preprocessing":emb_preprocessing,
            # GDUNet parameters
            "emb_height":emb_height, 
            "emb_width":emb_width,
            "cond_embed_dim":cond_embed_dim,
            "model_channels":model_channels,
            "num_res_blocks":num_res_blocks,
            "attention_resolutions":attention_resolutions,
            "dropout":dropout,
            "channel_mult":channel_mult,
            "conv_resample":conv_resample,
            "dims":dims,
            "use_checkpoint":use_checkpoint,
            "dtype":dtype,
            "num_heads":num_heads,
            "num_head_channels":num_head_channels,
            "num_heads_upsample":num_heads_upsample,
            "use_scale_shift_norm":use_scale_shift_norm,
            "resblock_updown":resblock_updown,
            "use_new_attention_order":use_new_attention_order,
        }

    def preprocess(self, emb, stats, is_prediction=False):
        emb = super().preprocess(emb, stats) # normalize | standardize

        # we now implement the reordering
        if is_prediction and "meanstd" in self.emb_preprocessing:
            # extra reordering according to mean * std (from statistics)
            if "order_meanstd" not in stats or "order_inv_meanstd" not in stats:
                # compute reordering
                order, order_inv = compute_idces_order(stats["mean"] * stats["std"], self.emb_height, self.emb_width)
                stats["order_meanstd"] = order
                stats["order_inv_meanstd"] = order_inv
            bs = emb.shape[0]
            return torch.gather(emb, 1, torch.repeat_interleave(torch.tensor(stats["order_meanstd"], device=emb.device).unsqueeze(0), bs, axis=0)) # reorder according to mean * std from embedding statistics
        else:
            return emb

    def undo_preprocess(self, emb, stats, is_prediction=False):
        emb = super().undo_preprocess(emb, stats) # normalize | standardize

        # we now implement the reordering
        if is_prediction and "meanstd" in self.emb_preprocessing:
            # extra reordering according to mean * std (from statistics)
            if "order_meanstd" not in stats or "order_inv_meanstd" not in stats:
                # compute reordering
                order, order_inv = compute_idces_order(stats["mean"] * stats["std"], self.emb_height, self.emb_width)
                stats["order_meanstd"] = order
                stats["order_inv_meanstd"] = order_inv
            bs = emb.shape[0]
            return torch.gather(emb, 1, torch.repeat_interleave(torch.tensor(stats["order_inv_meanstd"], device=emb.device).unsqueeze(0), bs, axis=0)) # reorder according to mean * std from embedding statistics
        else:
            return emb

    def forward(self, pred, timesteps, obs):
        # The embedding step is done in the LatentDiffusion class, and we receive the embeddings here already
        
        # reshape pred to most square shape possible
        pred_emb_reshaped = pred.reshape((-1, 1, self.emb_height, self.emb_width)) # [BS, channels, height, width]
        out = self.unet(pred_emb_reshaped, timesteps, obs) # predict the prediction embedding
        out = out.reshape((out.shape[0], -1)) # -> BSx128
        return out


if __name__ == '__main__':
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    emb_obs_path = "results/models/test/h36m/models/auto_all25_TPKsmall_LR_S50/220615_155310/checkpoint-epoch500.pth"
    emb_pred_path = "results/models/test/h36m/models/auto_all_TPKLR_S50/220615_124725/checkpoint-epoch500.pth"
    
    #device = torch.device('cuda:0')
    device = torch.device('cpu')
    n_landmarks, n_features = 16, 3
    obs_length, pred_length = 25, 100
    batch_size = 64
    participants = 1
    #nx = ny = n_features * n_landmarks
    model = LatentUNetMatcher(n_features, n_landmarks, obs_length, pred_length, 
                emb_obs_path, emb_pred_path, emb_width=8, emb_height=16, dtype='float64',
                cond_embed_dim=64).to(device)


    # dumb input
    x = torch.tensor(np.zeros((batch_size, obs_length, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=torch.float64).contiguous()
    y = torch.tensor(np.zeros((batch_size, pred_length, participants, n_landmarks, n_features), dtype=np.float64), device=device, dtype=torch.float64).contiguous()
    timesteps = torch.tensor(np.array(np.random.randint(0, 1000, batch_size), dtype=int), device=device)

    print("\n> INPUT:\nx", x.shape, "\ny", y.shape)


    out = model(model.embed_pred.encode(y), timesteps, obs=model.embed_obs.encode(x))
    print("\n> OUTPUT:\noutput", out.shape, "\nmu")#, mu.shape, "\nlogvar", logvar.shape)