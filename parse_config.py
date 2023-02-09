import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
import torch
import random

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, save=True):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        config = self.custom_modifications(config)
        self._config = _update_config(config, modification)
        self.resume = resume


        if save:
            assert "config_path" in self.config, "config_path not in config"
            # set save_dir where trained model and log will be saved.
            save_dir = os.path.dirname(self.config["config_path"])
            #Path(self.config['trainer']['save_dir'])

            exper_name = self.config['name']
            if run_id is None: # use timestamp as default run-id
                run_id = datetime.now().strftime(r'%y%m%d_%H%M%S') + f"_{random.randint(0,1000):03d}"
            if resume is not None and "models" in resume:
                resumed_folder = os.path.dirname(resume)
                self._save_dir = Path(resumed_folder)
                self._log_dir = self._save_dir
            else:
                self._save_dir = Path(save_dir)
                self._log_dir = Path(save_dir)

            # make directory for saving checkpoints and log.
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save updated config file to the checkpoint dir
            if not resume:
                config["unique_id"] = run_id
            #write_json(self.config, self.save_dir / 'config.json')

            # configure logging module
            setup_logging(self.log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def custom_modifications(self, config_dict):
        dtype = torch.float64 # default to double precision
        if "dtype" in config_dict:
            val = config_dict["dtype"].lower()
            assert val in ["float32", "float64"], "Project can only work with either float32 or float64 dtypes."
            dtype = torch.float32 if config_dict["dtype"].lower() == 'float32' else torch.float64
        else:
            config_dict["dtype"] = "float64"
        torch.set_default_dtype(dtype) 

        # setup data_loader instances
        for dl in ["data_loader_training", "data_loader_validation", "data_loader_test"]:
            if dl not in config_dict:
                continue
            config_dict[dl]["args"]["normalize_data"] = config_dict["normalize_data"] if "normalize_data" in config_dict else True
            config_dict[dl]["args"]["normalize_type"] = config_dict["normalize_type"] if "normalize_type" in config_dict else "standardize"
            config_dict[dl]["args"]["precomputed_folder"] = config_dict["precomputed_folder"]
            config_dict[dl]["args"]["obs_length"] = config_dict["obs_length"]
            config_dict[dl]["args"]["pred_length"] = config_dict["pred_length"]
            if "trainer" in config_dict:
                config_dict[dl]["args"]["batch_size"] = config_dict["trainer"]["batch_size"]
                config_dict[dl]["args"]["num_workers"] = config_dict["trainer"]["num_workers"]
            config_dict[dl]["args"]["seed"] = config_dict["seed"]
            config_dict[dl]["args"]["dtype"] = config_dict["dtype"]

        # build model architecture, then print to console
        for prefix in ["", "aux_"]: # "aux_" for auxiliary tasks
            if prefix + "arch" in config_dict:
                config_dict[prefix + "arch"]["args"]["n_landmarks"] = config_dict["landmarks"]
                config_dict[prefix + "arch"]["args"]["n_features"] = config_dict["dims"]
                config_dict[prefix + "arch"]["args"]["obs_length"] = config_dict["obs_length"]
                config_dict[prefix + "arch"]["args"]["pred_length"] = config_dict["pred_length"]
            
            if prefix + "loss" in config_dict:
                config_dict[prefix + "loss"]["args"]["n_dims"] = config_dict['eval_dims']
            if prefix + "metrics" in config_dict:
                for met in config_dict[prefix + "metrics"]:
                    if "args" not in met:
                        met["args"] = {}
                    met["args"]["n_dims"] = config_dict["eval_dims"]

        # for GAN training
        for key, suffix in zip(["generator", "discriminator"], ["_G", "_D"]):
            if key in config_dict:
                config_dict[key]["args"]["n_landmarks"] = config_dict["landmarks"]
                config_dict[key]["args"]["n_features"] = config_dict["dims"]
                config_dict[key]["args"]["obs_length"] = config_dict["obs_length"]
                config_dict[key]["args"]["pred_length"] = config_dict["pred_length"]

            loss_key = "loss" + suffix
            if loss_key in config_dict:
                config_dict[loss_key]["args"]["n_dims"] = config_dict['eval_dims']
            met_key = "metrics" + suffix
            if met_key in config_dict:
                for met in config_dict[met_key]:
                    if "args" not in met:
                        met["args"] = {}
                    met["args"]["n_dims"] = config_dict["eval_dims"]

            

        #assert config_dict['eval_dims'] in (2, 3), "'eval_dims' must be either 2 or 3"
        #config_dict["loss"] = config_dict["loss"].format(config_dict['eval_dims'])
        #config_dict["metrics"] = [met.format(config_dict['eval_dims']) for met in config_dict["metrics"]]

        return config_dict

    @classmethod
    def from_args(cls, args, options='', save=True):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None and args.config is None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = args.resume if args.resume is not None else None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, save=save)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
