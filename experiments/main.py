import sys
sys.path.append('/home/kehuiyao/ST-Imputation/')
sys.path.append('/Users/kehuiyao/Desktop/ST-Imputation/')


import argparse
import copy
import datetime
import os
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from data import GaussianProcess, DescriptiveST, DynamicST, SoilMoistureSparse, SoilMoistureHB, Sine, HealingMnist

from tsl.imputers import Imputer
from tsl.nn.metrics import MaskedMetric, MaskedMAE, MaskedMSE, MaskedMRE
from tsl.nn.models.imputation import GRINModel
from tsl.nn.utils import casting
from tsl.utils.parser_utils import ArgParser
from tsl.ops.imputation import add_missing_values
from tsl.utils import parser_utils, numpy_metrics

from spin.baselines import SAITS, TransformerModel, BRITS, MeanModel, InterpolationModel
from spin.imputers import SPINImputer, SAITSImputer, BRITSImputer, MeanImputer, InterpolationImputer, DiffgrinImputer, CsdiImputer, GrinImputer, TransformerImputer
from spin.models import SPINModel, SPINHierarchicalModel, DiffGrinModel, CsdiModel, GrinModel, SpatioTemporalTransformerModel
from spin.scheduler import CosineSchedulerWithRestarts

from tqdm import tqdm

def parse_args():
    # Argument parser
    ########################################
    parser = ArgParser()
    parser.add_argument("--dataset-name", type=str, default='healing_mnist_point')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--load-from-pretrained', type=str,
                        default=None)
    parser.add_argument('--precision', type=int, default=32)
    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.1)
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    parser.add_argument('--p-fault', type=float, default=0.0)
    parser.add_argument('--p-noise', type=float, default=0.95)

    known_args, _ = parser.parse_known_args()


    args = parser.parse_args()


    return args


def get_model_classes(model_str):
    if model_str == 'spin':
        model, filler = SPINModel, SPINImputer
    elif model_str == 'spin_h':
        model, filler = SPINHierarchicalModel, SPINImputer
    elif model_str == 'grin':
        model, filler = GrinModel, GrinImputer
    elif model_str == 'saits':
        model, filler = SAITS, SAITSImputer
    elif model_str == 'transformer':
        model, filler = TransformerModel, TransformerImputer
    elif model_str == 'brits':
        model, filler = BRITS, BRITSImputer
    elif model_str == 'mean':
        model, filler = MeanModel, MeanImputer
    elif model_str == 'interpolation':
        model, filler = InterpolationModel, InterpolationImputer
    elif model_str == 'diffgrin':
        model, filler = DiffGrinModel, DiffgrinImputer
    elif model_str == 'csdi':
        model, filler = CsdiModel, CsdiImputer
    elif model_str == 'st_transformer':
        model, filler = SpatioTemporalTransformerModel, TransformerImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name: str, args=None):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    # build missing dataset
    if dataset_name.endswith('_point'):
        p_fault, p_noise = args.p_fault, args.p_noise
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_block'):
        p_fault, p_noise = args.p_fault, args.p_noise
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)

    if dataset_name == 'soil_moisture_hb':
        return SoilMoistureHB(mode=args.mode)

    if dataset_name == 'soil_moisture_sparse':
        return SoilMoistureSparse(mode=args.mode)


    if dataset_name == 'gp':
        return add_missing_values(GaussianProcess(num_nodes=36, seq_len=400), p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789)

    if dataset_name == 'descriptive':
        return add_missing_values(DescriptiveST(num_nodes=36, seq_len=400), p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789)

    if dataset_name == 'dynamic':
        return add_missing_values(DynamicST(num_nodes=36, seq_len=400), p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789)

    if dataset_name == 'sine':
        return Sine(num_nodes=144, seq_len=40000)
        # return add_missing_values(Sine(num_nodes=144, seq_len=4000), p_fault=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'healing_mnist':
        return HealingMnist(mode='train')

    raise ValueError(f"Invalid dataset name: {dataset_name}.")



def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    elif scheduler_name == 'magic':
        scheduler_class = CosineSchedulerWithRestarts
        scheduler_kwargs = dict(num_warmup_steps=12, min_factor=0.1,
                                linear_decay=0.67,
                                num_training_steps=args.epochs,
                                num_cycles=args.epochs // 100)

    elif scheduler_name == 'multi_step':
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        p1 = int(0.6 * args.epochs)
        p2 = int(0.9 * args.epochs)
        scheduler_kwargs = dict(milestones=[p1, p2], gamma=0.1)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs


def get_dataloader(dataset, args=None):
    ########################################
    # data module                          #
    ########################################

    if args.model_name == 'csdi' and 'covariates' in dataset.attributes:
        exog_map = {'covariates': dataset.attributes['covariates']}
        input_map = {
            'side_info': 'covariates',
            'x': 'data'
        }

    elif args.model_name == 'grin' and 'covariates' in dataset.attributes:
        exog_map = {'covariates': dataset.attributes['covariates']}
        input_map = {
            'u': 'covariates',
            'x': 'data'
        }


    elif args.model_name == 'diffgrin' and 'covariates' in dataset.attributes:

        exog_map = {'covariates': dataset.attributes['covariates']}

        input_map = {
            'side_info': 'covariates',
            'x': 'data'
        }

    elif args.model_name == 'transformer' and 'covariates' in dataset.attributes:
        exog_map = {'covariates': dataset.attributes['covariates']}

        input_map = {
            'u': 'covariates',
            'x': 'data'
        }

    elif args.model_name == 'st_transformer' and 'covariates' in dataset.attributes:
        exog_map = {'covariates': dataset.attributes['covariates']}
        input_map = {
            'u': 'covariates',
            'x': 'data'
        }

    else:
        exog_map = input_map = None

    if 'st_coords' in dataset.attributes:
        if exog_map is None and input_map is None:
            exog_map = {'st_coords': dataset.attributes['st_coords']}
            input_map = {'x': 'data', 'st_coords': 'st_coords'}
        else:
            exog_map['st_coords'] = dataset.attributes['st_coords']
            input_map['st_coords'] = 'st_coords'

    if args.model_name in ['spin', 'spin_h', 'grin', 'diffgrin']:
        adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                       include_self=False,
                                       )
    else:
        adj = None

    # instantiate dataset
    torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)

    if args.mode == 'train':
        # get train/val/test indices
        splitter = dataset.get_splitter(val_len=args.val_len,
                                        test_len=args.test_len)

        # scalers = {'data': MinMaxScaler(axis=(0, 1), out_range=(-1, 1))}
        scalers = {'data': StandardScaler(axis=(0, 1))}

        dm = SpatioTemporalDataModule(torch_dataset,
                                      scalers=scalers,
                                      splitter=splitter,
                                      batch_size=args.batch_size // args.split_batch_in)


    elif args.mode=='test':
        scaler = StandardScaler(axis=(0, 1))
        scaler.fit(dataset.numpy(), dataset.training_mask)
        scaler.bias = torch.tensor(scaler.bias)
        scaler.scale = torch.tensor(scaler.scale)
        scalers = {'data': scaler}

        # instantiate dataset
        torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                          training_mask=dataset.training_mask,
                                          eval_mask=dataset.eval_mask,
                                          connectivity=adj,
                                          exogenous=exog_map,
                                          input_map=input_map,
                                          window=args.window,
                                          stride=args.stride,
                                          scalers=scalers)

        # get train/val/test indices
        splitter = dataset.get_splitter(val_len=0,
                                        test_len=len(torch_dataset))

        dm = SpatioTemporalDataModule(torch_dataset,
                                      splitter=splitter,
                                      batch_size=args.batch_size // args.split_batch_in)


    dm.setup()


    return dm

def evaluate_imputer(dataset, dm, imputer, logdir, args=None, save_dir='output'):

    imputer.freeze()


    y_hat = []
    y_true = []
    eval_mask = []
    observed_mask = []
    st_coords = []

    if args.model_name in ['csdi', 'diffgrin']:
        enable_multiple_imputation = True
    else:
        enable_multiple_imputation = False

    if enable_multiple_imputation:
        multiple_imputations = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch_id, batch in enumerate(
            tqdm(dm.test_dataloader(batch_size=args.batch_inference), desc="Processing", leave=True)):

        batch = batch.to(device)
        imputer = imputer.to(device)

        batch = imputer.on_after_batch_transfer(batch, 0)
        output = imputer.predict_step(batch, batch_id)

        y_hat.append(output['y_hat'].detach().cpu().numpy())
        y_true.append(output['y'].detach().cpu().numpy())
        eval_mask.append(output['eval_mask'].detach().cpu().numpy())
        observed_mask.append(output['observed_mask'].detach().cpu().numpy())
        st_coords.append(output['st_coords'].detach().cpu().numpy())

        if enable_multiple_imputation:
            multiple_imputations.append(output['imputed_samples'].detach().cpu().numpy())

    y_hat = np.concatenate(y_hat, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    eval_mask = np.concatenate(eval_mask, axis=0)
    observed_mask = np.concatenate(observed_mask, axis=0)
    st_coords = np.concatenate(st_coords, axis=0)

    y_hat = y_hat.squeeze(-1)
    y_true = y_true.squeeze(-1)
    eval_mask = eval_mask.squeeze(-1)
    observed_mask = observed_mask.squeeze(-1)

    if enable_multiple_imputation:
        multiple_imputations = np.concatenate(multiple_imputations, axis=0)
        multiple_imputations = multiple_imputations.squeeze(-1)

    # check_mae = numpy_metrics.masked_mae(y_hat, y_true, eval_mask)
    # print(f'Test MAE: {check_mae:.4f}')

    # seq_len = 1827
    seq_len = dataset.original_data['y'].shape[0]
    print(seq_len)
    # num_nodes = 1296
    num_nodes = dataset.original_data['y'].shape[1]
    y_true_original = np.zeros([seq_len, num_nodes])
    y_hat_original = np.zeros([seq_len, num_nodes])
    if enable_multiple_imputation:
        y_hat_multiple_imputation = np.zeros([multiple_imputations.shape[1], seq_len, num_nodes])
    observed_mask_original = np.zeros([seq_len, num_nodes])
    eval_mask_original = np.zeros([seq_len, num_nodes])
    count = np.zeros([seq_len, num_nodes])
    y_hat_original_sum = np.zeros([seq_len, num_nodes])

    B, L, K = y_hat.shape
    for b in range(B):
        for l in range(L):
            for k in range(K):
                ts_pos = st_coords[b, l, k, ::-1]
                y_true_original[ts_pos[0], ts_pos[1]] = y_true[b, l, k]
                y_hat_original_sum[ts_pos[0], ts_pos[1]] += y_hat[b, l, k]
                count[ts_pos[0], ts_pos[1]] += 1

                observed_mask_original[ts_pos[0], ts_pos[1]] = observed_mask[b, l, k]
                eval_mask_original[ts_pos[0], ts_pos[1]] = eval_mask[b, l, k]

                if enable_multiple_imputation:
                    y_hat_multiple_imputation[:, ts_pos[0], ts_pos[1]] = multiple_imputations[b, :, l, k]

    # for those positions that count is not 0, we divide the sum by count to get the average
    y_hat_original[count != 0] = y_hat_original_sum[count != 0] / count[count != 0]


    check_mae = numpy_metrics.masked_mae(y_hat_original, y_true_original, eval_mask_original)
    print(f'Test MAE: {check_mae:.6f}')

    check_mre = numpy_metrics.masked_mre(y_hat_original, y_true_original, eval_mask_original)
    print(f'Test MRE: {check_mre:.6f}')

    # save output to file
    output = {}
    output['y_hat'] = y_hat_original[np.newaxis, :, :, np.newaxis]
    output['y'] = y_true_original[np.newaxis, :, :, np.newaxis]
    output['eval_mask'] = eval_mask_original[np.newaxis, :, :, np.newaxis]
    output['observed_mask'] = observed_mask_original[np.newaxis, :, :, np.newaxis]

    if enable_multiple_imputation:
        output['imputed_samples'] = y_hat_multiple_imputation[np.newaxis, :, :, :, np.newaxis]

    np.savez(os.path.join(logdir, save_dir+'.npz'), **output)

    return check_mae, check_mre



def run_experiment(args):
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # script flags
    is_spin = args.model_name in ['spin', 'spin_h']



    # logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)




    ########################################
    # training                             #
    ########################################
    args.mode = 'train'
    model_cls, imputer_class = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name, args)
    dm = get_dataloader(dataset, args)

    ########################################
    # predictor                            #
    ########################################
    if args.model_name in ['spin', 'spin_h']:
        additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                        input_size=dm.n_channels,
                                        u_size=4,
                                        output_size=dm.n_channels,
                                        window_size=dm.window)
    else:
        additional_model_hparams = dict(n_nodes=dm.n_nodes, input_size=dm.n_channels, output_size=dm.n_channels, window_size=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class,
                                                       return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        **imputer_kwargs
    )

    require_training = True
    if args.model_name in ['mean', 'interpolation']:
        require_training = False

    if args.load_from_pretrained is not None:
        require_training = False

    # callbacks
    if args.loss_fn == 'l1_loss':
        monitor = 'val_mae'
    elif args.loss_fn == 'mse_loss':
        monitor = 'val_mse'

    early_stop_callback = EarlyStopping(monitor=monitor,
                                        patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1,
                                          monitor=monitor, mode='min')

    tb_logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=tb_logger,
                         precision=args.precision,
                         accumulate_grad_batches=args.split_batch_in,
                         gpus=int(torch.cuda.is_available()),
                         gradient_clip_val=args.grad_clip_val,
                         limit_train_batches=args.batches_epoch * args.split_batch_in,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         callbacks=[early_stop_callback, checkpoint_callback])
    if require_training:
        trainer.fit(imputer,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader(
                        batch_size=args.batch_inference))
    elif args.load_from_pretrained is not None:
        imputer = imputer_class.load_from_checkpoint(args.load_from_pretrained, model_class=model_cls,
         model_kwargs=model_kwargs,
         optim_class=torch.optim.Adam,
         optim_kwargs={'lr': args.lr,
                  'weight_decay': args.l2_reg},
         loss_fn=loss_fn,
         metrics=metrics,
         scheduler_class=scheduler_class,
         scheduler_kwargs=scheduler_kwargs,
         **imputer_kwargs)

    ########################################
    # in-sample testing
    ########################################
    args.mode='test'
    dm = get_dataloader(dataset, args)
    in_sample_mae, in_sample_mre = evaluate_imputer(dataset, dm, imputer, logdir, args, 'in_sample')

    ########################################
    # out-sample testing                              #
    ########################################
    args.mode = 'test'
    dataset = get_dataset(args.dataset_name, args)
    dm = get_dataloader(dataset, args)
    out_sample_mae, out_sample_mre = evaluate_imputer(dataset, dm, imputer, logdir, args, 'out_sample')


    return in_sample_mae, in_sample_mre, out_sample_mae, out_sample_mre


def main(args):
    # Set configuration and seed
    args = copy.deepcopy(args)

    # seed
    n_rep = 1
    seed_list = [i for i in range(n_rep)]

    # model
    # model_list = ['st_transformer', 'grin', 'csdi']
    # model_list = ['interpolation']
    # model_list = ['csdi', 'st_transformer', 'transformer', 'mean', 'interpolation']
    # model_list = ['st_transformer']
    # model_config = ['imputation/st_transformer_soil_moisture_hb.yaml']

    # model_list = ['grin']
    # model_config = ['imputation/grin_soil_moisture_hb.yaml']

    # model_list = ['mean']
    # model_config = ['imputation/mean_soil_moisture.yaml']

    # model_list = ['interpolation']
    # model_config = ['imputation/interpolation_soil_moisture.yaml']

    # model_list = ['st_transformer']
    # model_config = ['imputation/st_transformer_healing_mnist.yaml']

    model_list = ['grin']
    model_config = ['imputation/grin_healing_mnist.yaml']




    # svaing result
    result = {}
    for model in model_list:
        result[model] = {}
        result[model]['validation'] = {}
        result[model]['test'] = {}
        result[model]['validation']['mae'] = []
        result[model]['validation']['mre'] = []
        result[model]['test']['mae'] = []
        result[model]['test']['mre'] = []



    for seed in seed_list:
        args.seed = seed

        for model_ind in range(len(model_list)):
            model = model_list[model_ind]
            args.model_name = model
            args.config = model_config[model_ind]
            model_cls, imputer_cls = get_model_classes(args.model_name)
            parser = ArgParser()
            parser = model_cls.add_model_specific_args(parser)
            parser = imputer_cls.add_argparse_args(parser)
            additional_args = parser.parse_args()

            # but you can convert them to dictionaries:
            dict_args1 = vars(args)
            dict_args2 = vars(additional_args)

            # Now you can merge the dictionaries
            combined_args = {**dict_args1, **dict_args2}

            # If you want to turn this back into a Namespace object:
            args = argparse.Namespace(**combined_args)




            if args.config is not None:
                cfg_path = os.path.join(config.config_dir, args.config)
                with open(cfg_path, 'r') as fp:
                    config_args = yaml.load(fp, Loader=yaml.FullLoader)
                for arg in config_args:
                    setattr(args, arg, config_args[arg])


            in_sample_mae, in_sample_mre, out_sample_mae, out_sample_mre = run_experiment(args)

            result[model]['validation']['mae'].append(in_sample_mae)
            result[model]['validation']['mre'].append(in_sample_mre)
            result[model]['test']['mae'].append(out_sample_mae)
            result[model]['test']['mre'].append(out_sample_mre)


    # compute mean and std
    for model in model_list:
        for mode in ['validation', 'test']:
            for metric in ['mae', 'mre']:
                metric_mean = np.mean(result[model][mode][metric])
                metric_std = np.std(result[model][mode][metric])

                # print result
                print(f'{model} {mode} {metric}: {metric_mean:.6f} +/- {metric_std:.6f}')

    # save result
    np.savez(os.path.join(config.log_dir, 'result.npz'), **result)

if __name__ == '__main__':
    args = parse_args()
    main(args)

