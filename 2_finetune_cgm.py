import os
import torch
from pytorch_lightning import loggers
import pytorch_lightning as pl
import os
import torch.nn.functional as F


from src.parameter_paser import parse_args_area_2
from src.dataset import ComplexDatasetLocs, generate_three_dataset_v3
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from src import EMA


if __name__ == '__main__':
    # loadind the existing model from orchard 1

    base_dir = os.path.dirname(os.path.realpath(__file__))

    args = parse_args_area_2(base_dir)
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = os.path.join(base_dir, "input")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    frac_for_valid = args.frac_for_valid
    frac_for_test = args.frac_for_test

    # using data from area 2
    data_path_area_2 = data_path = os.path.join(input_dir, args.data_name_ft)
    loaded = torch.load(data_path_area_2)
    amplitude = loaded['amplitude']
    phase = loaded['phase']
    label = loaded['label']

    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    num_rows = args.num_rows
    num_cols = args.num_cols
    row_col_idx_dict = {f"{i+1}_{j+1}": (i * num_cols) + j for i in range(num_rows) for j in range(num_cols)}

    complex_dataset = ComplexDatasetLocs(amplitude, 
                                         phase, 
                                         label, 
                                         location_vector_path, 
                                         row_col_idx_dict)
    
    ratios_loctions = args.ratios_locs
    train_data_set, valid_data_set, test_data_set \
        = generate_three_dataset_v3(complex_dataset, ratios_loctions, 
                                    frac_for_valid, frac_for_test)

    input_dim = args.input_dim
    num_epochs = args.num_epochs_cgm_ft
    batch_si = args.batch_size_cgm
    learning_rate = args.learning_rate_cgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    dimension_scale = args.channel_dimension_scale

    model_path_fintune_cgm = os.path.join(output_dir, args.cgm_fine_tune_path)
    model_path_fintune_cgm_run = os.path.join(output_dir, f"finetuned_cgm_running.ckpt")
    
    loaded_pretrained_cgm = os.path.join(output_dir, args.cgm_pretrain_path)

    cgm_logs = os.path.join(output_dir, f"cgm_log")
    os.makedirs(cgm_logs, exist_ok=True)

    tb_logger = loggers.TensorBoardLogger(save_dir=cgm_logs, 
                                        name='', 
                                        version="cgm")

    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir, 
        filename=model_path_fintune_cgm_run,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True)

    print("\nThe loaded diffusion model: {}\n".format(loaded_pretrained_cgm))

    model = PixelDiffusionConditional_v2.load_from_checkpoint(checkpoint_path=loaded_pretrained_cgm, 
                                                                train_dataset=train_data_set, 
                                                                input_dim=input_dim, 
                                                                loc_dim=loc_dim, 
                                                                channels=data_channels, 
                                                                dim_mults=dimension_scale, 
                                                                valid_dataset=valid_data_set, 
                                                                batch_size=batch_si, 
                                                                lr=learning_rate, 
                                                                loss_fn=model_loss, 
                                                                schedule=schedule, 
                                                                num_timesteps=num_timesteps, 
                                                                sampler=sampler_ddpm)
    
    trainer = pl.Trainer(max_epochs=num_epochs, 
                        callbacks=[EMA(0.9999)], 
                        log_every_n_steps=10,
                        accelerator='gpu', 
                        devices=[0], 
                        check_val_every_n_epoch=1,
                        logger=tb_logger)
    
    trainer.fit(model)

    trainer.save_checkpoint(model_path_fintune_cgm)



        
