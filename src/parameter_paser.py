import argparse
import yaml
import os


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args_area_1(base_dir):
    default_config_name = "area_a.yml"
    
    parser_initial = argparse.ArgumentParser()
    parser_initial.add_argument('--config', default=default_config_name,
                                help='Configuration file name in the ./configs/*')
    
    args_initial = parser_initial.parse_known_args()[0]
    
    # Load the configuration
    config_file_path = os.path.join(base_dir, "configs", args_initial.config)
    with open(config_file_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                    type=str, 
                    default=default_config_name,
                    help="configure file path")

    parser.add_argument('--train_flag', 
                        type=str2bool, 
                        default=config_dict["train_flag"], 
                        help="train flag (True/False)")

    parser.add_argument('--data_name', 
                        type=str, 
                        default=config_dict["data_name"], 
                        help="input data file name")
    
    parser.add_argument('--save_model_name', 
                        type=str, 
                        default=config_dict["save_model_name"], 
                        help="save trained model")
    
    parser.add_argument('--frac_for_valid', 
                        type=float, 
                        default=config_dict["frac_for_valid"], 
                        help="frac_for_valid")
    
    parser.add_argument('--frac_for_test', 
                        type=float, 
                        default=config_dict["frac_for_test"], 
                        help="frac_for_test")

    parser.add_argument('--num_locs', 
                        type=int, 
                        default=config_dict["num_locs"], 
                        help="number of locations")
    
    parser.add_argument('--input_dim', 
                        type=int, 
                        default=config_dict["input_dim"], 
                        help="input_dim")
    
    parser.add_argument('--hidden_dim', 
                        type=int, 
                        default=config_dict["hidden_dim"], 
                        help="hidden_dim")
    
    parser.add_argument('--n_layers', 
                        type=int, 
                        default=config_dict["n_layers"], 
                        help="n_layers")
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=config_dict["learning_rate"], 
                        help="learning_rate")
    
    parser.add_argument('--dropout_pra', 
                        type=float, 
                        default=config_dict["dropout_pra"], 
                        help="dropout_pra")
    
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=config_dict["num_epochs"], 
                        help="num_epochs")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=config_dict["batch_size"], 
                        help="batch_size")

    args = parser.parse_args()

    return args


def parse_args_area_2(base_dir):

    default_config_name = "area_b.yml"
    parser_initial = argparse.ArgumentParser()
    parser_initial.add_argument('--config', default=default_config_name,
                                help='Configuration file name in the ./configs/*')
    
    args_initial = parser_initial.parse_known_args()[0]
    
    # Load the configuration
    config_file_path = os.path.join(base_dir, "configs", args_initial.config)
    with open(config_file_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', 
                    type=str, 
                    default=default_config_name,
                    help="configure file path")


    ##### Parameters used in area a #####
    # some parameters are reused for area b
    parser.add_argument('--train_flag', 
                        type=str2bool, 
                        default=config_dict["train_flag"], 
                        help="train flag (True/False)")

    parser.add_argument('--data_name', 
                        type=str, 
                        default=config_dict["data_name"], 
                        help="input data file name")
    
    parser.add_argument('--save_model_name', 
                        type=str, 
                        default=config_dict["save_model_name"], 
                        help="save trained model")
    
    parser.add_argument('--frac_for_valid', 
                        type=float, 
                        default=config_dict["frac_for_valid"], 
                        help="frac_for_valid")
    
    parser.add_argument('--frac_for_test', 
                        type=float, 
                        default=config_dict["frac_for_test"], 
                        help="frac_for_test")

    parser.add_argument('--num_locs', 
                        type=int, 
                        default=config_dict["num_locs"], 
                        help="number of locations")
    
    parser.add_argument('--input_dim', 
                        type=int, 
                        default=config_dict["input_dim"], 
                        help="input_dim")
    
    parser.add_argument('--hidden_dim', 
                        type=int, 
                        default=config_dict["hidden_dim"], 
                        help="hidden_dim")
    
    parser.add_argument('--n_layers', 
                        type=int, 
                        default=config_dict["n_layers"], 
                        help="n_layers")
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=config_dict["learning_rate"], 
                        help="learning_rate")
    
    parser.add_argument('--dropout_pra', 
                        type=float, 
                        default=config_dict["dropout_pra"], 
                        help="dropout_pra")
    
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=config_dict["num_epochs"], 
                        help="num_epochs")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=config_dict["batch_size"], 
                        help="batch_size")
    

    ##### New parameters for orchard #####
    parser.add_argument('--location_vector_name', 
                    type=str, 
                    default=config_dict["location_vector_name"], 
                    help="location_vector_name")
      
    parser.add_argument('--num_rows', 
                        type=int, 
                        default=config_dict["num_rows"], 
                        help="num_rows")
    
    parser.add_argument('--num_cols', 
                        type=int, 
                        default=config_dict["num_cols"], 
                        help="num_cols")
    
    parser.add_argument('--row_dist', 
                        type=float, 
                        default=config_dict["row_dist"], 
                        help="row_dist")
    
    parser.add_argument('--col_dist', 
                        type=float, 
                        default=config_dict["col_dist"], 
                        help="col_dist")
    
    parser.add_argument('--r_cylinder', 
                        type=float, 
                        default=config_dict["r_cylinder"], 
                        help="r_cylinder")
    
    parser.add_argument('--h_cylinder', 
                        type=float, 
                        default=config_dict["h_cylinder"], 
                        help="h_cylinder")
    
    parser.add_argument('--r_ellipsoid', 
                        type=float, 
                        default=config_dict["r_ellipsoid"], 
                        help="r_ellipsoid")
    

    parser.add_argument('--h_ellipsoid', 
                        type=float, 
                        default=config_dict["h_ellipsoid"], 
                        help="h_ellipsoid")
    
    parser.add_argument('--density', 
                        type=float, 
                        default=config_dict["density"], 
                        help="density")
    
    parser.add_argument('--maximum_comm_dis', 
                        type=float, 
                        default=config_dict["maximum_comm_dis"], 
                        help="maximum_comm_dis")
    
    parser.add_argument('--lora_frequency', 
                        type=float, 
                        default=config_dict["lora_frequency"], 
                        help="lora_frequency")
    

    ##### New parameters for pre-training CGM #####
    parser.add_argument('--loc_dim', 
                        type=int, 
                        default=config_dict["loc_dim"], 
                        help="loc_dim")
    
    parser.add_argument('--num_timesteps', 
                        type=int, 
                        default=config_dict["num_timesteps"], 
                        help="num_timesteps")
    
    parser.add_argument('--schedule', 
                        type=str, 
                        default=config_dict["schedule"], 
                        help="schedule")
        
    parser.add_argument('--data_channels', 
                        type=int, 
                        default=config_dict["data_channels"], 
                        help="data_channels")
    
    parser.add_argument('--channel_dimension_scale', 
                        type=tuple, 
                        default=tuple(config_dict["channel_dimension_scale"]), 
                        help="channel_dimension_scale")
    
    parser.add_argument('--num_epochs_cgm', 
                        type=int, 
                        default=config_dict["num_epochs_cgm"], 
                        help="num_epochs_cgm")
    
    parser.add_argument('--batch_size_cgm', 
                    type=int, 
                    default=config_dict["batch_size_cgm"], 
                    help="batch_size_cgm")

    parser.add_argument('--learning_rate_cgm', 
                        type=float, 
                        default=config_dict["learning_rate_cgm"], 
                        help="learning_rate_cgm")

    parser.add_argument('--cgm_pretrain_path', 
                        type=str, 
                        default=config_dict["cgm_pretrain_path"], 
                        help="cgm_pretrain_path")


    ##### New parameters for fine-tuning CGM #####
    parser.add_argument('--data_name_ft', 
                        type=str, 
                        default=config_dict["data_name_ft"], 
                        help="input data file name from area b")

    parser.add_argument('--ratios_locs', 
                        type=float, 
                        default=config_dict["ratios_locs"], 
                        help="ratios_locs")
    
    parser.add_argument('--data_name_fake', 
                        type=str, 
                        default=config_dict["data_name_fake"], 
                        help="generated and real CSI data file name for area b")
    
    parser.add_argument('--save_model_name_fake', 
                        type=str, 
                        default=config_dict["save_model_name_fake"], 
                        help="save trained classifier by fake data for area b")
    
    parser.add_argument('--num_epochs_cgm_ft', 
                type=int, 
                default=config_dict["num_epochs_cgm_ft"], 
                help="num_epochs_cgm_ft")

    parser.add_argument('--cgm_fine_tune_path', 
                        type=str, 
                        default=config_dict["cgm_fine_tune_path"], 
                        help="cgm_fine_tune_path")   


    args = parser.parse_args()

    return args


