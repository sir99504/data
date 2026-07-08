
import argparse


def get_args() -> dict:

    forcing = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "precipitation", "surface_pressure", "specific_humidity"]
    land_surface = ["surface_solar_radiation_downwards_w_m2", "surface_thermal_radiation_downwards_w_m2", "soil_temperature_level_1"]
    static_list = ['soil_water_capacity']
    """Parse input arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')

    # 经过处理后，将数据集保存到哪里
    parser.add_argument('--inputs_path', type=str, default='out/')

    # 数据输入路径（Landbench路径）
    parser.add_argument('--nc_data_path', type=str, default='data/')
    parser.add_argument('--product', type=str, default='LandBench')
    parser.add_argument('--workname', type=str, default='LandBench')
    # LSTM DARNN CNN
    parser.add_argument('--modelname', type=str, default='LSTM')
    parser.add_argument('--label', nargs='+', type=str, default=["volumetric_soil_water_layer_1"])
    parser.add_argument('--stride', type=float, default=20)
    parser.add_argument('--data_type', type=str, default='float32')
    # data
    parser.add_argument('--selected_year', nargs='+', type=int, default=[2015, 2020])
    parser.add_argument('--forcing_list', nargs='+', type=str, default=forcing)
    parser.add_argument('--land_surface_list', nargs='+', type=str, default=land_surface)
    parser.add_argument('--static_list', nargs='+', type=str, default=static_list)

    parser.add_argument('--memmap', type=bool, default=True)
    parser.add_argument('--test_year', nargs='+', type=int, default=[2020])
    parser.add_argument('--input_size', type=int, default=len(forcing) + len(land_surface) + len(static_list))
    parser.add_argument('--spatial_resolution', type=int, default=1)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--spatial_offset', type=int, default=3)
    parser.add_argument('--valid_split', type=bool, default=False)

    # model
    parser.add_argument('--normalize_type', type=str, default='global')
    parser.add_argument('--forecast_time', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--seq_len', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.015)
    parser.add_argument('--input_size_cnn', type=float, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride_cnn', type=int, default=2)
    cfg = vars(parser.parse_args())

    return cfg
