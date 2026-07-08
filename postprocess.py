import os
import numpy as np
from utils import unbiased_rmse, _rmse, _bias, GetKGE, r2_score, GetPCC, GetNSE, _rv, _fhv, _flv
from config import get_args
import warnings
warnings.filterwarnings("ignore")


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def get_swdi(y_pred_lstm,fc,awc):
    swdi = (y_pred_lstm - fc)/awc*10
    return swdi



def postprocess(cfg):
    PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    if cfg['modelname'] in ['ConvLSTM']:
        out_path_convlstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
            'workname'] + '/' + cfg['modelname'] + '/forecast_time ' + str(cfg['forecast_time']) + '/'
        y_pred_convlstm = np.load(out_path_convlstm + '_predictions.npy')
        y_test_convlstm = np.load(out_path_convlstm + 'observations.npy')
        print(y_pred_convlstm.shape, y_test_convlstm.shape)
        # get shape
        nt, nlat, nlon = y_test_convlstm.shape
        # cal perf
        r2_convlstm = np.full((nlat, nlon), np.nan)
        urmse_convlstm = np.full((nlat, nlon), np.nan)
        r_convlstm = np.full((nlat, nlon), np.nan)
        rmse_convlstm = np.full((nlat, nlon), np.nan)
        bias_convlstm = np.full((nlat, nlon), np.nan)

        scanler = np.full((2, nlat, nlon), np.nan)
        for aa in range(y_pred_convlstm.shape[1]):
            for bb in range(y_pred_convlstm.shape[2]):
                sorted_list = sorted(y_pred_convlstm[:, aa, bb])
                scanler[0, aa, bb] = sorted_list[18]
                scanler[1, aa, bb] = sorted_list[346]
        fc = scanler[1, :, :]
        wp = scanler[0, :, :]
        awc = fc - wp
        swdi_obs = np.full((nt, nlat, nlon), np.nan)
        swdi_lstm = np.full((nt, nlat, nlon), np.nan)


        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_convlstm[:, i, j]).any()):
                    urmse_convlstm[i, j] = unbiased_rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    # r2_convlstm[i, j] = r2_score(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    r_convlstm[i, j] = np.corrcoef(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])[0, 1]
                    rmse_convlstm[i, j] = _rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    bias_convlstm[i, j] = _bias(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    swdi_obs[:, i, j] = get_swdi(y_test_convlstm[:, i, j], fc[i, j], awc[i, j])
                    swdi_lstm[:, i, j] = get_swdi(y_pred_convlstm[:, i, j], fc[i, j], awc[i, j])
        np.save(out_path_convlstm + 'r2_' + cfg['modelname'] + '.npy', r2_convlstm)
        np.save(out_path_convlstm + 'r_' + cfg['modelname'] + '.npy', r_convlstm)
        np.save(out_path_convlstm + 'rmse_' + cfg['modelname'] + '.npy', rmse_convlstm)
        np.save(out_path_convlstm + 'bias_' + cfg['modelname'] + '.npy', bias_convlstm)
        np.save(out_path_convlstm + 'urmse_' + cfg['modelname'] + '.npy', urmse_convlstm)

        swdi_obs[swdi_obs > 0] = 1
        swdi_obs[(-2 < swdi_obs) & (swdi_obs <= 0)] = 2
        swdi_obs[(-5 < swdi_obs) & (swdi_obs <= -2)] = 3
        swdi_obs[(-10 < swdi_obs) & (swdi_obs <= -5)] = 4
        swdi_obs[swdi_obs <= -10] = 5

        swdi_lstm[swdi_lstm > 0] = 1
        swdi_lstm[(-2 < swdi_lstm) & (swdi_lstm <= 0)] = 2
        swdi_lstm[(-5 < swdi_lstm) & (swdi_lstm <= -2)] = 3
        swdi_lstm[(-10 < swdi_lstm) & (swdi_lstm <= -5)] = 4
        swdi_lstm[swdi_lstm <= -10] = 5

        np.save(out_path_convlstm + 'swdi_obs' + '.npy', swdi_obs)
        np.save(out_path_convlstm + 'swdi_' + cfg['modelname'] + '.npy', swdi_lstm)

        print('postprocess over, please go on')
    # ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['LSTM']:
        out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
            'workname'] + '/' + cfg['modelname'] + '/forecast_time ' + str(cfg['forecast_time']) + '/'
        y_pred_lstm = np.load(out_path_lstm + '_predictions.npy')
        y_test_lstm = np.load(out_path_lstm + 'observations.npy')

        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape
        # cal perf
        r2_lstm = np.full((nlat, nlon), np.nan)
        GetKGE_lstm = np.full((nlat, nlon), np.nan)
        GetPCC_lstm = np.full((nlat, nlon), np.nan)
        GetNSE_lstm = np.full((nlat, nlon), np.nan)
        urmse_lstm = np.full((nlat, nlon), np.nan)
        r_lstm = np.full((nlat, nlon), np.nan)
        rmse_lstm = np.full((nlat, nlon), np.nan)
        bias_lstm = np.full((nlat, nlon), np.nan)
        rv_lstm = np.full((nlat, nlon), np.nan)
        fhv_lstm = np.full((nlat, nlon), np.nan)
        flv_lstm = np.full((nlat, nlon), np.nan)


        scanler = np.full((2, nlat, nlon), np.nan)
        for aa in range(y_test_lstm.shape[1]):
            for bb in range(y_test_lstm.shape[2]):
                sorted_list = sorted(y_test_lstm[:, aa, bb])
                scanler[0, aa, bb] = sorted_list[18]
                scanler[1, aa, bb] = sorted_list[346]
        fc = scanler[1, :, :]
        wp = scanler[0, :, :]
        awc = fc - wp
        swdi_obs = np.full((nt, nlat, nlon), np.nan)
        swdi_lstm = np.full((nt, nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetKGE_lstm[i, j] = GetKGE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetPCC_lstm[i, j] = GetPCC(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetNSE_lstm[i, j] = GetNSE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    rv_lstm[i, j] = _rv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    fhv_lstm[i, j] = _fhv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    flv_lstm[i, j] = _flv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0, 1]
                    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])

                    swdi_obs[:, i, j] = get_swdi(y_test_lstm[:, i, j], fc[i, j], awc[i, j])
                    swdi_lstm[:, i, j] = get_swdi(y_pred_lstm[:, i, j], fc[i, j], awc[i, j])
        np.save(out_path_lstm + 'r2_' + cfg['modelname'] + '.npy', r2_lstm)
        np.save(out_path_lstm + 'KGE_' + cfg['modelname'] + '.npy', GetKGE_lstm)
        np.save(out_path_lstm + 'PCC_' + cfg['modelname'] + '.npy', GetPCC_lstm)
        np.save(out_path_lstm + 'NSE_' + cfg['modelname'] + '.npy', GetNSE_lstm)
        np.save(out_path_lstm + 'rv_' + cfg['modelname'] + '.npy', rv_lstm)
        np.save(out_path_lstm + 'fhv_' + cfg['modelname'] + '.npy', fhv_lstm)
        np.save(out_path_lstm + 'flv_' + cfg['modelname'] + '.npy', flv_lstm)
        np.save(out_path_lstm + 'r_' + cfg['modelname'] + '.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_' + cfg['modelname'] + '.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_' + cfg['modelname'] + '.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_' + cfg['modelname'] + '.npy', urmse_lstm)
        swdi_obs[swdi_obs > 0] = 1
        swdi_obs[(-2 < swdi_obs) & (swdi_obs <= 0)] = 2
        swdi_obs[(-5 < swdi_obs) & (swdi_obs <= -2)] = 3
        swdi_obs[(-10 < swdi_obs) & (swdi_obs <= -5)] = 4
        swdi_obs[swdi_obs <= -10] = 5

        swdi_lstm[swdi_lstm > 0] = 1
        swdi_lstm[(-2 < swdi_lstm) & (swdi_lstm <= 0)] = 2
        swdi_lstm[(-5 < swdi_lstm) & (swdi_lstm <= -2)] = 3
        swdi_lstm[(-10 < swdi_lstm) & (swdi_lstm <= -5)] = 4
        swdi_lstm[swdi_lstm <= -10] = 5

        np.save(out_path_lstm + 'swdi_obs' + '.npy', swdi_obs)
        np.save(out_path_lstm + 'swdi_' + cfg['modelname'] + '.npy', swdi_lstm)

        # ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['GraphLSTM', 'GraphConvLSTM']:
        out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg[
            'workname'] + '/' + cfg['modelname'] + '/forecast_time ' + str(cfg['forecast_time']) + '/'
        y_pred_lstm = np.load(out_path_lstm + '_predictions.npy')
        y_test_lstm = np.load(out_path_lstm + 'observations.npy')

        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape
        # cal perf
        r2_lstm = np.full((nlat, nlon), np.nan)
        GetKGE_lstm = np.full((nlat, nlon), np.nan)
        GetPCC_lstm = np.full((nlat, nlon), np.nan)
        GetNSE_lstm = np.full((nlat, nlon), np.nan)
        urmse_lstm = np.full((nlat, nlon), np.nan)
        r_lstm = np.full((nlat, nlon), np.nan)
        rmse_lstm = np.full((nlat, nlon), np.nan)
        bias_lstm = np.full((nlat, nlon), np.nan)
        rv_lstm = np.full((nlat, nlon), np.nan)
        fhv_lstm = np.full((nlat, nlon), np.nan)
        flv_lstm = np.full((nlat, nlon), np.nan)

        swdi_lstm = np.full((nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetKGE_lstm[i, j] = GetKGE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetPCC_lstm[i, j] = GetPCC(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    GetNSE_lstm[i, j] = GetNSE(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    rv_lstm[i, j] = _rv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    fhv_lstm[i, j] = _fhv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    flv_lstm[i, j] = _flv(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0, 1]
                    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])

                    swdi_lstm[i, j] = 10 * (y_pred_lstm[:, i, j] - y_test_lstm[346, i, j]) / (y_test_lstm[346, i, j] + 1e-8)
        np.save(out_path_lstm + 'r2_' + cfg['modelname'] + '.npy', r2_lstm)
        np.save(out_path_lstm + 'KGE_' + cfg['modelname'] + '.npy', GetKGE_lstm)
        np.save(out_path_lstm + 'PCC_' + cfg['modelname'] + '.npy', GetPCC_lstm)
        np.save(out_path_lstm + 'NSE_' + cfg['modelname'] + '.npy', GetNSE_lstm)
        np.save(out_path_lstm + 'rv_' + cfg['modelname'] + '.npy', rv_lstm)
        np.save(out_path_lstm + 'fhv_' + cfg['modelname'] + '.npy', fhv_lstm)
        np.save(out_path_lstm + 'flv_' + cfg['modelname'] + '.npy', flv_lstm)
        np.save(out_path_lstm + 'r_' + cfg['modelname'] + '.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_' + cfg['modelname'] + '.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_' + cfg['modelname'] + '.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_' + cfg['modelname'] + '.npy', urmse_lstm)
        swdi_obs[swdi_obs > 0] = 1
        swdi_obs[(-2 < swdi_obs) & (swdi_obs <= 0)] = 2
        swdi_obs[(-5 < swdi_obs) & (swdi_obs <= -2)] = 3
        swdi_obs[(-10 < swdi_obs) & (swdi_obs <= -5)] = 4
        swdi_obs[swdi_obs <= -10] = 5

        swdi_lstm[swdi_lstm > 0] = 1
        swdi_lstm[(-2 < swdi_lstm) & (swdi_lstm <= 0)] = 2
        swdi_lstm[(-5 < swdi_lstm) & (swdi_lstm <= -2)] = 3
        swdi_lstm[(-10 < swdi_lstm) & (swdi_lstm <= -5)] = 4
        swdi_lstm[swdi_lstm <= -10] = 5

        np.save(out_path_lstm + 'swdi_obs' + '.npy', swdi_obs)
        np.save(out_path_lstm + 'swdi_' + cfg['modelname'] + '.npy', swdi_lstm)



if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)
