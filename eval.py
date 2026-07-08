import numpy as np
import torch
from utils import r2_score, reverse_normalize
import time
from data_gen import earth_data_transform
import sys
from data import Dataset


def batcher_lstm(x_test, y_test, aux_test, seq_len, forecast_time):
    n_t, n_feat = x_test.shape
    n = n_t - seq_len - forecast_time
    x_new = np.zeros((n, seq_len, n_feat)) * np.nan
    y_new = np.zeros((n, 1)) * np.nan
    aux_new = np.zeros((n, aux_test.shape[0])) * np.nan
    for i in range(n):
        x_new[i] = x_test[i: i + seq_len]
        y_new[i] = y_test[i + seq_len + forecast_time]
        aux_new[i] = aux_test
    return x_new, y_new, aux_new


def batcher_graph(
    x_test,
    y_test,
    aux_test,
    seq_len,
    forecast_time,
    spatial_offset,
    i,
    j,
    lat_index,
    lon_index,
):
    x_test = x_test.transpose(0, 3, 1, 2)
    y_test = y_test.transpose(0, 3, 1, 2)
    aux_test = aux_test.transpose(2, 0, 1)
    n_t, n_feat, n_lat, n_lon = x_test.shape

    n = n_t - seq_len - forecast_time
    x_new = np.zeros((n, seq_len, n_feat, 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan
    y_new = np.zeros((n, 1, 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan
    aux_new = (np.zeros((n, aux_test.shape[0], 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan)
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni: ni + seq_len, :, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :,][:, :, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1], ]
        y_new[ni] = y_test[ni + seq_len + forecast_time][:, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1]][:, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1]]
        aux_new[ni] = aux_test[:, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :, ][:, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1],]
    y_new = y_new.reshape(y_new.shape[0], -1)
    x_new = np.nan_to_num(x_new)
    y_new = np.nan_to_num(y_new)
    aux_new = np.nan_to_num(aux_new)
    return x_new, y_new, aux_new


def batcher_convlstm(
    x_test,
    y_test,
    aux_test,
    seq_len,
    forecast_time,
    spatial_offset,
    i,
    j,
    lat_index,
    lon_index,
):
    x_test = x_test.transpose(0, 3, 1, 2)
    y_test = y_test.transpose(0, 3, 1, 2)
    aux_test = aux_test.transpose(2, 0, 1)
    n_t, n_feat, n_lat, n_lon = x_test.shape

    n = n_t - seq_len - forecast_time
    x_new = np.zeros((n, seq_len, n_feat, 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan
    y_new = np.zeros((n, 1)) * np.nan
    aux_new = np.zeros((n, aux_test.shape[0], 2 * spatial_offset + 1, 2 * spatial_offset + 1)) * np.nan

    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni: ni + seq_len, :, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :][:, :, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1]]
        y_new[ni] = y_test[ni + seq_len + forecast_time, :, i, j]
        aux_new[ni] = aux_test[:, lat_index[lat_index_bias - spatial_offset: lat_index_bias + spatial_offset + 1], :][:, :, lon_index[lon_index_bias - spatial_offset: lon_index_bias + spatial_offset + 1]]
    return x_new, y_new, aux_new


def test(x, y, static, scaler_y, cfg, model, device):
    print("=== eval.py test han shu kai shi ===")
    
    # 检查模型参数
    print("=== jian cha mo xing can shu ===")
    nan_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if torch.isnan(param).any():
            print(f"{name}: you NaN")
            nan_count += 1
        else:
            print(f"{name}: wu NaN")
    print(f"mo xing can shu zhong, you {nan_count}/{total_params} ge can shu han you NaN")
    
    # 打印scaler_y的具体值
    print(f"jie shou dao de scaler_y xing zhuang: {scaler_y.shape}")
    print(f"jie shou dao de scaler_y zhi: {scaler_y}")
    print(f"scaler_y[0] (min): {scaler_y[0]}")
    print(f"scaler_y[1] (max): {scaler_y[1]}")
    print(f"normalize_type: {cfg['normalize_type']}")
    
    if len(scaler_y.shape) == 2:
        print(f"Global scaler - min: {scaler_y[0, :5]}, max: {scaler_y[1, :5]}")
    elif len(scaler_y.shape) == 4:
        print(f"Region scaler - lat0 lon0 min: {scaler_y[0, 0, 0, :]}, max: {scaler_y[1, 0, 0, :]}")
    
    cls = Dataset(cfg)
    model.eval()
    
    if cfg["modelname"] in ["GraphLSTM", "ConvLSTM", "GraphConvLSTM"]:
        lat_index, lon_index = earth_data_transform(cfg, x)
        print("\033[1;31m%s\033[0m" % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg["modelname"]))
    
    # 初始化预测结果数组
    y_pred_ens = np.zeros((y.shape[0] - cfg["seq_len"] - cfg["forecast_time"], y.shape[1], y.shape[2])) * np.nan
    # 提取真实标签（未反归一化的原始归一化值）
    y_true = y[cfg["seq_len"] + cfg["forecast_time"]:, :, :, 0]
    
    print(f"y_true xing zhuang: {y_true.shape}")
    print(f"y_true zhong de NaN shu liang: {np.isnan(y_true).sum()}")
    print(f"y_true fei NaN de zhi fan wei: [{np.nanmin(y_true):.6f}, {np.nanmax(y_true):.6f}]")
    print("the true label shape is: {ts} and the predicton shape is: {ps}".format(ts=y_true.shape, ps=y_pred_ens.shape))
    
    mask = y_true == y_true
    t_begin = time.time()

    # ------------------------------------------------------------------------------------------------------------------------------
    # LSTM模型预测逻辑
    if cfg["modelname"] in ["LSTM"]:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_lstm(x[:, i, j, :], y[:, i, j, :], static[i, j, :], cfg["seq_len"], cfg["forecast_time"],)

                x_new, static_new = torch.from_numpy(x_new).to(device), torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1).repeat(1, x_new.shape[1], 1)
                x_new = torch.cat([x_new, static_new], 2).to(device)
                pred = model(x_new).squeeze()
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                
                # 预测值反归一化
                if cfg["normalize"]:
                    if cfg["normalize_type"] in ["region"]:
                        pred = reverse_normalize(pred, scaler_y[:, i, j, 0], "minmax")
                    elif cfg["normalize_type"] in ["global"]:
                        pred = reverse_normalize(pred, scaler_y[:, 0], "minmax")
                
                y_pred_ens[:, i, j] = pred
                if count % 1000 == 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print("\r", end="")
                    print("Remain {fs} thound predictions".format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

    # ------------------------------------------------------------------------------------------------------------------------------
    # GraphLSTM模型预测逻辑
    elif cfg["modelname"] in ["GraphLSTM"]:
        count = 1
        for lat in range(x.shape[1]):
            for lon in range(x.shape[2]):
                lat_index_bias = lat + cfg['spatial_offset']
                lon_index_bias = lon + cfg['spatial_offset']
                x_new, y_new, static_new = batcher_graph(
                    x,
                    y,
                    static,
                    cfg["seq_len"],
                    cfg["forecast_time"],
                    cfg["spatial_offset"],
                    lat,
                    lon,
                    lat_index,
                    lon_index,
                )
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1)
                static_new = static_new.repeat(1, x_new.shape[1], 1, 1, 1)
                x_new = torch.cat([x_new, static_new], 2)
                x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], -1)
                pred = model(x_new).squeeze()
                pred = pred.cpu().detach().numpy()
                
                # 预测值反归一化
                if cfg["normalize"]:
                    if cfg["normalize_type"] in ["region"]:
                        pred = reverse_normalize(pred, scaler_y[:, lat, lon, 0], "minmax")
                    elif cfg["normalize_type"] in ["global"]:
                        pred = reverse_normalize(pred, scaler_y[:, 0], "minmax")

                lat_indices = lat_index[lat_index_bias - cfg['spatial_offset']:lat_index_bias + cfg['spatial_offset'] + 1]
                lon_indices = lon_index[lon_index_bias - cfg['spatial_offset']:lon_index_bias + cfg['spatial_offset'] + 1]
                pred_reshaped = pred.reshape(pred.shape[0], 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)
                y_pred_ens[:, lat_indices[:, None], lon_indices] = pred_reshaped
                
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print("\r", end="")
                    print("Remain {fs} thound predictions".format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

    # ------------------------------------------------------------------------------------------------------------------------------
    # GraphConvLSTM模型预测逻辑
    elif cfg["modelname"] in ["GraphConvLSTM"]:
        count = 1
        for lat in range(x.shape[1]):
            for lon in range(x.shape[2]):
                lat_index_bias = lat + cfg['spatial_offset']
                lon_index_bias = lon + cfg['spatial_offset']
                x_new, y_new, static_new = batcher_graph(
                    x,
                    y,
                    static,
                    cfg["seq_len"],
                    cfg["forecast_time"],
                    cfg["spatial_offset"],
                    lat,
                    lon,
                    lat_index,
                    lon_index,
                )
                
                x_new_tensor = torch.from_numpy(x_new).to(device)
                if count <= 5:  # 只检查前5个格点
                    print(f"di {count} ge ge dian, shu ru shu ju NaN shu liang: {np.isnan(x_new).sum()}")
                    print(f"di {count} ge ge dian, shu ru shu ju fan wei: [{np.nanmin(x_new):.6f}, {np.nanmax(x_new):.6f}]")
                
                static_new_tensor = torch.from_numpy(static_new).to(device)
                static_new_tensor = static_new_tensor.unsqueeze(1)
                static_new_tensor = static_new_tensor.repeat(1, x_new_tensor.shape[1], 1, 1, 1)
                x_new_tensor = torch.cat([x_new_tensor, static_new_tensor], 2)
                
                # 模型预测
                pred = model(x_new_tensor.float())
                pred_numpy = pred.cpu().detach().numpy()
                
                if count <= 5:  # 只检查前5个格点
                    print(f"di {count} ge ge dian, mo xing shu chu NaN shu liang: {np.isnan(pred_numpy).sum()}")
                    print(f"di {count} ge ge dian, mo xing shu chu fan wei: [{np.nanmin(pred_numpy):.6f}, {np.nanmax(pred_numpy):.6f}]")
                    print(f"di {count} ge ge dian tiao shi xin xi ru shang")
                
                # 预测值反归一化
                if cfg["normalize"]:
                    pred_numpy = reverse_normalize(pred_numpy, scaler_y[:, 0], "minmax")
                
                # 重塑预测结果并赋值
                lat_indices = lat_index[lat_index_bias - cfg['spatial_offset']:lat_index_bias + cfg['spatial_offset'] + 1]
                lon_indices = lon_index[lon_index_bias - cfg['spatial_offset']:lon_index_bias + cfg['spatial_offset'] + 1]
            
                # 确保索引是整数
                lat_indices = lat_indices.astype(int)
                lon_indices = lon_indices.astype(int)
                pred_reshaped = pred_numpy.reshape(pred_numpy.shape[0], 2 * cfg['spatial_offset'] + 1, 2 * cfg['spatial_offset'] + 1)
                y_pred_ens[:, lat_indices[:, None], lon_indices] = pred_reshaped

                # 进度打印
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print("\r", end="")
                    print("Remain {fs} thound predictions".format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

    # ------------------------------------------------------------------------------------------------------------------------------
    # ConvLSTM模型预测逻辑
    elif cfg["modelname"] in ["ConvLSTM"]:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(
                    x,
                    y,
                    static,
                    cfg["seq_len"],
                    cfg["forecast_time"],
                    cfg["spatial_offset"],
                    i,
                    j,
                    lat_index,
                    lon_index,
                )
                x_new = np.nan_to_num(x_new)
                static_new = np.nan_to_num(static_new)
                x_new_tensor = torch.from_numpy(x_new).to(device)
                static_new_tensor = torch.from_numpy(static_new).to(device)
                static_new_tensor = static_new_tensor.unsqueeze(1)
                static_new_tensor = static_new_tensor.repeat(1, x_new_tensor.shape[1], 1, 1, 1)
                x_new_tensor = torch.cat([x_new_tensor, static_new_tensor], 2).float()
                pred = model(x_new_tensor)
                pred_numpy = pred.cpu().detach().numpy()
                pred_numpy = np.squeeze(pred_numpy)
                
                # 预测值反归一化
                if cfg["normalize"]:
                    if cfg["normalize_type"] in ["region"]:
                        pred_numpy = reverse_normalize(pred_numpy, scaler_y[:, i, j, 0], "minmax")
                    elif cfg["normalize_type"] in ["global"]:
                        pred_numpy = reverse_normalize(pred_numpy, scaler_y[:, 0], "minmax")
                
                y_pred_ens[:, i, j] = pred_numpy
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print("\r", end="")
                    print("Remain {fs} thound predictions".format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
    else:
        print(f"Unsupported model: {cfg['modelname']}")
        return None, None
    
    # ----------------------------------------------------------------------------------------------------------------------------
    t_end = time.time()
    print("y_pred_ens shape is", y_pred_ens.shape)
    print("scaler shape is", scaler_y.shape)

    # ===================== 真实值反归一化 =====================
    print(f"fan gui yi hua qian, y_true fan wei: [{np.nanmin(y_true):.6f}, {np.nanmax(y_true):.6f}]")
    if cfg["normalize"]:
        if cfg["normalize_type"] in ["global"]:
            # 全局归一化：给真实值整体反归一化
            y_true = reverse_normalize(y_true, scaler_y[:, 0], "minmax")
            print(f"zhen shi zhi fan gui yi hua hou fan wei: [{np.nanmin(y_true):.6f}, {np.nanmax(y_true):.6f}]")
        elif cfg["normalize_type"] in ["region"]:
            # 区域归一化：逐格点反归一化
            for i in range(y_true.shape[1]):
                for j in range(y_true.shape[2]):
                    y_true[:, i, j] = reverse_normalize(y_true[:, i, j], scaler_y[:, i, j, 0], "minmax")
            print(f"zhen shi zhi fan gui yi hua hou fan wei: [{np.nanmin(y_true):.6f}, {np.nanmax(y_true):.6f}]")
    # ========== 新增：从文件中加载mask ==========
    print("\n" + "="*60)
    print("zai lu di qu yu ji suan ping gu zhi biao")
    print("="*60)
    
    # 构造mask文件路径
    mask_path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/Mask with ' + str(cfg['spatial_resolution']) + ' spatial resolution.npy'
    print(f"mask wen jian lu jing: {mask_path}")
    
    try:
        mask = np.load(mask_path).astype(bool)
        print(f"mask xing zhuang: {mask.shape}")
        print(f"mask zhong lu di (1) shu liang: {(mask == 1).sum()}")
        print(f"mask zhong hai yang (0) shu liang: {(mask == 0).sum()}")
    except Exception as e:
        print(f"jia zai mask shi bai: {e}")
        mask = None
    
    # 如果成功加载mask，只计算陆地区域的指标
    if mask is not None:
        # 扩展mask维度以匹配y_true
        mask_expanded = np.expand_dims(mask, axis=0)  # [1, 纬度, 经度]
        mask_expanded = np.repeat(mask_expanded, y_true.shape[0], axis=0)  # [时间, 纬度, 经度]
        
        # 提取陆地区域的数据
        land_mask = mask_expanded == 1
        
        # 创建有效数据掩码（非NaN且是陆地）
        valid_mask = ~np.isnan(y_true) & land_mask
        
        y_true_land = y_true[valid_mask]
        y_pred_land = y_pred_ens[valid_mask]
        
        print(f"lu di you xiao shu ju shu liang: {len(y_true_land)}")
        print(f"y_true_land fan wei: [{np.nanmin(y_true_land):.6f}, {np.nanmax(y_true_land):.6f}]")
        print(f"y_pred_land fan wei: [{np.nanmin(y_pred_land):.6f}, {np.nanmax(y_pred_land):.6f}]")
        
        if len(y_true_land) > 0:
            # 计算陆地区域的R2
            ss_res = np.sum((y_true_land - y_pred_land) ** 2)
            ss_tot = np.sum((y_true_land - np.mean(y_true_land)) ** 2)
            r2_land = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            # 计算陆地区域的R
            if len(y_true_land) > 1:
                r_land = np.corrcoef(y_true_land, y_pred_land)[0, 1]
            else:
                r_land = np.nan
            
            print(f"\nLU DI QU YU PING GU JIE GUO:")
            print(f"  Lu di R2: {r2_land:.6f}")
            print(f"  Lu di R:  {r_land:.6f}")
        else:
            print("jing gao: mei you you xiao de lu di shu ju!")
            r2_land = np.nan
            r_land = np.nan
    else:
        print("wei jia zai dao mask, shi yong quan bu qu yu ji suan")
        r2_land = np.nan
        r_land = np.nan
    
    # 原来的全局指标计算（保留用于对比）
    # ===================== 全局指标计算（所有区域） =====================
    # 正确应用 mask：广播到时间维度
    mask_3d = np.broadcast_to(mask, y_true.shape)
    y_true_mask = y_true[mask_3d]
    y_pred_ens_mask = y_pred_ens[mask_3d]
    
    print("y_true_mask shape is : {ts}".format(ts=y_true_mask.shape))
    print("the true label shape is: {ts} and the predicton shape is: {ps}".format(ts=y_true.shape, ps=y_pred_ens.shape))

    # 计算统计信息
    print(f"y_pred_ens zhong de NaN shu liang: {np.isnan(y_pred_ens).sum()}")
    print(f"y_pred_ens fei NaN de zhi fan wei: [{np.nanmin(y_pred_ens):.6f}, {np.nanmax(y_pred_ens):.6f}]")
    print(f"y_pred_ens_mask fei NaN de zhi fan wei: [{np.nanmin(y_pred_ens_mask):.6f}, {np.nanmax(y_pred_ens_mask):.6f}]")
    
    # 计算R2
    r2_ens = r2_score(y_true_mask, y_pred_ens_mask)
    
    # 计算相关系数R（逐时间步）
    R = np.zeros(y_true.shape[0])
    valid_count = 0
    for i in range(y_true.shape[0]):
        obs = y_true[i, :, :][mask]      # 应用 mask
        pre = y_pred_ens[i, :, :][mask]
        msk = (obs == obs) & (pre == pre)
        valid_obs = obs[msk]
        valid_pre = pre[msk]
        if len(valid_obs) >= 2 and len(valid_pre) >= 2:
            try:
                R[i] = np.corrcoef(valid_obs, valid_pre)[0, 1]
                valid_count += 1
            except:
                R[i] = np.nan
        else:
            R[i] = np.nan

    print(f"you xiao de R ji suan shu: {valid_count}/{y_true.shape[0]}")
    if valid_count > 0:
        print(f"R fei NaN de zhi (qian 5 ge): {R[~np.isnan(R)][:5]}")
    
    print("\033[1;31m%s\033[0m" % "Median R2 {:.3f} time cost {:.2f}".format(np.nanmedian(r2_ens), t_end - t_begin))
    print("\033[1;31m%s\033[0m" % "Median R {:.3f} time cost {:.2f}".format(np.nanmedian(R), t_end - t_begin))
    
    return y_pred_ens, y_true