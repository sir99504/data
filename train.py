import time
import tqdm
import sys
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn, load_train_data_for_rnn, \
    load_train_data_for_graph, \
    load_test_data_for_cnn, load_train_data_for_cnn, earth_data_transform, \
    sea_mask_rnn, sea_mask_cnn
from loss import NaNMSELoss
from model import LSTMModel, ConvLSTMModel, GraphLSTMModel, GraphConvLSTMModel


def train(x, y, static, mask, scaler_x, scaler_y, cfg, num_repeat, PATH, out_path, device, num_task=None, valid_split=True):
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split = cfg['valid_split']
    print('the device is {d}'.format(d=device))
    
    # ========== 添加：检查标签范围 ==========
    print("\n" + "="*60)
    print("jian cha biao qian shu ju fan wei")
    print("="*60)
    
    # zhi jian cha lu di qu yu de biao qian
    land_mask_expanded = np.expand_dims(mask, axis=0)  # [1, 45, 90]
    land_mask_expanded = np.expand_dims(land_mask_expanded, axis=-1)  # [1, 45, 90, 1]
    land_mask_expanded = np.repeat(land_mask_expanded, y.shape[0], axis=0)  # [1812, 45, 90, 1]
    
    # ti qu lu di biao qian
    land_labels = y[land_mask_expanded == 1]
    
    print(f"biao qian zong shu: {y.size}")
    print(f"lu di qu yu biao qian shu: {land_labels.shape[0]}")
    print(f"biao qian fan wei: zui xiao zhi={np.nanmin(land_labels):.6f}, zui da zhi={np.nanmax(land_labels):.6f}")
    print(f"biao qian jun zhi: {np.nanmean(land_labels):.6f}")
    
    # jian cha shi fou zai [0,1] fan wei nei
    if np.nanmin(land_labels) >= 0 and np.nanmax(land_labels) <= 1:
        print("? biao qian zai [0,1] fan wei nei, ke yi shi yong sigmoid ji huo han shu")
    else:
        print("? biao qian bu zai [0,1] fan wei nei, xu yao shi yong xian xing ji huo han shu")
    
    print("="*60 + "\n")
    # ========== jian cha jie shu ==========
    
    if cfg['modelname'] in ['GraphLSTM', 'ConvLSTM', 'GraphConvLSTM']:
        # Splice x according to the sphere shape
        lat_index, lon_index = earth_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    
    if valid_split:
        nt, nf, nlat, nlon = x.shape
        N = int(nt * cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]

    lossmse = torch.nn.MSELoss()
    print('\t\t\t\t\t\t\t\tx_train shape is', x.shape)
    print('\t\t\t\t\t\t\t\ty_train shape is', y.shape)
    print('\t\t\t\t\t\t\t\tstatic_train shape is', static.shape)
    print('\t\t\t\t\t\t\t\tmask shape is', mask.shape)

    if cfg['modelname'] in ['LSTM']:
        if valid_split:
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
    elif cfg['modelname'] in ['GraphLSTM', 'ConvLSTM', 'GraphConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)

    for num_ in range(cfg['num_repeat']):
        if cfg['modelname'] in ['LSTM']:
            model = LSTMModel(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)
        elif cfg['modelname'] in ['GraphLSTM']:
            model = GraphLSTMModel(cfg).to(device)
        elif cfg['modelname'] in ['GraphConvLSTM']:
            model = GraphConvLSTMModel(cfg).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

        for epoch in range(1, cfg['epochs'] + 1):
            t_begin = time.time()
            MSELoss = 0
            pbar = trange(1, cfg["niter"] + 1, desc=f'Epoch:{epoch} / {cfg["epochs"]}', file=sys.stdout)
            for number_iter in pbar:
                if cfg["modelname"] in ['LSTM']:
                    x_batch, y_batch, aux_batch, _, _ = load_train_data_for_rnn(cfg, x, y, static, scaler_y)
                    x_batch = torch.from_numpy(x_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = aux_batch.unsqueeze(1)
                    aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1)
                    x_batch = torch.cat([x_batch, aux_batch], 2)
                    pred = model(x_batch)
                    pred = torch.squeeze(pred, 1)
                    loss = lossmse(pred.float(), y_batch.float())
                    
                elif cfg['modelname'] in ['ConvLSTM']:
                    x_batch, y_batch, aux_batch, _, _ = load_train_data_for_cnn(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)
                    x_batch[np.isnan(x_batch)] = 0
                    x_batch = torch.from_numpy(x_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = aux_batch.unsqueeze(1)
                    aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1, 1, 1)
                    x_batch = x_batch.squeeze(dim=1)
                    x_batch = torch.cat([x_batch, aux_batch], 2).float()
                    pred = model(x_batch)
                    loss = lossmse(pred.float(), y_batch.float())
                    
                elif cfg['modelname'] in ['GraphConvLSTM']:
                    x_batch, y_batch, aux_batch, _, _ = load_train_data_for_graph(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)
                    x_batch[np.isnan(x_batch)] = 0
                    x_batch = torch.from_numpy(x_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = aux_batch.unsqueeze(1)
                    aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1, 1, 1)
                    x_batch = x_batch.squeeze(dim=1)
                    x_batch = torch.cat([x_batch, aux_batch], 2).float()
                    
                    pred = model(x_batch)
                    
                    batch_size = pred.shape[0]
                    y_batch_reshaped = y_batch.view(batch_size, 1, 7, 7)
                    
                    valid_mask = ~torch.isnan(y_batch_reshaped)
                    
                    if valid_mask.sum() > 0:
                        pred_valid = pred[valid_mask]
                        y_valid = y_batch_reshaped[valid_mask]
                        loss = lossmse(pred_valid.float(), y_valid.float())
                        
                        # 调试信息
                        if epoch == 1 and number_iter == 1:
                            print(f"DEBUG: pred range = [{pred.min().item():.6f}, {pred.max().item():.6f}]")
                            print(f"DEBUG: y_valid range = [{y_valid.min().item():.6f}, {y_valid.max().item():.6f}]")
                            print(f"DEBUG: loss = {loss.item():.6f}")
                            print(f"DEBUG: valid_mask count = {valid_mask.sum().item()}")
                    else:
                        loss = torch.tensor(0.0).to(device)
                    
                    # ========== 修改：这里只计算loss，不执行优化 ==========
                    # 优化步骤由外面的统一代码执行
                    
                        
                elif cfg['modelname'] in ['GraphLSTM']:
                    x_batch, y_batch, aux_batch, _, _ = load_train_data_for_graph(cfg, x, y, static, scaler_y, lat_index, lon_index, mask_index)
                    x_batch[np.isnan(x_batch)] = 0
                    x_batch = torch.from_numpy(x_batch).to(device)
                    aux_batch = torch.from_numpy(aux_batch).to(device)
                    y_batch = torch.from_numpy(y_batch).to(device)
                    aux_batch = aux_batch.unsqueeze(1)
                    aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1, 1, 1)
                    x_batch = x_batch.squeeze(dim=1)
                    x_batch = torch.cat([x_batch, aux_batch], 2).float()
                    pred = model(x_batch.reshape(x_batch.shape[0], x_batch.shape[1], -1))
                    loss = lossmse(pred.float(), y_batch.float())

                optim.zero_grad()
                loss.backward()
                optim.step()
                MSELoss += loss.item()
                pbar.set_postfix(loss='{:.3f}'.format(MSELoss / number_iter))
            
            t_end = time.time()

            if valid_split:
                del x_batch, y_batch, aux_batch
                MSE_valid_loss = 0
                if epoch % 20 == 0:
                    wait += 1
                    t_begin = time.time()
                    
                    if cfg["modelname"] in ['LSTM']:
                        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'], cfg["stride"])]
                        n = (x_valid.shape[0] - cfg["seq_len"]) // cfg["stride"]
                        for i in range(0, n):
                            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid, scaler_y, cfg["stride"], i, n)
                            x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                            y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                            aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                            aux_valid_batch = aux_valid_batch.unsqueeze(1)
                            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1)
                            x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                            with torch.no_grad():
                                pred_valid = model(x_valid_batch, aux_valid_batch)
                            mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch, lossmse)
                            MSE_valid_loss += mse_valid_loss.item()
                    
                    elif cfg['modelname'] in ['ConvLSTM']:
                        gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'] - cfg['forecast_time'], cfg["stride"])]
                        for i in gt_list:
                            x_valid_batch, y_valid_batch, aux_valid_batch, _, _ = \
                                load_test_data_for_cnn(cfg, x_valid, y_valid, static_valid, scaler_y, gt_list,
                                                       lat_index, lon_index, i,
                                                       cfg["stride"])

                            x_valid_batch[np.isnan(x_valid_batch)] = 0
                            x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                            y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                            aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                            aux_valid_batch = aux_valid_batch.unsqueeze(1)
                            aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1, 1, 1)
                            x_valid_batch = x_valid_batch
                            with torch.no_grad():
                                pred_valid = model(x_valid_batch, aux_valid_batch, cfg)
                            mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid, y_valid_batch, lossmse)
                            MSE_valid_loss += mse_valid_loss.item()

                    t_end = time.time()
                    mse_valid_loss = MSE_valid_loss / (len(gt_list))
                    
                    loss_str = '\033[1;31m%s\033[0m' % \
                               "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch, mse_valid_loss,
                                                                                  t_end - t_begin)
                    print(loss_str)
                    val_save_acc = mse_valid_loss

                    if val_save_acc < best:
                        torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                        wait = 0
                        best = val_save_acc
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
            else:
                if MSELoss < best:
                    best = MSELoss
                    wait = 0
                    torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                    tqdm.tqdm.write('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model With Loss {MSELoss / cfg["niter"]}')
            
            if wait >= patience:
                return
        return