import numpy as np
from train import train
from eval import test
from data import Dataset
from config import get_args
import torch
import os
import random
import warnings
print("=== jian cha dao ru ===")
print("shi tu cong eval dao ru test han shu")

try:
    from eval import test
    print("dao ru test cheng gong")
except Exception as e:
    print(f"dao ru test shi bai: {e}")
    print(f"dao ru shi bai yuan yin: {type(e).__name__}")
    
    
    import os
    if os.path.exists("eval.py"):
        print("eval.py wen jian cun zai")
        with open("eval.py", "r", encoding="utf-8") as f:
            first_lines = f.readlines()[:10]
            print("eval.py qian 10 hang:")
            for i, line in enumerate(first_lines):
                print(f"{i}: {line.strip()}")
    else:
        print("eval.py wen jian bu cun zai")
warnings.filterwarnings("ignore")


def seed(seed=5201314):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(cfg):
    seed()

    device = torch.device(cfg['device']) if torch.cuda.is_available() else torch.device('cpu')

    print('Now we training {d_p} product in {sr} spatial resolution'.format(d_p=cfg['product'], sr=str(cfg['spatial_resolution'])))
    print('1 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Make & load inputs'.format(d_p=cfg['workname']))
    path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
    if not os.path.isdir(path):
        os.makedirs(path)
    if os.path.exists(path + '/x_train_norm.npy'):
        print(' [ATAI {d_p} work ] loading input data'.format(d_p=cfg['workname']))
        x_train_shape = np.load(path + 'x_train_norm_shape.npy', mmap_mode='r')
        x_train = np.memmap(path + 'x_train_norm.npy', dtype=cfg['data_type'], mode='r+', shape=(x_train_shape[0], x_train_shape[1], x_train_shape[2], x_train_shape[3]))
        x_test_shape = np.load(path + 'x_test_norm_shape.npy', mmap_mode='r')
        x_test = np.memmap(path + 'x_test_norm.npy', dtype=cfg['data_type'], mode='r+', shape=(x_test_shape[0], x_test_shape[1], x_test_shape[2], x_test_shape[3]))
        y_train = np.load(path + 'y_train_norm.npy', mmap_mode='r')
        y_test = np.load(path + 'y_test_norm.npy', mmap_mode='r')
        static = np.load(path + 'static_norm.npy')
        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
        mask = np.load(path + file_name_mask)

    else:
        print('[ATAI {d_p} work ] making input data'.format(d_p=cfg['workname']))
        cls = Dataset(cfg)
        x_train, y_train, x_test, y_test, static, lat, lon, mask = cls.fit(cfg)
    # load scaler for inverse
    print("=== jian cha scaler jia zai ===")

    print("=== xiuzheng scaler jia zai ===")

    # 
    scaler_x_path = path + 'scaler_x.npy'
    scaler_y_path = path + 'scaler_y.npy'

    print(f"scaler_x wenjian lujing: {scaler_x_path}")
    print(f"scaler_y wenjian lujing: {scaler_y_path}")

    # =========
    input_feat_count = len(cfg['forcing_list']) + len(cfg['land_surface_list'])   # 6+3=9
    output_feat_count = len(cfg['label'])   # 1

    print(f"Expected scaler_x shape: (2, {input_feat_count})")
    print(f"Expected scaler_y shape: (2, {output_feat_count})")

    # 
    scaler_x = np.memmap(scaler_x_path, dtype=cfg['data_type'], mode='r', shape=(2, input_feat_count))
    scaler_y = np.memmap(scaler_y_path, dtype=cfg['data_type'], mode='r', shape=(2, output_feat_count))

    # 
    scaler_x = np.array(scaler_x)
    scaler_y = np.array(scaler_y)
    # ========== ==========

    print(f"scaler_x jia zai hou xing zhuang: {scaler_x.shape}")
    print(f"scaler_y jia zai hou xing zhuang: {scaler_y.shape}")
    print(f"scaler_y jia zai hou zhi: {scaler_y}")

    # 
    if cfg['data_type'] == 'float32':
        scaler_x = scaler_x.astype(np.float32)
        scaler_y = scaler_y.astype(np.float32)
        print(f"scaler_y zhuanhuan float32 hou: {scaler_y}")

    #
    
        # ------------------------------------------------------------------------------------------------------------------------------
        print('2 step:-----------------------------------------------------------------------------------------------------------------')
        print('[ATAI {d_p} work ] Train & load {m_n} Model'.format(d_p=cfg['workname'], m_n=cfg['modelname']))
        print('[ATAI {d_p} work ] Wandb info'.format(d_p=cfg['workname']))
# ------------------------------------------------------------------------------------------------------------------------------
    # Model training
    out_path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/forecast_time ' + str(cfg['forecast_time']) + '/'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    if os.path.exists(out_path + cfg['modelname'] + '_para.pkl'):
        print('[ATAI {d_p} work ] loading trained model'.format(d_p=cfg['workname']))
        model = torch.load(
            out_path + cfg['modelname'] + '_para.pkl',
            weights_only=False
        )
    else:
        # train
        print('[ATAI {d_p} work ] training {m_n} model'.format(d_p=cfg['workname'], m_n=cfg['modelname']))
        for j in range(cfg["num_repeat"]):
            train(x_train, y_train, static, mask, scaler_x, scaler_y, cfg, j, path, out_path, device)
            model = torch.load(
                out_path + cfg['modelname'] + '_para.pkl',
                weights_only=False
            )
        print('[ATAI {d_p} work ] finish training {m_n} model'.format(d_p=cfg['workname'], m_n=cfg['modelname']))
    # ------------------------------------------------------------------------------------------------------------------------------
    print('3 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Make predictions by {m_n} Model'.format(d_p=cfg['workname'], m_n=cfg['modelname']))
# ------------------------------------------------------------------------------------------------------------------------------
    print('x_test shape :', x_test.shape)
    print('y_test shape :', y_test.shape)
    print('static shape :', static.shape)
    print('scaler_x shape is', scaler_x.shape)
    print('scaler_y shape is', scaler_y.shape)
    # Model testing
    print("=== diao yong test han shu qian yan zheng scaler ===")
    print(f"chuan di gei test de scaler_y xing zhuang: {scaler_y.shape}")
    print(f"chuan di gei test de scaler_y zhi: {scaler_y}")

    y_pred, y_test = test(x_test, y_test, static, scaler_y, cfg, model, device)
# ------------------------------------------------------------------------------------------------------------------------------
    # save predicted values and true values
    print('[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "plot_test" codes for detailed analyzing'.format(d_p=cfg['workname'], m_n=cfg['modelname']))
    np.save(out_path + '_predictions.npy', y_pred)
    np.save(out_path + 'observations.npy', y_test)


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)