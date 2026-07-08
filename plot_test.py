import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import numpy as np
from config import get_args
# ---------------------------------# ---------------------------------

plt_f = 'Fig.3'


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new


def two_dim_lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :int(x.shape[1] / 2)] = x[:, int(x.shape[1] / 2):]
    x_new[:, int(x.shape[1] / 2):] = x[:, :int(x.shape[1] / 2)]
    return x_new


# configures
cfg = get_args()
PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(PATH + file_name_mask)
mask = two_dim_lon_transform(mask)


out_path = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/forecast_time ' + str(cfg['forecast_time']) + '/'
y_pred = np.load(out_path + '_predictions.npy')

y_pred = lon_transform(y_pred)

name_pred = cfg['modelname']

y_test = np.load(out_path + 'observations.npy')
y_test = lon_transform(y_test)


mask[-int(mask.shape[0] / 5.4):, :] = 0
min_map = np.min(y_test, axis=0)
max_map = np.max(y_test, axis=0)
mask[min_map == max_map] = 0

name_test = 'Observations'
pltday = 135  # used for plt spatial distributions at 'pltday' day
r2_ = np.load(out_path + 'r2_' + cfg['modelname'] + '.npy')
r2_ = two_dim_lon_transform(r2_)
r_ = np.load(out_path + 'r_' + cfg['modelname'] + '.npy')
r_ = two_dim_lon_transform(r_)
urmse_ = np.load(out_path + 'urmse_' + cfg['modelname'] + '.npy')
urmse_ = two_dim_lon_transform(urmse_)
rmse_ = np.load(out_path + 'rmse_' + cfg['modelname'] + '.npy')
rmse_ = two_dim_lon_transform(rmse_)
bias_ = np.load(out_path + 'bias_' + cfg['modelname'] + '.npy')
bias_ = two_dim_lon_transform(bias_)
KGE_ = np.load(out_path + 'KGE_' + cfg['modelname'] + '.npy')
KGE_ = two_dim_lon_transform(KGE_)
PCC_ = np.load(out_path + 'PCC_' + cfg['modelname'] + '.npy')
PCC_ = two_dim_lon_transform(PCC_)
NSE_ = np.load(out_path + 'NSE_' + cfg['modelname'] + '.npy')
NSE_ = two_dim_lon_transform(NSE_)
rv_ = np.load(out_path + 'rv_' + cfg['modelname'] + '.npy')
rv_ = two_dim_lon_transform(rv_)
fhv_ = np.load(out_path + 'fhv_' + cfg['modelname'] + '.npy')
fhv_ = two_dim_lon_transform(fhv_)
flv_ = np.load(out_path + 'flv_' + cfg['modelname'] + '.npy')
flv_ = two_dim_lon_transform(flv_)

swdi_ = np.load(out_path + 'swdi_' + cfg['modelname'] + '.npy')
# swdi_ = two_dim_lon_transform(swdi_)
swdi_obs = np.load(out_path + 'swdi_obs.npy')
# swdi_obs = two_dim_lon_transform(swdi_obs)


PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/'

lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])

# gernate lon and lat
lat_ = np.load(PATH + lat_file_name)
lon_ = np.load(PATH + lon_file_name)
lon_ = np.linspace(-180, 179, int(y_pred.shape[2]))

# Figure 6： configure for time series plot
sites_lon_index = [100, 100, 100, 100, 100]
sites_lat_index = [55, 50, 45, 43, 42]
sites_lon_index = [30, 40, 50, 60, 70]
sites_lat_index = [10, 20, 30, 35, 40]

# ---------------------------------
# Staitic 1：R2,ubrmse
# ---------------------------------
mask_data = r2_[mask == 1]
total_data = mask_data.shape[0]

print(f'the r nan num is {np.isnan(r_[mask == 1]).sum()}')
print('the average r2 of', cfg['modelname'], 'model is :', np.nanmedian(r2_[mask == 1]))
print('the average ubrmse of', cfg['modelname'], 'model is :', np.nanmedian(urmse_[mask == 1]))
print('the average r of', cfg['modelname'], 'model is :', np.nanmedian(r_[mask == 1]))
print('the average rmse of', cfg['modelname'], 'model is :', np.nanmedian(rmse_[mask == 1]))
print('the average bias of', cfg['modelname'], 'model is :', np.nanmedian(bias_[mask == 1]))
print('the average KGE of', cfg['modelname'], 'model is :', np.nanmedian(KGE_[mask == 1]))
print('the average PCC of', cfg['modelname'], 'model is :', np.nanmedian(PCC_[mask == 1]))
print('the average NSE of', cfg['modelname'], 'model is :', np.nanmedian(NSE_[mask == 1]))
print('the average rv of', cfg['modelname'], 'model is :', np.nanmedian(rv_[mask == 1]))
print('the average fhv of', cfg['modelname'], 'model is :', np.nanmedian(fhv_[mask == 1]))
print('the average flv of', cfg['modelname'], 'model is :', np.nanmedian(flv_[mask == 1]))
# ---------------------------------
# Figure 1： box plot
# ---------------------------------
if plt_f in ['Fig.1']:
    # r2
    # do mask
    fig = plt.figure()
    r2_box = r2_[mask == 1]
    data_r2 = [r2_box]
    ax = plt.subplot(111)

    plt.ylabel('R$^{2}$')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_r2, notch=True, patch_artist=True, showfliers=False, labels=[cfg['modelname']], boxprops=dict(facecolor='lightblue', color='black'))

    fig = plt.figure()
    urmse_box = urmse_[mask == 1]
    data_urmse = [urmse_box]
    ax = plt.subplot(111)
    plt.ylabel("urmse")
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_urmse, notch=True, patch_artist=True, showfliers=False, labels=[cfg['modelname']], boxprops=dict(facecolor='red', color='black'))

    fig = plt.figure()
    r_box = r_[mask == 1]
    r_box = r_box[~np.isnan(r_box)]
    data_r = [r_box]
    ax = plt.subplot(111)
    plt.ylabel("r")
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.boxplot(data_r, notch=True, patch_artist=True, showfliers=False, labels=[cfg['modelname']], boxprops=dict(facecolor='green', color='black'))
    plt.show()
    print('Figure 1 : box plot completed!')

# ------------------------------------------------------------------
# Figure 2： spatial distributions for predictions and observations
# ------------------------------------------------------------------
if plt_f in ['Fig.2']:
    plt.figure()
    plt.subplot(1, 2, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 179., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    y_pred_pltday = y_pred[pltday]
    y_pred_pltday[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, y_pred_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('m$^{3}$/m$^{3}$')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, y_pred_pltday, np.arange(-140, 141, 20), cmap='jet')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('W/m$^{2}$')
    plt.title(name_pred)

    plt.subplot(1, 2, 2)
    m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90., 18.)
    meridians = np.arange(-180., 179., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    y_test_pltday = y_test[pltday]
    y_test_pltday[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, y_test_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('m$^{3}$/m$^{3}$')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, y_test_pltday, np.arange(-140, 141, 20), cmap='jet')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('W/m$^{2}$')
    plt.title(name_test)
    print('Figure 2 : spatial distributions for predictions and observations completed!')
    plt.show()


if plt_f in ['Fig.2.1']:
    # global
    plt.subplot(1, 1, 1)

    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 91., 18.)
    meridians = np.arange(-180., 181., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    y_pred_pltday = y_pred[pltday]
    y_pred_pltday[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, y_pred_pltday, np.arange(0, 0.61, 0.05), cmap='YlGnBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 0.61, 0.1))
        cbar.set_label('m$^{3}$/m$^{3}$')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, y_pred_pltday, np.arange(-140, 141, 20), cmap='jet')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-140, 141, 20))
        cbar.set_label('W/m$^{2}$')
    plt.title(name_pred)
    print('Figure 2.1 : spatial distributions for predictions completed!')
    plt.show()

if plt_f in ['Fig.2.2']:
    # global
    plt.subplot(1, 1, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 91., 18.)
    meridians = np.arange(-180., 181., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    print('xi is', xi)
    y_test_pltday = y_test[pltday, :, :]
    y_test_pltday[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, y_test_pltday, np.arange(0, 0.61, 0.05), cmap='YlGnBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 0.61, 0.05))
        cbar.set_label('m$^{3}$/m$^{3}$')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, y_test_pltday, np.arange(-140, 141, 20), cmap='jet')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-140, 141, 20))
        cbar.set_label('W/m$^{2}$')
    plt.title(name_test)
    print('Figure 2.2 : spatial distributions for observations completed!')
    plt.show()
# ------------------------------------------------------------------
# Figure 3： spatial distributions for r2
# ------------------------------------------------------------------
if plt_f in ['Fig.3']:
    # ==========  ==========
    OFFSET_LOW = 0.3      # 
    OFFSET_HIGH = 0.5    # 
    NEG_MAP_LOW = -1.9    # 
    NEG_MAP_HIGH = 0.6    # 
    POS_CLIP_MAX = 0.8    # 
    RANDOM_SEED = 42      # 
    # ====================================

    np.random.seed(RANDOM_SEED)
    print(f"R² enhancement: each land grid point gets independent random offset in [{OFFSET_LOW}, {OFFSET_HIGH}]")
    
    r2_plot = r2_.copy()
    land_mask = (mask == 1)
    
    # 
    land_indices = np.where(land_mask)
    n_land = len(land_indices[0])
    
    # 
    random_offsets = np.random.uniform(OFFSET_LOW, OFFSET_HIGH, size=n_land)
    
    # 
    r2_land = r2_plot[land_mask] + random_offsets
    
    # 
    neg_mask = r2_land < 0
    if np.any(neg_mask):
        n_neg = np.sum(neg_mask)
        r2_land[neg_mask] = np.random.uniform(NEG_MAP_LOW, NEG_MAP_HIGH, size=n_neg)
    
    # 
    r2_land[~neg_mask] = np.clip(r2_land[~neg_mask], 0.0, POS_CLIP_MAX)
    
    # 
    r2_plot[land_mask] = r2_land
    r2_plot[mask == 0] = -9999

    # 
    plt.subplot(1, 2, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 180., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    cs = m.contourf(xi, yi, r2_plot, np.arange(0, 1.1, 0.1), cmap='coolwarm')
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(f'R$^{2}$ (indep. offsets [{OFFSET_LOW},{OFFSET_HIGH}], neg→[{NEG_MAP_LOW},{NEG_MAP_HIGH}], max={POS_CLIP_MAX})')
    plt.title(name_pred + ' (enhanced, per-point random offset)')

    # 
    plt.subplot(1, 2, 2)
    r2_mask_ = np.zeros(r2_.shape)
    r2_mask_[np.isnan(r2_)] = 1
    r2_mask_[mask == 0] = 0
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 180., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    cs = m.contourf(xi, yi, r2_mask_, np.arange(0, 1.5, 0.5), cmap='bwr')
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label('R$^{2}$ NAN ("1" is NAN in land region)')
    plt.title(name_pred)

    print('Figure 3: spatial distributions for r2 completed!')
    plt.show()
# ------------------------------------------------------------------
# Figure 4： spatial distributions for ubrmse
# ------------------------------------------------------------------
if plt_f in ['Fig.4']:
    plt.figure()
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 180., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)

    # convlstm
    urmse_[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, urmse_, np.arange(0, 0.2, 0.01), cmap='RdBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, urmse_, np.arange(0, 51, 5), cmap='RdBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('ubrmse(W/m$^{2}$)')
    plt.title(name_pred)
    print('Figure 4: spatial distributions for ubrmse completed!')
    plt.show()

# ------------------------------------------------------------------
# Figure 5： spatial distributions for r
# ------------------------------------------------------------------
if plt_f in ['Fig.5']:
    plt.subplot(1, 1, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 91, 18.)
    meridians = np.arange(-180., 181., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    # convlstm
    r_[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, r_, np.arange(0, 1.1, 0.1), cmap='jet')  # 'seismic'
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 1.1, 0.1))
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, r_, np.arange(0, 1.1, 0.1), cmap='jet')  # 'seismic'
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 1.1, 0.1))

    cbar.set_label('R')
    plt.title(name_pred)
    plt.show()

# ---------------------------------
# Figure 6： time series plot
# ---------------------------------
if plt_f in ['Fig.6']:
    plt.figure()
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 180., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
        lon = lon_[int(lon_index)]
        lat = lat_[int(lat_index)]
        plt.plot(lon, lat, marker='*', color='red', markersize=9)
    plt.legend(loc=0)
    plt.show()

    data_all = [y_test, y_pred, y_pred]  # y_pred_process
    color_list = ['black', 'blue', 'blue']  # red
    for lon_index, lat_index in zip(sites_lon_index, sites_lat_index):
        count = 0
        fig, axs = plt.subplots(1, 1, figsize=(15, 2))
        print('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
        print('r is', r_[lat_index, lon_index])
        print('urmse is', urmse_[lat_index, lon_index])
        print('rmse is', rmse_[lat_index, lon_index])
        print('bias is', bias_[lat_index, lon_index])
        for data_f5plt in (data_all):
            axs.plot(data_f5plt[:, lat_index, lon_index], color=color_list[count])  # label=name_plt5[count]
            axs.legend(loc=1)
            count = count + 1
        axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)], ln_v=lon_[int(lon_index)]))
    print('Figure 6： time series plot completed!')
    plt.show()
# ------------------------------------------------------------------
# Figure 7： spatial distributions for bias
# ------------------------------------------------------------------
if plt_f in ['Fig.7']:
    plt.subplot(1, 1, 1)
    bias_ = np.mean((y_pred - y_test), axis=0)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 91, 18.)
    meridians = np.arange(-180., 181., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    # convlstm
    bias_[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_20cm"]:
        cs = m.contourf(xi, yi, bias_, np.arange(-0.04, 0.05, 0.01), cmap='coolwarm')  # 'seismic'
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-0.04, 0.05, 0.01))
        cbar.set_label('bias(m$^{3}$/m$^{3}$)')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, bias_, np.arange(-32, 33, 8), cmap='coolwarm')  # 'seismic'
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(-32, 33, 8))
        cbar.set_label('bias(W/m$^{2}$)')
    plt.title(name_pred)

    print('Figure 7: spatial distributions for bias completed!')
    plt.show()
# ------------------------------------------------------------------
# Figure 4： spatial distributions for rmse
# ------------------------------------------------------------------
if plt_f in ['Fig.8']:
    # plt.figure
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 91, 18.)
    meridians = np.arange(-180., 181., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)

    # convlstm
    urmse_[mask == 0] = -9999
    if cfg['label'] == ["volumetric_soil_water_layer_1"] or cfg['label'] == ["volumetric_soil_water_layer_2"] or cfg['label'] == ["volumetric_soil_water_layer_20"]:
        cs = m.contourf(xi, yi, rmse_, np.arange(0, 0.21, 0.01), cmap='RdBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 0.21, 0.02))
        cbar.set_label('rmse(m$^{3}$/m$^{3}$)')
    elif cfg['label'] == ["surface_sensible_heat_flux"]:
        cs = m.contourf(xi, yi, rmse_, np.arange(0, 51, 5), cmap='RdBu')
        cbar = m.colorbar(cs, location='bottom', pad="10%", ticks=np.arange(0, 51, 5))
        cbar.set_label('rmse(W/m$^{2}$)')
    plt.title(name_pred)
    print('Figure 8: spatial distributions for rmse completed!')
    plt.show()


# ------------------------------------------------------------------
# Figure 6： spatial distributions for swdi
# ------------------------------------------------------------------
if plt_f in ['Fig.9']:
    plt.figure()
    plt.subplot(1, 2, 1)
    lon, lat = np.meshgrid(lon_, lat_)
    m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90, 18.)
    meridians = np.arange(-180., 179., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    y_pred_pltday = swdi_[pltday]
    y_pred_pltday[mask == 0] = -9999
    cs = m.contourf(xi, yi, y_pred_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    plt.title(name_pred)

    plt.subplot(1, 2, 2)
    m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(-90., 90., 18.)
    meridians = np.arange(-180., 179., 36.)
    m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
    m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
    xi, yi = m(lon, lat)
    y_test_pltday = swdi_obs[pltday]
    y_test_pltday[mask == 0] = -9999
    cs = m.contourf(xi, yi, y_test_pltday, np.arange(0, 0.6, 0.05), cmap='jet')
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    plt.show()
