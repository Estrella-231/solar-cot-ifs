import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import netCDF4 as nc
from datetime import datetime
import glob
from tqdm import tqdm

# ============================================================
# 固定参数：256×256 裁剪区域（湖北+周边，等经纬度 4km 网格）
# ============================================================
R1, R2 = 1112, 1368
C1, C2 = 2077, 2333
# 地理范围：26.3°N ~ 36.5°N, 107.1°E ~ 117.3°E

CHANNEL_INFO = {
    "0":  "AGRI C01 reflectance (0.47um)",
    "1":  "AGRI C02 reflectance (0.65um)",
    "2":  "AGRI C03 reflectance (0.83um)",
    "3":  "AGRI C04 reflectance (1.38um)",
    "4":  "AGRI C05 reflectance (1.61um)",
    "5":  "AGRI C06 reflectance (2.22um)",
    "6":  "AGRI C07 brightness_temperature (3.72um)",
    "7":  "AGRI C08 brightness_temperature (6.25um)",
    "8":  "AGRI C09 brightness_temperature (7.10um)",
    "9":  "AGRI C10 brightness_temperature (7.80um)",
    "10": "AGRI C11 brightness_temperature (8.60um)",
    "11": "AGRI C12 brightness_temperature (10.3um)",
    "12": "AGRI C13 brightness_temperature (11.8um)",
    "13": "AGRI C14 brightness_temperature (12.8um)",
    "14": "AGRI C15 brightness_temperature (13.5um)",
    "15": "SAA sensor_azimuth_angle (fixed)",
    "16": "SAZ sensor_zenith_angle (fixed)",
    "17": "SOA solar_azimuth_angle (dynamic)",
    "18": "SOZ solar_zenith_angle (dynamic)",
    "19": "day_mask (1=day, 0=night)",
}


def format_timestamp(val):
    val_str = str(val)
    if len(val_str) >= 14 and val_str.isdigit():
        dt = datetime.strptime(val_str[:14], "%Y%m%d%H%M%S")
        return dt.strftime('%Y_%m%d_%H%M%S')
    else:
        return "invalid"


def detect_hdf_format(fdata):
    if 'Data' in fdata.groups and 'Calibration' in fdata.groups:
        return 'group', fdata['Data'], fdata['Calibration']
    else:
        return 'flat', fdata, fdata


# ============================================================
# 太阳角度计算（NOAA 算法）
# ============================================================

def _julian_day(dt):
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + h / 24.0


def compute_solar_angles(lat, lon, dt_utc):
    """
    返回:
        soz:      太阳天顶角 (度), 夜间保留 >90°
        soa:      太阳方位角 (度, 0°=北顺时针)
        day_mask: 白天=1, 夜间=0
    """
    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0

    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    gmas_rad = np.deg2rad(gmas)
    sec = (np.sin(gmas_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
           + np.sin(2 * gmas_rad) * (0.019993 - 0.000101 * jc)
           + np.sin(3 * gmas_rad) * 0.000289)
    stl = gmls + sec

    omega_m = 125.04 - 1934.136 * jc
    sal = stl - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega_m))
    moe = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    oc = moe + 0.00256 * np.cos(np.deg2rad(omega_m))
    oc_rad = np.deg2rad(oc)
    decl_rad = np.arcsin(np.sin(oc_rad) * np.sin(np.deg2rad(sal)))

    vary = np.tan(oc_rad / 2.0) ** 2
    gmls_rad = np.deg2rad(gmls)
    eot = 4.0 * (vary * np.sin(2 * gmls_rad)
                 - 2.0 * eeo * np.sin(gmas_rad)
                 + 4.0 * eeo * vary * np.sin(gmas_rad) * np.cos(2 * gmls_rad)
                 - 0.5 * vary * vary * np.sin(4 * gmls_rad)
                 - 1.25 * eeo * eeo * np.sin(2 * gmas_rad))

    utc_min = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    tst = (utc_min + eot + 4.0 * lon) % 1440.0
    ha_rad = np.deg2rad(tst / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat)
    cos_soz = (np.sin(lat_rad) * np.sin(decl_rad)
               + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    cos_soz = np.clip(cos_soz, -1.0, 1.0)
    soz = np.rad2deg(np.arccos(cos_soz))

    sin_soz = np.sqrt(np.maximum(1.0 - cos_soz ** 2, 1e-12))
    cos_soa = (np.sin(decl_rad) - np.sin(lat_rad) * cos_soz) / (np.cos(lat_rad) * sin_soz)
    cos_soa = np.clip(cos_soa, -1.0, 1.0)
    sin_soa = -np.cos(decl_rad) * np.sin(ha_rad) / sin_soz
    sin_soa = np.clip(sin_soa, -1.0, 1.0)
    soa = np.rad2deg(np.arctan2(sin_soa, cos_soa)) % 360.0

    day_mask = (soz < 90.0).astype(np.float32)

    return soz, soa, day_mask


# ============================================================
# 预加载固定数据
# ============================================================

agri_funclat_index_full = np.load(
    r'/home/Data_Pool/data/FY/FY4B/AGRI/LAT_4DEG_INDEX_105.npy')
agri_funclon_index_full = np.load(
    r'/home/Data_Pool/data/FY/FY4B/AGRI/LON_4DEG_INDEX_105.npy')
sub_lat_idx = agri_funclat_index_full[R1:R2, C1:C2].copy()
sub_lon_idx = agri_funclon_index_full[R1:R2, C1:C2].copy()
del agri_funclat_index_full, agri_funclon_index_full

Geo = nc.Dataset(
    r'/home/Data_Pool/data/FY/FY4B/AGRI/FY4B_AGRI_SAZ_SOZ_SAA_SOA_202403180000_4km.nc')
sub_lat = Geo['lat'][R1:R2, C1:C2].data.copy()
sub_lon = Geo['lon'][R1:R2, C1:C2].data.copy()
sub_saa = Geo['SAA'][R1:R2, C1:C2].data.copy()
sub_saz = Geo['SAZ'][R1:R2, C1:C2].data.copy()
Geo.close()


def process_fy4b_file(hdf_file_path, data_dir, skip_existing=False):
    """处理单个 FY4B 文件，返回 (success, err_msg_or_entry)"""
    filename = os.path.basename(hdf_file_path)

    obs_dt = None
    for part in filename.split('_'):
        if len(part) == 14 and part.isdigit():
            obs_dt = datetime.strptime(part, "%Y%m%d%H%M%S")
            break
    if obs_dt is None:
        return False, f"{filename}: 无法解析时间"

    year_str = obs_dt.strftime("%Y")
    month_str = obs_dt.strftime("%Y%m")
    date_str = obs_dt.strftime("%Y%m%d")
    time_str = obs_dt.strftime("%Y%m%d%H%M")
    out_subdir = os.path.join(data_dir, year_str, month_str, date_str)
    out_path = os.path.join(out_subdir, f"{time_str}.npy")
    npy_rel = os.path.join("data", year_str, month_str, date_str, f"{time_str}.npy")
    bjt_h = (obs_dt.hour + 8) % 24
    entry = {
        "time_utc": obs_dt.strftime("%Y-%m-%d %H:%M"),
        "time_bjt": obs_dt.strftime(f"%Y-%m-%d ") + f"{bjt_h:02d}:00",
        "npy_path": npy_rel,
        "source_hdf": filename,
    }

    if skip_existing and os.path.isfile(out_path):
        try:
            existing = np.load(out_path, mmap_mode="r")
            if (
                    existing.shape == (20, 256, 256)
                    and existing.dtype == np.dtype("float32")
                    and os.path.getmtime(out_path)
                    >= os.path.getmtime(hdf_file_path)
            ):
                return True, entry
        except (OSError, ValueError):
            pass

    # ---- 动态太阳角度 ----
    soz, soa, day_mask = compute_solar_angles(sub_lat, sub_lon, obs_dt)
    night = day_mask == 0

    # ---- 读取 AGRI ----
    fdata = nc.Dataset(hdf_file_path)
    fmt, ds, cs = detect_hdf_format(fdata)
    try:
        if fmt == 'group':
            channels = [ds[f'NOMChannel{str(i).zfill(2)}'][:].data for i in range(1, 16)]
            cal = [cs[f'CALChannel{str(i).zfill(2)}'][:].data for i in range(1, 16)]
        else:
            channels = [ds[f'NOMChannel{str(i).zfill(2)}'][:].data for i in range(1, 16)]
            cal = [cs[f'CalChannel{str(i).zfill(2)}'][:].data for i in range(1, 16)]
    finally:
        fdata.close()

    channels = np.array(channels, dtype=np.float32)
    cal = np.array(cal, dtype=np.float32)
    channels[(channels == 65534) | (channels == 65535)] = 4096
    channels = np.clip(channels, 0, 4096)  # 安全裁剪

    BT = np.empty((15, 2748, 2748), dtype=np.float32)
    for i in range(15):
        BT[i] = np.append(cal[i], 0)[channels[i].astype(np.int32)]

    func = BT[:, sub_lat_idx, sub_lon_idx]

    func[0:6, night] = 0.0
    soa_out = soa.copy()
    soa_out[night] = 0.0

    out = np.empty((20, 256, 256), dtype=np.float32)
    out[0:15] = func
    out[15] = sub_saa
    out[16] = sub_saz
    out[17] = soa_out
    out[18] = soz
    out[19] = day_mask
    os.makedirs(out_subdir, exist_ok=True)
    np.save(out_path, out)
    return True, entry


def _collect_files(base_dir, start_dt, end_dt):
    """收集文件并按月份分组"""
    monthly = {}  # {month_key: {date_key: [file_paths]}}
    missing = []
    for year_dir in sorted(glob.glob(os.path.join(base_dir, '20[2-9][0-9]'))):
        for date_dir in sorted(glob.glob(os.path.join(year_dir, '20[2-9][0-9][0-1][0-9][0-3][0-9]'))):
            date_str = os.path.basename(date_dir)
            try:
                cur_date = datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                continue
            if not (start_dt <= cur_date < end_dt):
                continue

            hdf = sorted(glob.glob(os.path.join(date_dir, 'FY4B*HDF*')))
            if hdf:
                month_key = date_str[:6]
                monthly.setdefault(month_key, {})[date_str] = hdf
            else:
                missing.append(date_str)
    return monthly, missing


def process_all_data(base_dir, output_dir, start_dt, end_dt, num_workers=4,
                     skip_existing=False):
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    data_dir = os.path.join(output_dir, "data")

    np.savez(os.path.join(output_dir, "grid_static.npz"),
             lat=sub_lat, lon=sub_lon,
             saa=sub_saa, saz=sub_saz,
             row_index=sub_lat_idx, col_index=sub_lon_idx)
    print(f"已保存: grid_static.npz")

    with open(os.path.join(output_dir, "channel_info.json"), 'w', encoding='utf-8') as f:
        json.dump(CHANNEL_INFO, f, ensure_ascii=False, indent=2)
    print(f"已保存: channel_info.json")

    monthly, missing = _collect_files(base_dir, start_dt, end_dt)
    if missing:
        print(f"\n警告: 以下日期无数据: {', '.join(missing[:10])}"
              + ("..." if len(missing) > 10 else ""))

    if not monthly:
        print("无文件需要处理")
        return

    all_entries = []
    total_ok, total_fail = 0, 0

    for month_key in sorted(monthly.keys()):
        dates = monthly[month_key]
        all_files = [(f, d) for d in sorted(dates.keys()) for f in dates[d]]
        # 期望的理论文件数：每天 96 个 (15min间隔)
        expected = len(dates) * 96
        print(f"\n{'='*50}")
        print(f"月份 {month_key}: {len(dates)} 天, {len(all_files)} 个文件 (期望 {expected})")
        print(f"{'='*50}")

        if len(all_files) < expected:
            pct = len(all_files) / expected * 100
            print(f"警告: 文件数仅为期望的 {pct:.0f}%，可能有缺失时次")

        month_ok, month_fail = 0, 0
        month_entries = []
        errors = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for hdf_file, date_str in all_files:
                fut = executor.submit(process_fy4b_file, hdf_file, data_dir,
                                      skip_existing)
                futures[fut] = hdf_file

            with tqdm(total=len(all_files), desc="  进度", unit="file",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                for fut in as_completed(futures):
                    success, result = fut.result()
                    if success:
                        month_ok += 1
                        month_entries.append(result)
                    else:
                        month_fail += 1
                        errors.append(result)
                    pbar.update(1)
                    pbar.set_postfix_str(f"OK:{month_ok} FAIL:{month_fail}")

        # 月度汇总
        print(f"\n--- {month_key} 汇总 ---")
        print(f"  期望时次: {expected}")
        print(f"  实际文件: {len(all_files)}")
        print(f"  成功: {month_ok}, 失败: {month_fail}")
        if errors:
            print(f"  失败详情:")
            for e in errors[:5]:
                print(f"    {e}")
        all_entries.extend(month_entries)
        total_ok += month_ok
        total_fail += month_fail

    # 保存 file_index.csv
    if all_entries:
        csv_path = os.path.join(output_dir, "file_index.csv")
        with open(csv_path, 'w') as f:
            f.write("time_utc,time_bjt,npy_path,source_hdf\n")
            for e in sorted(all_entries, key=lambda x: x["npy_path"]):
                f.write(f"{e['time_utc']},{e['time_bjt']}"
                        f",{e['npy_path']},{e['source_hdf']}\n")
        print(f"\n已保存: file_index.csv ({len(all_entries)} 条)")

    print(f"\n{'='*50}")
    print(f"全部完成! 成功 {total_ok}, 失败 {total_fail}, 总计 {total_ok+total_fail}")
    print(f"{'='*50}")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FY-4B AGRI 预处理 — 湖北 256×256")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD (不包含)")
    parser.add_argument("--data-dir", default="/home/Data_Pool/data/FY/FY4B/AGRI/4KM",
                        help="AGRI 数据根目录")
    parser.add_argument("--output-dir",
                        default="/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed",
                        help="预处理输出目录")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行线程数 (默认 4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过形状、类型和源文件时间均通过校验的既有 NPY")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y%m%d")
    end_dt = datetime.strptime(args.end, "%Y%m%d")

    print("=" * 60)
    print("FY-4B AGRI 预处理 — 湖北 256×256")
    print("=" * 60)
    print(f"  日期范围: {args.start} ~ {args.end}")
    print(f"  输入目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  并行线程: {args.workers}")
    print(f"  跳过有效既有文件: {args.skip_existing}")
    print(f"  输出: (20, 256, 256) float32")
    print("=" * 60)

    process_all_data(args.data_dir, args.output_dir, start_dt, end_dt,
                     num_workers=args.workers, skip_existing=args.skip_existing)
