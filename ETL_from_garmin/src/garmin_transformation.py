import os
import numpy as np
import pandas as pd
import pytz
from pathlib import Path
from fitparse import FitFile

from ETL_from_garmin.src import fsql

def _convert_fit_to_rawdata(filepath: str) -> pd.DataFrame:
    required_columns = ["timestamp", "position_lat", "position_long", "distance",
                        "enhanced_speed", "enhanced_altitude", "heart_rate", "cadence", "fractional_cadence"]

    fitfile = FitFile(filepath)
    raw_arrays = {col: [] for col in required_columns}

    for record in fitfile.get_messages("record"):
        row_dict = {}
        for data in record:
            name, val = data.name, data.value
            if name == "timestamp":
                val = pd.to_datetime(val)
            if name in required_columns:
                row_dict[name] = val
        if row_dict:
            for col in required_columns:
                raw_arrays[col].append(row_dict.get(col, np.nan))

    if not any(raw_arrays.values()):
        return pd.DataFrame()

    def clean_array(arr, is_coord=False, is_timestamp=False):
        if is_timestamp:
            s = pd.Series(arr)
        else:
            arr = np.array(arr, dtype=float)
            s = pd.Series(arr)
        if s.notna().any():
            s = s.interpolate(method="linear", limit_direction="both").bfill().ffill()
        arr_cleaned = s.to_numpy().copy()
        if is_coord:
            idx = np.where(~np.isnan(arr_cleaned))[0]
            if idx.size:
                arr_cleaned[:idx[0]] = arr_cleaned[idx[0]]
        if not is_timestamp and np.isnan(arr_cleaned).all():
            arr_cleaned = np.zeros_like(arr_cleaned)
        return arr_cleaned

    cleaned = {col: clean_array(raw_arrays[col],
                                is_coord=("lat" in col or "long" in col),
                                is_timestamp=(col == "timestamp"))
               for col in required_columns}

    df = pd.DataFrame(cleaned).set_index("timestamp")
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="s")
    df = df.reindex(full_idx).interpolate(method="time").bfill().ffill()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["elapsed"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    return df


def _raw_to_processed(df_raw: pd.DataFrame, file_id: str) -> pd.DataFrame:
    timestamp        = df_raw["timestamp"]
    position_lat     = df_raw["position_lat"] * 180 / (2**31)
    position_long    = df_raw["position_long"] * 180 / (2**31)
    distance         = df_raw["distance"]
    enhanced_speed   = df_raw["enhanced_speed"]
    enhanced_altitude = df_raw["enhanced_altitude"]
    heart_rate       = df_raw["heart_rate"]
    cadence          = df_raw["cadence"]
    fractional_cadence = df_raw["fractional_cadence"]

    # Time & distance
    timestamp_seconds   = (timestamp - timestamp.iloc[0]).dt.total_seconds()
    instant_time        = np.concatenate(([0], np.diff(timestamp_seconds) / 60))
    accumulated_time    = timestamp_seconds / 60
    instant_distance    = np.concatenate(([0], np.diff(distance)))
    accumulated_distance = distance.copy()

    # Level calc — VERY PERSONAL CRITERIA
    n_level            = 10
    margin_error_lower = 0.5
    margin_error_upper = 20
    beach_altitude     = 15
    beach_margin       = margin_error_lower / n_level * 2
    num_points         = len(enhanced_altitude)
    instant_level               = np.zeros(num_points)
    accumulated_positive_level  = np.zeros(num_points)
    accumulated_negative_level  = np.zeros(num_points)

    i = 0
    while i < num_points:
        if (i + n_level) <= num_points:
            delta = (enhanced_altitude[i + n_level - 1] - enhanced_altitude[i]) / n_level
            if enhanced_altitude[i] < beach_altitude and delta < beach_margin:
                delta = 0

            if (i + n_level) < num_points:
                diff_val = abs(enhanced_altitude[i] - enhanced_altitude[i + n_level])
                instant_level[i:i+n_level] = 0 if (diff_val <= margin_error_lower or diff_val > margin_error_lower * n_level) else delta
            else:
                instant_level[i:i+n_level] = delta

            cur_max = np.max(accumulated_positive_level[:i]) if i > 0 else 0
            cur_min = np.min(accumulated_negative_level[:i]) if i > 0 else 0

            if 0 < delta < margin_error_upper / n_level:
                accumulated_positive_level[i:i+n_level] = cur_max + n_level * delta
                accumulated_negative_level[i:i+n_level] = cur_min
            elif -margin_error_upper / n_level < delta < 0:
                accumulated_positive_level[i:i+n_level] = cur_max
                accumulated_negative_level[i:i+n_level] = cur_min + n_level * delta
            else:
                accumulated_positive_level[i:i+n_level] = cur_max
                accumulated_negative_level[i:i+n_level] = cur_min
            i += n_level
        else:
            instant_level[i:] = 0
            accumulated_positive_level[i:] = accumulated_positive_level[i-1] if i > 0 else 0
            accumulated_negative_level[i:] = accumulated_negative_level[i-1] if i > 0 else 0
            break

    # Activity type — VERY PERSONAL CRITERIA
    max_top100_speed    = np.nanmean(np.sort(enhanced_speed)[-100:])
    max_top100_altitude = np.nanmean(np.sort(enhanced_altitude)[-100:])
    total_time          = np.nanmax(accumulated_time)
    total_distance      = np.nanmax(accumulated_distance) / 1000
    mean_pace           = total_time / total_distance if total_distance > 0 else 0
    ratio_level         = np.nanmax(accumulated_positive_level) / total_distance if total_distance > 0 else 0

    if max_top100_speed > 10 and max_top100_altitude > 1000:
        activity_type = "Snow"
    elif mean_pace > 12 or total_distance < 3.5:
        if total_distance == 0:
            activity_type = "Gym"
        elif mean_pace > 30 and total_distance < 3:
            activity_type = "Climb"
        elif 12 <= mean_pace <= 25:
            activity_type = "Walk"
        else:
            activity_type = "Other"
    elif ratio_level < 10:
        activity_type = "Road"
    else:
        activity_type = "Trail"

    # Running economy
    with np.errstate(divide="ignore", invalid="ignore"):
        instant_pace = 60.0 / (enhanced_speed * 3.6)
    pace_limit = 8.5 if activity_type == "Road" else 20
    instant_pace[instant_pace > pace_limit] = 0

    instant_bpm      = heart_rate.copy()
    instant_bpmxpace = instant_bpm * instant_pace
    instant_bpmxpace[(instant_bpmxpace > 1500) | (instant_bpmxpace < 400)] = 0

    instant_cadence       = cadence * 2
    instant_split_cadence = fractional_cadence * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        instant_stride = 1000.0 / (instant_pace * instant_cadence)
    instant_stride[(instant_stride > 5) | (instant_stride < 0.2)] = 0

    # Energy (MET model)
    MET_paces  = np.array([8.1, 7.5, 7.1, 6.2, 5.6, 5.3, 5.0, 4.7, 4.3, 4.0, 3.7, 3.4, 3.1, 2.9, 2.7])
    MET_values = np.array([6, 8.3, 9, 9.8, 10.5, 11, 11.5, 11.8, 12.3, 12.8, 14.5, 16, 19, 19.8, 23])
    MET_model  = np.poly1d(np.polyfit(MET_paces, MET_values, 3))
    instant_MET = MET_model(instant_pace)
    instant_MET[instant_pace > 8.1]  = 6
    instant_MET[instant_pace <= 2.7] = 23
    mean_MET = np.nanmean(instant_MET[np.isfinite(instant_MET)])
    instant_MET[instant_pace < 2] = mean_MET
    instant_consum_per_kg = (instant_MET * 4186) * (instant_time / 60.0)

    # Timestamps
    tz_spain = pytz.timezone("Europe/Madrid")
    t_start  = df_raw["timestamp"].iloc[0].tz_localize("UTC").tz_convert(tz_spain)
    t_end    = df_raw["timestamp"].iloc[-1].tz_localize("UTC").tz_convert(tz_spain)
    activity_date = t_start.strftime("%d/%m/%Y")
    inicial_time  = t_start.strftime("%H:%M:%S")
    finish_time   = t_end.strftime("%H:%M:%S")

    return pd.DataFrame({
        "file_id": file_id, "activity_type": activity_type,
        "activity_date": activity_date, "inicial_time": inicial_time, "finish_time": finish_time,
        "timestamp_raw": timestamp, "distance_raw": distance,
        "position_lat_raw": df_raw["position_lat"], "position_long_raw": df_raw["position_long"],
        "enhanced_speed_raw": enhanced_speed, "enhanced_altitude_raw": enhanced_altitude,
        "heart_rate_raw": heart_rate, "cadence_raw": cadence, "fractional_cadence_raw": fractional_cadence,
        "position_lat": position_lat, "position_long": position_long,
        "instant_time": instant_time, "accumulated_time": accumulated_time,
        "instant_distance": instant_distance, "accumulated_distance": accumulated_distance,
        "instant_level": instant_level,
        "accumulated_positive_level": accumulated_positive_level,
        "accumulated_negative_level": accumulated_negative_level,
        "instant_pace": instant_pace, "instant_bpm": instant_bpm,
        "instant_bpmxpace": instant_bpmxpace,
        "instant_cadence": instant_cadence, "instant_split_cadence": instant_split_cadence,
        "instant_stride": instant_stride,
        "instant_MET": instant_MET, "instant_consum_per_kg": instant_consum_per_kg,
    })


# Summary
def _processed_to_stats(df: pd.DataFrame, FTP_bpm: int, FTP_rap: int, weight: int) -> pd.DataFrame:
    accumulated_time             = df["accumulated_time"]
    accumulated_distance         = df["accumulated_distance"]
    accumulated_positive_level   = df["accumulated_positive_level"]
    accumulated_negative_level   = df["accumulated_negative_level"]
    instant_pace                 = df["instant_pace"]
    instant_bpm                  = df["instant_bpm"]
    instant_cadence              = df["instant_cadence"]
    instant_stride               = df["instant_stride"]
    instant_consum_per_kg        = df["instant_consum_per_kg"]

    file_id       = df["file_id"].iloc[0]
    activity_date = df["activity_date"].iloc[0]
    inicial_time  = df["inicial_time"].iloc[0]
    finish_time   = df["finish_time"].iloc[0]
    activity_type = df["activity_type"].iloc[0]
    inicial_lat   = df["position_lat"].iloc[0]
    inicial_long  = df["position_long"].iloc[0]

    total_time     = np.max(accumulated_time)
    total_distance = np.max(accumulated_distance) / 1000
    total_pos_level = np.max(accumulated_positive_level)
    total_neg_level = np.min(accumulated_negative_level)
    equivalence_distance = total_distance + 0.792 * total_pos_level / 100

    valid_pace = instant_pace[instant_pace != 0]
    moving_average_pace = np.mean(valid_pace) if valid_pace.size > 0 else 0
    real_average_pace   = total_time / total_distance if total_distance else 0
    dist_eq_rap = total_distance + 5 * total_pos_level / 1000 - 2.5 * total_neg_level / 1000
    average_rap = total_time / dist_eq_rap if dist_eq_rap else 0

    valid_bpm   = instant_bpm[(instant_bpm != 0) & (~np.isnan(instant_bpm))]
    average_bpm = np.mean(valid_bpm) if valid_bpm.size > 0 else 0
    max_bpm     = np.max(instant_bpm)
    average_beatsxkm = average_bpm * moving_average_pace

    deviation_pace = (np.std(instant_pace) / np.mean(instant_pace)) * 100 if np.mean(instant_pace) else 0
    deviation_bpm  = (np.std(instant_bpm)  / np.mean(instant_bpm))  * 100 if np.mean(instant_bpm)  else 0

    average_cadence = np.mean(instant_cadence)
    average_stride  = np.mean(instant_stride)

    PI_ref = FTP_bpm * FTP_rap
    PI = (1.0 / (average_beatsxkm / PI_ref)) * 100 if PI_ref and average_beatsxkm else 0

    # VDOT
    vdot_final = 0
    if activity_type in ["Road", "Trail"]:
        vdot_table = pd.DataFrame({
            "VDOT": [30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62],
            "1500":   [8.5,8.033,7.617,7.233,6.9,6.583,6.317,6.05,5.817,5.5,5.4,5.217,5.033,4.883,4.733,4.583,4.45],
            "3000":   [17.933,16.983,16.15,15.383,14.683,14.05,13.467,12.917,12.433,11.75,11.55,11.15,10.783,10.45,10.133,9.833,9.55],
            "5000":   [30.667,29.083,27.65,26.367,25.2,24.133,23.15,22.25,21.417,20.3,19.95,19.283,18.667,18.083,17.55,17.05,16.567],
            "10000":  [63.767,60.433,57.433,54.733,52.283,50.05,48.017,46.15,44.417,42.067,41.35,39.983,38.7,37.517,36.4,35.367,34.383],
            "15000":  [98.233,93.117,88.5,84.333,80.55,77.1,73.933,71.033,68.367,64.733,63.6,61.483,59.5,57.65,55.917,54.3,52.783],
            "21100":  [141.067,133.817,127.267,121.317,115.917,110.983,106.45,102.283,98.45,93.2,91.583,88.517,85.667,83,80.5,78.15,75.57],
            "42200":  [289.283,274.983,262.05,250.317,239.583,229.75,220.717,212.383,204.65,194.1,190.817,184.6,178.783,173.333,168.233,163.417,158.9],
            "50000":  [359.89,342.1,326.01,311.41,298.06,285.83,274.59,264.22,254.6,241.48,237.39,229.66,222.42,215.64,209.29,203.30,197.68],
            "150000": [1079.67,1026.3,978.03,934.24,894.18,857.48,823.77,792.66,763.8,724.43,712.17,688.97,667.26,646.92,627.89,609.91,593.05],
        })
        dist_cols = [c for c in vdot_table.columns if c != "VDOT"]
        distances = np.array([float(c) for c in dist_cols])
        sort_idx  = np.argsort(distances)
        distances = distances[sort_idx]
        d_interp  = total_distance * 1000 if activity_type == "Road" else equivalence_distance * 1000
        predicted_times = np.array([
            np.interp(d_interp, distances, row[dist_cols].values.astype(float)[sort_idx])
            for _, row in vdot_table.iterrows()
        ])
        vdot_values = vdot_table["VDOT"].values.astype(float)
        order = np.argsort(predicted_times)
        vdot_final = np.interp(total_time, predicted_times[order], vdot_values[order])

    # Energy & power
    running_efficiency = 0.25
    average_power      = np.mean(instant_consum_per_kg) * weight * running_efficiency
    standardized_power = average_power * 170 / average_bpm if average_bpm else 0
    total_energy       = np.sum(instant_consum_per_kg) * weight * running_efficiency
    energy_km          = total_energy / total_distance if total_distance else 0

    # Training load
    if_pace = average_rap / FTP_rap if FTP_rap else 0
    if_bpm  = average_bpm / FTP_bpm if FTP_bpm else 0
    TSS_pace      = (total_time / 60) * (if_pace ** 2) * 100
    TSS_bpm       = (total_time / 60) * (if_bpm  ** 2) * 100
    intensity_factor = (if_pace + if_bpm) / 2
    training_load    = (TSS_pace + TSS_bpm) / 2 if activity_type in ["Road", "Trail"] else 0

    return pd.DataFrame([{
        "file_id": file_id, "activity_type": activity_type,
        "activity_date": activity_date, "inicial_time": inicial_time, "finish_time": finish_time,
        "total_time": total_time, "total_distance": total_distance, "equivalence_distance": equivalence_distance,
        "total_accumulated_positive_level": total_pos_level,
        "total_accumulated_negative_level": total_neg_level,
        "moving_average_pace": moving_average_pace, "real_average_pace": real_average_pace,
        "average_rap": average_rap, "deviation_pace": deviation_pace,
        "average_bpm": average_bpm, "max_bpm": max_bpm, "deviation_bpm": deviation_bpm,
        "average_beatsxkm": average_beatsxkm, "average_cadence": average_cadence, "average_stride": average_stride,
        "total_energy": total_energy, "average_power": average_power, "standardized_power": standardized_power,
        "training_load": training_load, "vdot": vdot_final, "PI": PI, "intensity_factor": intensity_factor,
        "energy_km": energy_km, "TSS_pace": TSS_pace, "TSS_bpm": TSS_bpm,
        "FTP_bpm": FTP_bpm, "FTP_rap": FTP_rap,
        "inicial_lat": inicial_lat, "inicial_long": inicial_long, "weight": weight,
    }])


def _processed_to_zones(df: pd.DataFrame, FTP_bpm: int, FTP_pace: float) -> pd.DataFrame:
    bpm_z1, bpm_z2 = FTP_bpm * 0.800, FTP_bpm * 0.885
    bpm_z3, bpm_z4 = FTP_bpm * 0.925, FTP_bpm * 1.000
    pace_z1, pace_z2 = FTP_pace / 0.775, FTP_pace / 0.877
    pace_z3, pace_z4 = FTP_pace / 0.943, FTP_pace / 1.000

    bpm_arr  = df["instant_bpm"].to_numpy()
    pace_arr = df["instant_pace"].to_numpy()

    def count_zones_bpm(arr):
        z1 = np.sum(arr <= bpm_z1)
        z2 = np.sum((arr > bpm_z1) & (arr <= bpm_z2))
        z3 = np.sum((arr > bpm_z2) & (arr <= bpm_z3))
        z4 = np.sum((arr > bpm_z3) & (arr <= bpm_z4))
        z5 = np.sum(arr > bpm_z4)
        return z1, z2, z3, z4, z5

    def count_zones_pace(arr):
        z1 = np.sum(arr >= pace_z1)
        z2 = np.sum((arr < pace_z1) & (arr >= pace_z2))
        z3 = np.sum((arr < pace_z2) & (arr >= pace_z3))
        z4 = np.sum((arr < pace_z3) & (arr >= pace_z4))
        z5 = np.sum(arr < pace_z4)
        return z1, z2, z3, z4, z5

    b1, b2, b3, b4, b5 = count_zones_bpm(bpm_arr)
    p1, p2, p3, p4, p5 = count_zones_pace(pace_arr)

    return pd.DataFrame([{
        "time_z1_bpm": b1, "time_z2_bpm": b2, "time_z3_bpm": b3, "time_z4_bpm": b4, "time_z5_bpm": b5,
        "time_z1_pace": p1, "time_z2_pace": p2, "time_z3_pace": p3, "time_z4_pace": p4, "time_z5_pace": p5,
        "limit_bpm_z1": bpm_z1, "limit_bpm_z2": bpm_z2, "limit_bpm_z3": bpm_z3, "limit_bpm_z4": bpm_z4,
        "limit_pace_z1": pace_z1, "limit_pace_z2": pace_z2, "limit_pace_z3": pace_z3, "limit_pace_z4": pace_z4,
    }])


def _processed_to_marks(df_stats: pd.DataFrame) -> pd.DataFrame:
    race_distances = {"3km_daily_mark": 3, "5km_daily_mark": 5, "10km_daily_mark": 10,
                      "21_1km_daily_mark": 21.1, "42_2km_daily_mark": 42.2}
    total_time     = df_stats.loc[0, "total_time"]
    total_distance = df_stats.loc[0, "total_distance"]
    marks = {k: round(d * total_time / total_distance, 2) if total_distance >= d else None
             for k, d in race_distances.items()}
    return pd.DataFrame([marks])


def _build_summary(df_processed: pd.DataFrame, FTP_bpm: int, FTP_pace: float, FTP_rap: int, weight: int) -> pd.DataFrame:
    df_stats = _processed_to_stats(df_processed, FTP_bpm, FTP_rap, weight)
    df_zones = _processed_to_zones(df_processed, FTP_bpm, FTP_pace)
    df_marks = _processed_to_marks(df_stats)
    df_summary = pd.concat([df_stats, df_zones, df_marks], axis=1)
    df_summary["activity_date"] = pd.to_datetime(df_summary["activity_date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    return df_summary


# Main
def fit_to_sqlite(fit_carp_path: Path, FTP_bpm: int, FTP_pace: float, FTP_rap: int, weight: int,
                  ddbb_sqlite: Path, processed_table: str, summary_table: str):

    # Collect local file_ids
    local_file_ids = {}
    for root, _, files in os.walk(fit_carp_path):
        for file in files:
            if not file.endswith(".fit"):
                continue
            filepath = os.path.join(root, file)
            try:
                fitfile = FitFile(filepath, data_processor=None)
                min_ts = None
                for record in fitfile.get_messages("record"):
                    for data in record:
                        if data.name == "timestamp":
                            ts = data.value
                            if min_ts is None or ts < min_ts:
                                min_ts = ts
                    if min_ts:
                        break
                if min_ts:
                    local_file_ids[filepath] = min_ts.strftime("%Y%m%d%H%M")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    sql_ids = fsql.get_existing_ids(ddbb_sqlite, summary_table)
    files_to_process = [fp for fp, fid in local_file_ids.items() if fid not in sql_ids]

    for filepath in files_to_process:
        file_id = local_file_ids[filepath]
        print(f"Processing: {file_id}")

        # Step 1: FIT → raw
        df_raw = _convert_fit_to_rawdata(filepath)
        if df_raw.empty:
            print(f"  No valid data, skipping.")
            continue

        # Step 2: raw → processed + upload
        df_processed = _raw_to_processed(df_raw, file_id)
        fsql.upload_df(df_processed, ddbb_sqlite, processed_table)

        # Step 3: processed → summary + upload
        df_summary = _build_summary(df_processed, FTP_bpm, FTP_pace, FTP_rap, weight)
        fsql.upload_df(df_summary, ddbb_sqlite, summary_table)

    print("===== Activities processed and uploaded =====")

