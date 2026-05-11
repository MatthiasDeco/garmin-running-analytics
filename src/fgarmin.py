import zipfile
from garminconnect import Garmin, GarminConnectConnectionError, GarminConnectTooManyRequestsError

def fit_activitys_download(garmin_mail, garmin_password, fit_carp_path):
    client = Garmin(garmin_mail, garmin_password)
    try:
        client.login()        
        offset, limit = 0, 1000
        while True:
            # If activitas dowload then finish
            activities = client.get_activities(offset, limit)
            if not activities:
                print("===== Activities downloaded =====")
                break
            
            # Dowload or skipping (if exists) each activity
            for act in activities:
                activity_id = act.get("activityId")
                fit_file = fit_carp_path / f"{activity_id}.fit"

                if fit_file.exists():
                    continue

                zip_data = client.download_activity(activity_id, dl_fmt=client.ActivityDownloadFormat.ORIGINAL)
                temp_zip = fit_carp_path / f"{activity_id}.zip"

                with open(temp_zip, "wb") as f:
                    f.write(zip_data)

                with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                    for name in zip_ref.namelist():
                        if name.endswith(".fit"):
                            zip_ref.extract(name, fit_carp_path)
                            extracted = fit_carp_path / name
                            if extracted != fit_file:
                                extracted.rename(fit_file)
                            print(f"Dowloading: {fit_file.name}")

                temp_zip.unlink()
            offset += limit
    except (GarminConnectConnectionError, GarminConnectTooManyRequestsError) as e:
        print(f"Garmin conection error: {e}")

