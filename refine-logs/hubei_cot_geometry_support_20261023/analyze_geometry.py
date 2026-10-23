"""Height sensitivity only: no actual CTH, GHI labels, or model fitting."""
import csv,hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

P=Path(__file__).resolve().parent;R=P.parents[1];E=R/'refine-logs/hubei_history_cot_confirmation_20261022'
Q=R/'refine-logs/hubei_cot_conditional_information_20261023/inputs_attempt2'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
geom=module('solar',R/'refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/solar_geometry_noaa.py')
ray=module('rays',R/'experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/project_cth_shadows.py')
gridfile=E/'grid_static256.npz'
with np.load(gridfile,allow_pickle=False) as z:grid={k:z[k] for k in z.files}
centers={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)};rows=[];summary={};source=[]
for st,(cy,cx) in centers.items():
    file=Q/f'{st}.npz'
    # Deliberately do not materialize truth, clear or feature arrays in the archive.
    with np.load(file,allow_pickle=False) as z:targets=z['target_time'];inits=z['init_time'];leads=z['lead']
    assert all(datetime.strptime(str(i),'%Y%m%d%H%M')+timedelta(minutes=int(l))==datetime.strptime(str(t),'%Y%m%d%H%M') for i,t,l in zip(inits,targets,leads))
    times=sorted(set(targets));lat=float(grid['lat'][cy,cx]);lon=float(grid['lon'][cy,cx]);saa=float(grid['saa'][cy,cx]);saz=float(grid['saz'][cy,cx])
    la,lo=np.deg2rad([lat,lon]);east=np.array([-np.sin(lo),np.cos(lo),0.]);north=np.array([-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)]);up=np.cross(east,north)
    en=np.stack([east,north]);ground=ray.geodetic_to_ecef(np.array(lat),np.array(lon),0.)
    jr=ray.geodetic_to_ecef(grid['lat'][cy+1,cx],grid['lon'][cy+1,cx],0.)-ground
    jc=ray.geodetic_to_ecef(grid['lat'][cy,cx+1],grid['lon'][cy,cx+1],0.)-ground
    jac=np.column_stack([en@jr,en@jc])/1000.;assert abs(np.linalg.det(jac))>1
    view_enu=np.array([np.sin(np.deg2rad(saz))*np.sin(np.deg2rad(saa)),np.sin(np.deg2rad(saz))*np.cos(np.deg2rad(saa)),np.cos(np.deg2rad(saz))])
    view=view_enu[0]*east+view_enu[1]*north+view_enu[2]*up
    errors=[];critical=[];magnitudes=[]
    for stamp in times:
        t=datetime.strptime(str(stamp),'%Y%m%d%H%M');z,az,day=geom.compute_solar_angles(np.array(lat),np.array(lon),t-timedelta(hours=8));z=float(z);az=float(az)
        domain=bool(day>.5 and z<80)
        row=dict(station=st,target_time_BJT=t.isoformat(sep=' '),solar_zenith_deg=z,solar_azimuth_deg=az,view_zenith_deg=saz,view_azimuth_deg=saa,within_R_solar_domain=domain,
            east_km_per_km='',north_km_per_km='',row_pixels_per_km='',col_pixels_per_km='',critical_height_km='',WGS84_derivative_error_km_per_km='')
        if domain:
            sun_en=np.tan(np.deg2rad(z))*np.array([np.sin(np.deg2rad(az)),np.cos(np.deg2rad(az))])
            k=sun_en-view_enu[:2]/view_enu[2];pixel=np.linalg.solve(jac,k)
            hcrit=1.5/abs(pixel).max() if abs(pixel).max()>0 else np.inf
            # Independent ECEF ray-shell finite perturbation at 1 m; no observed height.
            sun=ray.sun_ecef_from_local_angles(np.array(lat),np.array(lon),np.array(90-z),np.array(az))
            cloud=ray.ellipsoid_intersection(ground,sun,1.)
            apparent=ray.ellipsoid_intersection(cloud,-view,0.)
            numerical=en@(apparent-ground)  # m displacement / 1 m perturbation == km/km.
            error=float(np.max(abs(numerical-k)));assert error<2e-4
            row.update(east_km_per_km=float(k[0]),north_km_per_km=float(k[1]),row_pixels_per_km=float(pixel[0]),col_pixels_per_km=float(pixel[1]),critical_height_km=float(hcrit),WGS84_derivative_error_km_per_km=error)
            errors.append(error);critical.append(hcrit);magnitudes.append(np.linalg.norm(k))
        rows.append(row)
    summary[st]=dict(forecast_rows=len(targets),unique_target_times=len(times),valid_geometry_times=len(critical),
        critical_height_km_percentiles=dict(zip(['min','p05','p25','p50','p75','p95','max'],np.percentile(critical,[0,5,25,50,75,95,100]).tolist())),
        horizontal_km_per_km_percentiles=np.percentile(magnitudes,[0,50,100]).tolist(),max_WGS84_derivative_error=max(errors),
        pixel_jacobian_km=jac.tolist(),station_grid_lat=lat,station_grid_lon=lon)
    source.append(dict(station=st,input_archive_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),loaded_fields=['init_time','target_time','lead']))
with (P/'geometry_sensitivity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'RESULTS.json').write_text(json.dumps(dict(state='HEIGHT_SENSITIVITY_ONLY_NOT_A_CORRECTION_RESULT',summary=summary,sources=source,
    grid_sha256=hashlib.sha256(gridfile.read_bytes()).hexdigest(),actual_CTH_loaded=False,GHI_labels_loaded=False,new_models=0,test_payloads_read=0),indent=2))
print(json.dumps(summary,indent=2))
