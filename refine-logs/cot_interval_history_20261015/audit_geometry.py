"""Reproduce the producer bug and check the corrected formula against NREL SPA."""
import ast, csv, hashlib, importlib.util, json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from solar_geometry_corrected import compute_solar_angles

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cot_interval_history_20261015/audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).with_name('source_snapshot')/'preprocess_hunan_256.py'
    text=src.read_text();tree=ast.parse(text)
    ns={'np':np,'datetime':datetime,'Tuple':tuple}
    exec('\n\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_julian_day','compute_solar_angles')),ns)
    path=ROOT/'experiments/monthly_R_station16_20261013_v1/results/t16_deterioration_forensics_20261014/python_deps/pvlib/spa.py'
    spec=importlib.util.spec_from_file_location('spa_audit',path);spa=importlib.util.module_from_spec(spec);spec.loader.exec_module(spa)
    prev=ROOT/'results/gate_clear_physics_20261015';z=np.load(prev/'source_agri_geometry.npz')
    records=json.loads((prev/'source_geometry_metadata.json').read_text())['records'];frames=[]
    for r in records:
        tag=r['array_tag'];ch=z[tag+'_channels']
        frames.append(dict(station=r['station'],utc=r['utc'],lat=z[tag+'_lat'],lon=z[tag+'_lon'],
            zen=ch[7],az=ch[6],viewaz=ch[4],offset=np.full(ch[7].shape,r['obs_time_offset_s'])))
    witness=OUT/'geometry_witness.npz'
    if witness.exists():
        w=np.load(witness)
        for r in csv.DictReader((OUT/'geometry_witness_rows.csv').open()):
            k=r['index'];ch=w[k+'_angles'];frames.append(dict(station=r['station'],utc=r['utc'],lat=w[k+'_lat'],lon=w[k+'_lon'],zen=ch[2],az=ch[1],viewaz=ch[0],offset=w[k+'_offset'],pack_geo=w[k+'_pack_geo'],aux_geo=w[k+'_aux_cos']))
    checks=[];old_zen=[];new_zen=[];old_az=[];new_az=[];producer=[]
    for r in frames:
        t=datetime.fromisoformat(r['utc']);la=r['lat'].astype(float);lo=r['lon'].astype(float)
        u=(t-datetime(1970,1,1)).total_seconds();ut=np.full(la.size,u)
        ref=spa.solar_position(ut,la.ravel(),lo.ravel(),0,0,12,69,.5667,1)
        scan=spa.solar_position(ut+r['offset'].ravel(),la.ravel(),lo.ravel(),0,0,12,69,.5667,1)
        oz,oa=ns['compute_solar_angles'](r['lat'],r['lon'],t);nz,na=compute_solar_angles(r['lat'],r['lon'],t)
        rz=ref[1].reshape(la.shape);ra=ref[4].reshape(la.shape)
        angular=lambda a,b:(a-b+180)%360-180
        pe=max(float(np.abs(oz-r['zen']).max()),float(np.abs(angular(oa,r['az'])).max()));producer.append(pe)
        ze=r['zen']-rz;zne=nz-rz;ae=angular(r['az'],ra);ane=angular(na,ra)
        old_zen.extend(ze.ravel());new_zen.extend(zne.ravel());old_az.extend(ae.ravel());new_az.extend(ane.ravel())
        c=np.cos(np.deg2rad(r['zen']));a=np.cos(np.deg2rad(r['az']-r['viewaz']));pk=r.get('pack_geo');aux=r.get('aux_geo')
        check=dict(station=r['station'],utc=r['utc'],pixels=la.size,producer_max_error_deg=pe,
            source_zenith_minus_SPA_nominal_mean_deg=float(ze.mean()),corrected_zenith_minus_SPA_nominal_mean_deg=float(zne.mean()),
            source_azimuth_minus_SPA_nominal_mean_deg=float(ae.mean()),corrected_azimuth_minus_SPA_nominal_mean_deg=float(ane.mean()),
            source_zenith_minus_SPA_scan_mean_deg=float((r['zen'].ravel()-scan[1]).mean()),equation_of_time_SPA_minutes=float(ref[5].mean()),
            scan_offset_seconds=float(r['offset'].mean()))
        if pk is not None:
            check['pack_vs_source_cosine_max_error']=float(np.max(np.abs(pk-np.stack([c,a]))))
            check['aux_vs_source_cosine_max_error']=float(np.max(np.abs(aux-np.stack([c,a]))))
        checks.append(check)
    assert max(producer)<1e-3, 'Retrieved producer does not reproduce cache'
    assert max(abs(np.array(new_zen)))<.05 and max(abs(np.array(new_az)))<.1
    j=dict(state='PASS_SOURCE_FORMULA_REPRODUCTION_AND_INDEPENDENT_SPA_CHECK',frames=len(frames),pixels=len(old_zen),
        source_file_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),producer_reproduction_max_deg=max(producer),
        original_zenith_SPA_nominal_max_abs_deg=float(np.max(np.abs(old_zen))),corrected_zenith_SPA_nominal_max_abs_deg=float(np.max(np.abs(new_zen))),
        original_azimuth_SPA_nominal_max_abs_deg=float(np.max(np.abs(old_az))),corrected_azimuth_SPA_nominal_max_abs_deg=float(np.max(np.abs(new_az))),
        identified_error='equation_of_time = 4 * trig_expression; trig_expression is radians, missing rad2deg before minutes conversion',
        correction='equation_of_time = 4 * rad2deg(trig_expression)',
        scope='GHI-only corrected geometry for new experiment; original caches, frozen R and SimVP are unchanged; no claim of downstream improvement yet',
        test_payloads_read=0,checks=checks)
    (OUT/'geometry_audit.json').write_text(json.dumps(j,indent=2))
    plt.rcParams.update({'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,old,new,name in [(axs[0],old_zen,new_zen,'Solar zenith'),(axs[1],old_az,new_az,'Solar azimuth')]:
        ax.hist(old,bins=35,alpha=.6,label='Saved producer minus SPA',color='#D55E00');ax.hist(new,bins=35,alpha=.75,label='EOT-corrected minus SPA',color='#0072B2');ax.axvline(0,color='k',lw=.7);ax.set_xlabel(name+' error (degree)');ax.set_ylabel('Pixel records');ax.legend(fontsize=8)
    fig.suptitle('Actual train/validation geometry: missing radians-to-degrees in equation of time')
    fig.savefig(OUT/'solar_geometry_correction.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in j.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
