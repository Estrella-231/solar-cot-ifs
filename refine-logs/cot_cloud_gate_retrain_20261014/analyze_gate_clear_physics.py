"""Validation-only interval audit and bounded solar/cloud-shadow geometry diagnosis."""
import csv,json,importlib.util
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/gate_clear_physics_20261015'
PREV=ROOT/'results/gate_ghi_forensics_20261015'
COORD={'sili':(25.9478611111,112.347916667),'zhujia':(29.566666667,112.45)}

def main():
    z=dict(np.load(PREV/'validation_forensics.npz'));cases=json.loads((OUT/'cases.json').read_text());source=dict(np.load(OUT/'source_agri_geometry.npz'))
    meta=json.loads((OUT/'source_geometry_metadata.json').read_text());raw=json.loads((OUT/'raw_time_cpp_audit.json').read_text());cpp=dict(np.load(OUT/'raw_cpp_case_fields.npz'))
    original=json.loads((OUT/'original_workbook_5min.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((OUT/'val_labels_5min.csv').open(encoding='utf-8')));labels={(r['station'],r['target_time_utc']):r for r in rows};native={}
    checked=0;maxerr=0.;native_conflicts=0
    for row in rows:
        assert row['new_split']=='val'
        t=datetime.fromisoformat(row['target_time_bjt']);u=datetime.fromisoformat(row['target_time_utc']);assert t-u==timedelta(hours=8)
        tt=row['label_record_times_bjt'].split('|');vv=row['raw_ghi_5min_wm2'].split('|')
        assert len(tt)==len(vv)==3
        assert [datetime.fromisoformat(x) for x in tt]==[t-timedelta(minutes=m) for m in (10,5,0)]
        vals=[]
        for ti,vi in zip(tt,vv):
            try:v=float(vi)
            except ValueError:v=np.nan
            if np.isfinite(v):
                key=(row['station'],ti)
                if key in native and abs(native[key]-v)>1e-6:native_conflicts+=1
                native[key]=v
            vals.append(v)
        if np.isfinite(vals).all() and row['observed_ghi_15min_wm2']:
            checked+=1;maxerr=max(maxerr,abs(np.mean(vals)-float(row['observed_ghi_15min_wm2'])))
    assert native_conflicts==0 and maxerr<1e-6
    source_comp=[]
    observed_native=native.copy()
    for st,j in original.items():
        vals=[abs(native[(st,t)]-v) for t,v in j['values'].items() if v is not None and (st,t) in native]
        source_comp.append(dict(station=st,compared_5min_cells=len(vals),max_abs_difference=max(vals,default=None)))
        assert max(vals,default=0)<1e-6
        for t,v in j['values'].items():
            if v is not None and np.isfinite(v):observed_native[(st,t)]=v
    s=z['station'];y=z['ghi'];ri=z['refidx'];matched=ri>=0;complete=np.zeros(len(y),bool);center=np.zeros(len(y),bool)
    complete[matched]=(z['ref_valid'][ri[matched],7:10,7:10]&(z['ref_clp'][ri[matched],7:10,7:10]==0)).all((1,2))
    center[matched]=z['ref_valid'][ri[matched],8,8]&(z['ref_clp'][ri[matched],8,8]==0)
    end5=np.full(len(y),np.nan);scan5=end5.copy();spread=end5.copy();scan_end5={}
    for i,(st,utc) in enumerate(zip(s,z['utc'])):
        r=labels[(str(st),str(utc))];v=np.array([float(t) for t in r['raw_ghi_5min_wm2'].split('|')]);end5[i]=v[-1];spread[i]=v.max()-v.min()
        t=datetime.fromisoformat(r['target_time_bjt']);scan5[i]=native.get((str(st),(t+timedelta(minutes=5)).isoformat(sep=' ')),np.nan)
        # These stations scan ~3 min after nominal; +5 min is the native interval containing the scan, not an instantaneous sample.
        assert abs(np.mean(v)-y[i])<.001
    benefit=(z['gatehead_gateinput']-y)**2-(z['gatehead_restore_CPPclear']-y)**2
    strata={};concentration={}
    for st in ('sili','zhujia'):
        m=(s==st)&complete
        strata[st]={}
        mc=(s==st)&matched;bjt_days=(z['utc'].astype('datetime64[s]')+np.timedelta64(8,'h')).astype('datetime64[D]')
        daily=sorted([dict(date=str(day),benefit_sse=float(benefit[mc&(bjt_days==day)].sum())) for day in np.unique(bjt_days[mc])],key=lambda r:-r['benefit_sse'])
        concentration[st]=dict(all_CPP_clear_restoration_benefit_sse=float(benefit[mc].sum()),zero_GHI_subset_benefit_sse=float(benefit[mc&(y==0)].sum()),top_dates=daily[:5])
        for name,g in [('all_CPP_3x3_clear',m),('zero_label',m&(y==0)),('nonzero_low_variation',m&(y>0)&(spread<=25)),('nonzero_medium_variation',m&(y>0)&(spread>25)&(spread<=100)),('nonzero_high_variation',m&(y>0)&(spread>100))]:
            if not g.any():continue
            strata[st][name]=dict(n=int(g.sum()),dates=int(len(np.unique(bjt_days[g]))),
                total_restoration_benefit_sse=float(benefit[g].sum()),mean_abs_end5_minus_mean15=float(np.mean(np.abs(end5[g]-y[g]))),
                median_range_5min=float(np.median(spread[g])),last5_exceeds_mean_by100=int(((end5-y>100)&g).sum()),
                daylight_zero_label_count=int((g&(y==0)).sum()))
    spa_path=ROOT/'experiments/monthly_R_station16_20261013_v1/results/t16_deterioration_forensics_20261014/python_deps/pvlib/spa.py'
    spec=importlib.util.spec_from_file_location('standalone_pvlib_spa',spa_path);spa=importlib.util.module_from_spec(spec);spec.loader.exec_module(spa)
    case_reports=[];figs=OUT/'figures';figs.mkdir(exist_ok=True);font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    with PdfPages(OUT/'time_and_shadow_case_audit.pdf') as pdf:
        for ci,c in enumerate(cases):
            st=c['station'];t=datetime.fromisoformat(c['utc']);tb=t+timedelta(hours=8)
            i=np.where((s==st)&(z['utc']==c['utc']))[0].item();j=z['refidx'][i]
            geom=next(r for r in meta['records'] if r['case']==ci and r['offset_min']==0);tm=raw['records'][ci]
            assert not tm['errors'];assert abs(tm['scan_offset_seconds']-geom['obs_time_offset_s'])<1e-3
            # Verify current raw CPP and prior reference cache use the same spatial orientation.
            rv=z['ref_valid'][j];np.testing.assert_allclose(cpp[f'c{ci}_COT'][rv],z['ref_cot'][j][rv],atol=1e-5)
            np.testing.assert_array_equal(cpp[f'c{ci}_CLP'][rv],z['ref_clp'][j][rv])
            tag=geom['array_tag'];ch=source[tag+'_channels'];lat=source[tag+'_lat'];lon=source[tag+'_lon'];station_lat,station_lon=COORD[st]
            scant=datetime.fromisoformat(tm['scan_midpoint_utc']);unix=np.array([(scant-datetime(1970,1,1)).total_seconds()]);sol=spa.solar_position(unix,station_lat,station_lon,0,0,12,69,.5667,1)[:,0]
            zen,az=float(sol[1]),float(sol[4]);vaz,vzen=geom['solar_source_degrees'][:2]
            solar=np.tan(np.deg2rad(zen))*np.array([np.sin(np.deg2rad(az)),np.cos(np.deg2rad(az))])
            view=np.tan(np.deg2rad(vzen))*np.array([np.sin(np.deg2rad(vaz)),np.cos(np.deg2rad(vaz))]);ray=solar-view
            east=(lon-station_lon)*111.32*np.cos(np.deg2rad(station_lat));north=(lat-station_lat)*111.32
            cot=cpp[f'c{ci}_COT'];clp=cpp[f'c{ci}_CLP'];cloud=np.isfinite(cot)&np.isin(clp,[1,2])&(cot>=3)
            shadow={}
            for name,vec in [('solar_only',solar),('solar_and_view_parallax',ray),('opposite_control',-ray)]:
                h=(east*vec[0]+north*vec[1])/np.dot(vec,vec);dist=np.sqrt((east-h*vec[0])**2+(north-h*vec[1])**2);eligible=cloud&(h>=.5)&(h<=15)
                if eligible.any():
                    k=np.argmin(np.where(eligible,dist,np.inf));r,col=np.unravel_index(k,dist.shape)
                    shadow[name]=dict(nearest_cloud_pixel=[int(r),int(col)],COT=float(cot[r,col]),required_height_km=float(h[r,col]),cross_track_km=float(dist[r,col]),product_CTH_value=float(cpp[f'c{ci}_CTH'][r,col]))
                else:shadow[name]=dict(nearest_cloud_pixel=None)
            v=[native.get((st,(tb-timedelta(minutes=m)).isoformat(sep=' ')),None) for m in (10,5,0)]
            scan_end=tb+timedelta(minutes=5);obs_scan=observed_native.get((st,scan_end.isoformat(sep=' ')))
            # Latest nominal and previous frame clear classes show whether clouds changed inside the label interval.
            prev=(st,(t-timedelta(minutes=15)).isoformat(sep=' '));previx=np.where((s==st)&(z['utc']==prev[1]))[0]
            prevref=int(ri[previx[0]]) if len(previx) else -1
            rec=dict(**c,index=int(i),target_bjt=tb.isoformat(sep=' '),label_window=[(tb-timedelta(minutes=15)).isoformat(sep=' '),tb.isoformat(sep=' ')],
                native_GHI_end_minus10_minus5_0=v,label_mean15=float(y[i]),last_5min_GHI=float(end5[i]),scan_containing_5min_GHI=obs_scan,
                actual_scan_bjt=(scant+timedelta(hours=8)).isoformat(sep=' '),scan_offset_s=tm['scan_offset_seconds'],
                interval15_center_to_scan_s=tm['scan_offset_seconds']+450,
                rawhead_GHI=float(z['raw_saved'][i]),gatehead_GHI=float(z['gate_saved'][i]),restore_CPPclear_GHI=float(z['gatehead_restore_CPPclear'][i]),
                CPP_station_CLP=float(clp[8,8]),CPP_station_COT=float(cot[8,8]),CPP_3x3_clear=bool(complete[i]),
                previous_frame_station_CLP=int(z['ref_clp'][prevref,8,8]) if prevref>=0 else None,
                solar_zenith_at_actual_scan_deg=zen,solar_azimuth_at_actual_scan_deg=az,
                input_solar_azimuth_deg=geom['solar_source_degrees'][2],input_solar_zenith_deg=geom['solar_source_degrees'][3],
                shadow_geometry=shadow,CTH_units_in_file=None,CTH_used_as_measured_km=False)
            case_reports.append(rec)
            fig=plt.figure(figsize=(15,9));gs=fig.add_gridspec(2,3,height_ratios=[1,1.2],left=.065,right=.94,bottom=.13,top=.83,hspace=.36,wspace=.30)
            ax=fig.add_subplot(gs[0,0]);rgb=np.clip(np.moveaxis(ch[[2,1,0]],0,-1)/.8,0,1)**(1/2.2);ax.imshow(rgb,interpolation='nearest');ax.plot(8,8,'w*',mec='k',ms=10);ax.set_title('真实 AGRI 假彩色 C03/C02/C01');ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
            ax=fig.add_subplot(gs[0,1]);im=ax.pcolormesh(east,north,np.ma.masked_where(~np.isfinite(cot),cot),cmap='magma',vmin=0,vmax=60,shading='nearest');ax.plot(0,0,'w*',mec='k',ms=10);ax.set_title('原始 CPP COT 与太阳方向');ax.set_xlabel('站点以东（km）');ax.set_ylabel('站点以北（km）');ax.set_aspect('equal');fig.colorbar(im,ax=ax,fraction=.03,pad=.03,label='COT',extend='max')
            for vec,lab,color,style in [(solar,'仅太阳：假设0.5–15km云高','#00FFFF','--'),(ray,'加卫星视差：同一云高范围','#00FF00','-')]:
                hs=np.array([.5,2,5,10,15]);ax.plot(hs*vec[0],hs*vec[1],style,color=color,label=lab,marker='.',ms=4)
            ax.set_xlim(east.min(),east.max());ax.set_ylim(north.min(),north.max());ax.legend(fontsize=6,loc='best')
            ax=fig.add_subplot(gs[0,2]);ph=np.ma.masked_where(~np.isfinite(clp),clp);ax.imshow(ph,cmap=matplotlib.colors.ListedColormap(['#F0F0F0','#56B4E9','#0072B2']),vmin=0,vmax=2,interpolation='nearest');ax.plot(8,8,'*',mfc='white',mec='k',ms=10);ax.set_title('原始 CPP 云相：0晴空，1/2有云');ax.set_xticks([0,8,15]);ax.set_yticks([0,8,15])
            ax=fig.add_subplot(gs[1,:]);xx=[];yy=[]
            for k in range(-90,96,5):
                tt=tb+timedelta(minutes=k);xx.append(tt);yy.append(observed_native.get((st,tt.isoformat(sep=' ')),np.nan))
            ax.plot(xx,yy,'o-',color='black',ms=3,label='原工作簿5分钟区间均值（结束时刻）');ax.axvspan(tb-timedelta(minutes=15),tb,alpha=.18,color='#E69F00',label='模型15分钟标签区间')
            ax.hlines(y[i],tb-timedelta(minutes=15),tb,colors='#E69F00',lw=3,label=f'标签均值 {y[i]:.2f}')
            ax.axvline(scant+timedelta(hours=8),color='#CC79A7',lw=1.8,label='实际 HDF 行扫描时刻')
            jj=np.where((s==st)&(np.abs(z['utc'].astype('datetime64[s]')-np.datetime64(t))<=np.timedelta64(90,'m')))[0]
            tt=z['utc'][jj].astype('datetime64[s]')+np.timedelta64(8,'h')
            for key,lab,color,style in [('raw_saved','原 COT 头','#0072B2','-'),('gate_saved','门控 COT 头','#D55E00','--'),('gatehead_restore_CPPclear','固定门控头+恢复 CPP 晴空处 R 值','#009E73',':')]:ax.plot(tt,z[key][jj],style,marker='.',color=color,label=lab)
            ax.set_xlabel('北京时间；5分钟值不是瞬时辐照度');ax.set_ylabel('GHI（W/m²）');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
            nm='四里' if st=='sili' else '竺家';fig.suptitle(f'{nm} {tb}：时间支持与云影几何核查',y=.97,fontsize=15)
            fig.text(.5,.87,f'标签三条原始记录：{v} → 均值 {y[i]:.2f}；结束记录 {end5[i]:.2f}；实际扫描所在5分钟区间 {obs_scan}\n图像实际晚于标签结束 {tm["scan_offset_seconds"]:.2f} 秒。云影轨迹只是0.5–15km云高敏感性，不是已确认云影。',ha='center',fontsize=10)
            fig.text(.5,.035,'相同目标时刻验证；不移动标签、不改变正式评分、不训练。CPP CTH 文件没有单位，未把它当作已确认的km云高。\n云影图给出站点被遮蔽时，云在表观卫星网格上可能出现的位置（含视差方案）；需要可信云高才能确认。\nAGRI为真实观测固定拉伸假彩色；显示原16×16像元，未合成云。',ha='center',fontsize=9)
            fig.savefig(figs/f'case{ci}_{st}.png',dpi=170);pdf.savefig(fig);plt.close(fig)
    report=dict(state='COMPLETE_BOUNDED_TIMING_AND_SHADOW_AUDIT',label_audit=dict(val_rows=len(rows),complete_15min_labels_checked=checked,max_mean_recomputation_error=maxerr,native_timestamp_conflicts=native_conflicts,workbook_cell_comparisons=source_comp),
        strata_CPP_3x3_clear=strata,restoration_benefit_concentration=concentration,cases=case_reports,test_payloads_read=0,
        shadow_limits='CTH unit missing in source NetCDF. Only explicit 0.5..15km geometric sensitivity; not a verified physical shadow attribution.',
        native_time_semantics='All raw 5min values are interval-ending means, not instantaneous radiation.',
        source_model='5 cases chosen post hoc; descriptive and no model training or label modification.')
    (OUT/'physics_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
