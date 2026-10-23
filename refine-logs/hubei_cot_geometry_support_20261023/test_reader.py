"""Synthetic netCDF4 regression; never opens project CPP, GHI or test payloads."""
import copy,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import netCDF4
from extract_height_source import validate_history_request
P=Path(__file__).resolve().parent;checks=[]

def write_case(root,transpose=False):
    root.mkdir();init=datetime(2025,8,6,12);lat=29+np.arange(260)*.04;lon=100+np.arange(260)*.04
    yy=np.arange(257,1,-1);xx=np.arange(2,258)
    grid=root/'grid.npz';np.savez_compressed(grid,lat=np.repeat(lat[yy,None],256,1),lon=np.repeat(lon[None,xx],256,0))
    request=dict(init_time_BJT=init.isoformat(),rows=[]);expected=[]
    for i,off in enumerate(range(105,-1,-15)):
        t=init-timedelta(minutes=off);utc=t-timedelta(hours=8);file=root/f'FY4B_AGRI_{utc:%Y%m%d%H%M%S}.nc'
        field=(7000+2*np.arange(260)[:,None]+.5*np.arange(260)[None,:]) if i%2==0 else (7+.02*np.arange(260)[:,None]+.01*np.arange(260)[None,:])
        field[8,12]=np.nan;field[9,13]=0 # A valid zero must remain zero, not become missing.
        with netCDF4.Dataset(file,'w',format='NETCDF4') as ds:
            ds.createDimension('LAT',260);ds.createDimension('LON',260)
            ds.createVariable('LAT','f8',('LAT',))[:]=lat;ds.createVariable('LON','f8',('LON',))[:]=lon
            axes=('LON','LAT') if transpose and i==0 else ('LAT','LON')
            v=ds.createVariable('CTH','i2',axes,fill_value=-32768,zlib=True)
            v.units='m' if i%2==0 else 'km';v.scale_factor=.5 if i%2==0 else .01;v.add_offset=0.
            v[:]=np.ma.masked_invalid(field)
            ds.createVariable('CLP','i1',('LAT','LON'),fill_value=-127)[:]=2
            ds.createVariable('COT','f4',('LAT','LON'),fill_value=-999)[:]=0.
            if i%2==0:
                qa=ds.createVariable('QA','i1',('LAT','LON'),fill_value=-127);qa[:]=2;qa[8,12]=np.ma.masked
            ds.fixture_kind='SYNTHETIC_ONLY'
        request['rows'].append(dict(time_bjt=t.strftime('%Y%m%d%H%M'),UTC=utc.strftime('%Y%m%d%H%M'),split='train',source=str(file)))
        expected.append(field[np.ix_(yy,xx)])
    requestfile=root/'request.json';requestfile.write_text(json.dumps(request))
    return request,requestfile,grid,expected

with tempfile.TemporaryDirectory(prefix='cth_reader_synthetic_') as temp:
    root=Path(temp);req,requestfile,grid,expected=write_case(root/'valid')
    before={r['source']:hashlib.sha256(Path(r['source']).read_bytes()).hexdigest() for r in req['rows']}
    out=root/'valid_output';command=[sys.executable,str(P/'extract_height_source.py'),'--request',str(requestfile),'--grid',str(grid),'--out',str(out)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    receipt=json.loads((out/'COMPLETE.json').read_text());assert len(receipt['records'])==8
    max_error=0.
    for i,r in enumerate(req['rows']):
        with np.load(out/(r['time_bjt']+'.npz')) as z:
            actual=z['CTH_raw'];truth=expected[i];assert np.array_equal(np.isnan(actual),np.isnan(truth))
            error=float(np.nanmax(abs(actual-truth)));max_error=max(max_error,error);assert error<1e-10
            assert actual[257-9,13-2]==0
            assert ('QA_raw' in z.files)==(i%2==0)
        metadata=receipt['records'][i]['metadata'];assert metadata['QA_present']==(i%2==0)
        assert metadata['variable_attributes']['CTH']['units']==('m' if i%2==0 else 'km')
        assert metadata['decoded_values_in_original_units'] is True
        assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==before[r['source']]
    checks.append(dict(name='real_NetCDF4_packed_scale_units_descending_crop_fill_zero_QA_source_preservation',state='PASS',max_error=max_error))
    bad=copy.deepcopy(req);bad['rows'][1]=copy.deepcopy(bad['rows'][0])
    try:validate_history_request(bad)
    except AssertionError:checks.append(dict(name='duplicate_history_rejected',state='PASS'))
    else:raise AssertionError('Duplicate history accepted')
    for name,change in [('nontrain',lambda q:q['rows'][0].update(split='test')),('wrong_UTC',lambda q:q['rows'][0].update(UTC='202508060000')),('future_or_gap',lambda q:q['rows'][-1].update(time_bjt='202508061215'))]:
        bad=copy.deepcopy(req);change(bad)
        try:validate_history_request(bad)
        except AssertionError:checks.append(dict(name=name+'_rejected_before_source_read',state='PASS'))
        else:raise AssertionError(name+' accepted')
    repeated=subprocess.run(command,capture_output=True,text=True);assert repeated.returncode!=0 and 'FileExistsError' in repeated.stderr
    checks.append(dict(name='existing_output_not_overwritten',state='PASS'))
    _,tr,gr,_=write_case(root/'transpose',True)
    failed=subprocess.run([sys.executable,str(P/'extract_height_source.py'),'--request',str(tr),'--grid',str(gr),'--out',str(root/'transpose_output')],capture_output=True,text=True)
    assert failed.returncode!=0 and 'AssertionError' in failed.stderr and not (root/'transpose_output/COMPLETE.json').exists()
    checks.append(dict(name='swapped_dimension_names_rejected',state='PASS'))
    with np.load(grid) as z:g={k:z[k] for k in z.files}
    g['lon']=g['lon']+.005;badgrid=root/'badgrid.npz';np.savez_compressed(badgrid,**g)
    failed=subprocess.run([sys.executable,str(P/'extract_height_source.py'),'--request',str(requestfile),'--grid',str(badgrid),'--out',str(root/'badgrid_output')],capture_output=True,text=True)
    assert failed.returncode!=0 and 'AssertionError' in failed.stderr and not (root/'badgrid_output/COMPLETE.json').exists()
    checks.append(dict(name='unmatched_geographic_grid_rejected',state='PASS'))

result=dict(state='SYNTHETIC_NETCDF_READER_PASS',checks=checks,netCDF4_version=netCDF4.__version__,numpy_version=np.__version__,
    input_data='generated temporary synthetic NetCDF only',actual_CPP_opened=False,GHI_labels_opened=False,test_payloads_opened=False,
    source_code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['extract_height_source.py','test_reader.py']})
(P/'SYNTHETIC_READER_TEST.json').write_text(json.dumps(result,indent=2));print('SYNTHETIC_READER_PASS',len(checks),max_error)
