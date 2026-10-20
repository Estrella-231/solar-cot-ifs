const fs=require('fs'),vm=require('vm'),crypto=require('crypto');
const [sourceFile,casesFile,outFile]=process.argv.slice(2);
const source=fs.readFileSync(sourceFile,'utf8'), ctx={Math,Date,parseFloat,parseInt};vm.createContext(ctx);vm.runInContext(source,ctx,{timeout:3000});ctx.calcRefraction=()=>0;
let count=0,zmax=0,amax=0,worst=null,bySplit={};
for(const line of fs.readFileSync(casesFile,'utf8').split(/\r?\n/)){
 if(!line.trim())continue;const c=JSON.parse(line),s=c.utc;let y=+s.slice(0,4),m=+s.slice(4,6),d=+s.slice(6,8),minutes=(+s.slice(8,10))*60+(+s.slice(10,12));
 let jd=ctx.getJD(y,m,d)+minutes/1440,T=ctx.calcTimeJulianCent(jd),a=ctx.calcAzEl(T,minutes,c.lat,c.lon,0),z=90-a.elevation,ze=Math.abs(z-c.zenith),ae=Math.abs(((a.azimuth-c.azimuth+540)%360)-180);
 if(ze>zmax||ae>amax)worst={...c,noaa_zenith:z,noaa_azimuth:a.azimuth,zenith_error:ze,azimuth_error:ae};zmax=Math.max(zmax,ze);amax=Math.max(amax,ae);count++;bySplit[c.split]=(bySplit[c.split]||0)+1;
}
if(count<1||zmax>1e-6||amax>1e-6)throw Error(JSON.stringify({count,zmax,amax,worst}));
const result={state:'PASS_AGAINST_OFFICIAL_NOAA_JAVASCRIPT_GEOMETRIC_ANGLES',reference_url:'https://gml.noaa.gov/grad/solcalc/main.js',reference_sha256:crypto.createHash('sha256').update(source).digest('hex'),cases:count,by_split:bySplit,zenith_max_abs_deg:zmax,azimuth_max_circular_abs_deg:amax,refraction:'disabled explicitly in reference; compare geometric not apparent zenith',worst_case:worst};fs.writeFileSync(outFile,JSON.stringify(result,null,2));console.log(JSON.stringify(result));
