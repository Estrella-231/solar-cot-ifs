"""Local FP64 checks and actual-satellite scientific figures from frozen diagnostics."""
from pathlib import Path
import json,csv
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).parent;O=ROOT/'runtime_attempt3/fixed_head_diagnosis';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42})
ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
COL={'FA_pred':'#4c78a8','FAC_pred':'#e45756','FAC_observedCOT':'#54a24b','FAC_observedAGRI_observedCOT':'#b279a2'}
def read(p):return list(csv.DictReader(p.open()))
def metric(y,p):
    e=np.asarray(p,np.float64)-np.asarray(y,np.float64);return float(np.sqrt(np.mean(e*e)))
def save(fig,n):
    fig.savefig(F/(n+'.png'),bbox_inches='tight');fig.savefig(F/(n+'.pdf'),bbox_inches='tight');plt.close(fig)
def rgb(a):
    # Identical physical reflectance stretch for observations and forecasts, all dates.
    return np.clip(np.moveaxis(a[[2,1,0]],0,-1)/.8,0,1)**.6
def main():
    s=json.loads((O/'summary.json').read_text());z=np.load(O/'paired_predictions.npz');maps=np.load(O/'case_maps.npz');cases=json.loads((O/'selected_cases.json').read_text());rows=read(O/'rows.csv');modes=list(s['equal_station_seed_rmse']);N=len(rows);assert N==s['rows']
    independent={}
    for mode in modes:
        vals=[]
        for st in ST:
            take=z['station']==st
            for si in range(3):vals.append(metric(z['truth'][take],z[mode][si,take]))
        independent[mode]=float(np.mean(vals));assert abs(independent[mode]-s['equal_station_seed_rmse'][mode])<1e-8
    y=z['truth'];a=z['FA_pred'].mean(0);b=z['FAC_pred'].mean(0);c=z['FAC_observedCOT'].mean(0);co=z['observed_cot'][:,0,1,1];cp=z['predicted_cot'][:,0,1,1]
    extra={};groups={'all':np.ones(N,bool),'COT_log_error_le_0.25':abs(np.log1p(co)-np.log1p(cp))<=.25,'COT_log_error_gt_1':abs(np.log1p(co)-np.log1p(cp))>1,'observed_COT_ge_10':co>=10,'observed_GHI_ge_600':y>=600}
    for name,t in groups.items():
        if not t.any():continue
        extra[name]={'n':int(t.sum()),'FA_ensemble_mean_rmse':metric(y[t],a[t]),'FAC_ensemble_mean_rmse':metric(y[t],b[t]),'fixed_FAC_observedCOT_rmse':metric(y[t],c[t]),'mean_abs_fixed_head_COT_replacement':float(abs(c[t]-b[t]).mean()),'mean_abs_fixed_head_zero_slot':float(abs(z['FAC_zeroCOTslot'].mean(0)[t]-b[t]).mean()),'fraction_COT_replacement_reduces_SE':float(np.mean((c[t]-y[t])**2<(b[t]-y[t])**2))}
    # Grouping uses validation references only for diagnosis; never an inference rule.
    changes={}
    for name in ('FAC_observedCOT','FAC_zeroCOTslot','FAC_permutedCOT'):
        d=z[name]-z['FAC_pred'];changes[name]={'mean_absolute_wm2':float(abs(d).mean()),'rms_wm2':float(np.sqrt(np.mean(d*d))),'fraction_changes_over_1_wm2':float((abs(d)>1).mean())}
    (O/'local_independent_analysis.json').write_text(json.dumps({'fp64_equal_station_seed_rmse_recomputed':independent,'diagnostic_groups_ensemble_mean_not_primary_metric':extra,'within_same_head_response':changes},indent=2))
    for case in cases:
        st=case['station'];y0,x0=ST[st];si=int(case['sequence_index']);mi=int(np.flatnonzero(maps['sequence_index']==si)[0]);init=case['init'];take=np.flatnonzero((z['station']==st)&(z['init']==init));take=take[np.argsort(z['lead_minutes'][take])];leads=z['lead_minutes'][take];sel=[0,3,7,11,15]
        fig=plt.figure(figsize=(12.7,11.0));gs=fig.add_gridspec(6,5,height_ratios=[1,1,1,1,1.3,1.3],hspace=.48,wspace=.10)
        for row in range(4):
            for col,l in enumerate(sel):
                ax=fig.add_subplot(gs[row,col]);ax.set_xticks([]);ax.set_yticks([])
                night=not bool(maps['geometry'][mi,l,2,y0,x0]>.5)
                if row<2:
                    if night:
                        ax.set_facecolor('#ededed');ax.text(.5,.5,'Night',ha='center',va='center',transform=ax.transAxes,color='#555555')
                    else:ax.imshow(rgb(maps['observed_agri' if row==0 else 'predicted_agri'][mi,l]))
                else:
                    field=np.log1p(maps['observed_cot' if row==2 else 'predicted_cot'][mi,l,0]);field=np.where(maps['geometry'][mi,l,2]>.5,field,np.nan)
                    ax.set_facecolor('#ededed');im=ax.imshow(field,vmin=0,vmax=np.log1p(100),cmap='magma')
                ax.scatter(x0,y0,marker='*',s=70,c='white',edgecolors='black',linewidths=.7);ax.add_patch(Rectangle((x0-1.5,y0-1.5),3,3,fill=False,edgecolor='#4dd8ed',lw=1))
                if row==0:ax.set_xlabel(f'+{15*(l+1)} min');ax.xaxis.set_label_position('top')
                if col==0:ax.set_ylabel(['Observed AGRI\nC03/C02/C01','SimVP AGRI\nC03/C02/C01','R(observed AGRI)\nCOT','R(SimVP AGRI)\nCOT'][row])
        cax=fig.add_axes([.92,.38,.013,.24]);cb=fig.colorbar(im,cax=cax,ticks=np.log1p([0,1,3,10,30,100]));cb.ax.set_yticklabels(['0','1','3','10','30','>=100']);cb.set_label('COT (log colour spacing)')
        ax=fig.add_subplot(gs[4,:]);lt=np.arange(1,17)*15
        day=maps['geometry'][mi,:,2,y0,x0]>.5
        ax.plot(lt,np.where(day,maps['observed_cot'][mi,:,0,y0,x0],np.nan),'-o',color='#54a24b',ms=3,label='R(observed AGRI), offline reference');ax.plot(lt,np.where(day,maps['predicted_cot'][mi,:,0,y0,x0],np.nan),'-s',color='#e45756',ms=3,label='R(SimVP AGRI)')
        valid=maps['cpp_valid'][mi,:,0,y0,x0]&day;ref=np.where(valid,maps['cpp'][mi,:,0,y0,x0],np.nan)
        if valid.any():ax.plot(lt,ref,'k:D',ms=4,lw=1.6,label='CPP retrieval reference (valid pixels only)')
        ax.set_ylabel('Station-pixel COT');ax.set_xlim(0,245);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0)
        ax=fig.add_subplot(gs[5,:]);full_y=np.full(16,np.nan);positions=(leads//15-1).astype(int);full_y[positions]=y[take]
        ax.plot(lt,full_y,'-ok',ms=3,label='Observed 15-min mean GHI')
        for name,label in [('FA_pred','Forecast without COT'),('FAC_pred','Forecast with predicted COT'),('FAC_observedCOT','Same forecast head; observed-image COT (offline)')]:
            full=np.full(16,np.nan);full[positions]=z[name].mean(0)[take];ax.plot(lt,full,'-o',ms=2.5,color=COL[name],label=label)
        ax.set_ylabel('GHI (W m$^{-2}$)');ax.set_xlabel('Lead from history end (min); GHI interval ends at this time');ax.set_xlim(0,245);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0)
        fig.text(.10,.025,'Validation case; GHI curves: mean of 3 seed predictions. Observed-image COT is unavailable at forecast time.',fontsize=9)
        date=datetime.strptime(init,'%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M');ax.set_xlabel(f'{st}: lead from {date} BJT (min); GHI interval ends at this time')
        fig.subplots_adjust(left=.10,right=.90,top=.97,bottom=.10);save(fig,f'{st}_{init}_{case["category"]}')
    fig,axes=plt.subplots(1,2,figsize=(12,5.1),gridspec_kw={'width_ratios':[1,1.2]})
    cm=read(O/'cot_metrics.csv')
    for st,col in zip(ST,['#4c78a8','#f58518','#54a24b']):
        rs=[r for r in cm if r['station']==st and r['comparison']=='predR_vs_observedR_station'];axes[0].plot([int(r['lead_minutes']) for r in rs],[float(r['rmse']) for r in rs],'-o',ms=3,color=col,label=st)
    axes[0].set(xlabel='Forecast lead (min)',ylabel='Station COT RMSE vs R(observed AGRI)');axes[0].legend();axes[0].grid(alpha=.2)
    names=['FA_pred','FAC_pred','FAC_observedCOT','FAC_observedAGRI_predCOT','FAC_observedAGRI_observedCOT'];labels=['Forecast: AGRI, no COT','Forecast: AGRI + predicted COT','Offline: predicted AGRI + observed-image COT','Offline: observed AGRI + predicted COT','Offline: observed AGRI + observed-image COT'];vals=[independent[n] for n in names];positions=[0,1,3,4,5]
    bars=axes[1].barh(positions,vals,color=['#4c78a8','#e45756','#54a24b','#f2cf5b','#b279a2'],edgecolor='#444444',linewidth=.5)
    for bar in list(bars)[2:]:bar.set_hatch('//')
    axes[1].set_yticks(positions,labels,fontsize=8);axes[1].invert_yaxis();axes[1].set(xlabel='GHI RMSE (W m$^{-2}$)');axes[1].axhline(2,color='#888888',lw=.6)
    for j,v in zip(positions,vals):axes[1].text(v+.5,j,f'{v:.2f}',va='center',fontsize=9)
    axes[1].set_xlim(0,max(vals)*1.16)
    fig.text(.07,.025,'Left: support varies by lead; R(observed AGRI) is an offline retrieval reference, not independent COT truth.\nValidation only: 407 windows, 15,275 station-lead rows, 22 dates; 3 stations × 3 seeds. Bars: mean of 9 station-seed RMSE values.\nHatched bars use future observations, unavailable at forecast time. Forecast COT − no COT: +1.687 W m$^{-2}$; paired two-day 95% CI [−0.340, 4.089].',fontsize=8)
    fig.tight_layout(rect=[0,.17,1,1]);save(fig,'full_validation_attribution')
    lines=['# 湖北修正COT：真实云图、轨迹与冻结GHI头归因','', f'范围：407个验证序列，{N:,}个站点—初始化—目标时刻记录，22个目标日期，三站三seed。实际BJT时间修复后的复用验证。真实未来输入替换仅为离线诊断，不是预报成绩，也不是独立test。','', '## 完整样本上的固定权重输入替换','', '| 输入/处理 | 等站等seed RMSE (W/m²) |','|---|---:|']
    lines.extend(f'| {n} | {v:.6f} |' for n,v in independent.items());lines+=['','逐站、逐seed原始表见runtime_attempt3/fixed_head_diagnosis/head_interventions.csv。差值区间见runtime_attempt3/fixed_head_diagnosis/summary.json；本地独立复算见runtime_attempt3/fixed_head_diagnosis/local_independent_analysis.json。','', '## 真实图和曲线','', '每站两例：最大COT误差云事件，以及另一日期的中位误差活跃云事件。选择只看COT轨迹，不按GHI增益筛选。案例曲线为三seed预测均值，仅作诊断，不替代逐seed等权主指标；无效GHI标签断线，CPP有效的离散时刻用菱形标记，夜间图像留灰。所有帧使用相同反射率拉伸；青框为3×3输入，白星为站点。COT采用log1p色距，地图中COT≥100饱和为顶色；曲线保留超过100的实际值。云图是目标时刻快照，GHI是该时刻结束的15分钟区间均值。','']
    for case in cases:lines.append(f'- {case["station"]} {case["init"]}，{case["category"]}；站点COT RMSE={case["cot_rmse"]:.3f}，真实图R峰值={case["observed_peak"]:.3f}，预测图R峰值={case["predicted_peak"]:.3f}。')
    lines+=['','## 时间修复的剩余限制','','本轮只修复8小时标签时钟绑定；冻结S/R太阳几何仍保留源均时差漏乘180/pi的公式误差，天顶角最大偏差3.236度，不能称几何全链已修好。详见SOLAR_FORMULA_AUDIT.json和FINDINGS.md。','','## 解释边界','','真实图反演COT与预测图反演COT来自同一个修正R，可测量外推导致的表示差异，但不是独立物理真值；有CPP有效值的子集另列比较，不填补参考缺口。替换真实COT与预测AGRI可能构成不一致的训练外组合，替换不改善本身不能证明头部无能力；替换改善只能说明该冻结头在这个诊断条件下有响应。训练均值归一化槽置零及跨日期同站同lead置换，测量的是同一个头的输入依赖，与两个独立训练头之差不同。峰值时间偏移只能称站点COT信号偏移，不能把局部峰值跳动当作已追踪同一云团。','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'rmse':independent,'groups':extra,'head_response':changes},indent=2))
if __name__=='__main__':main()
