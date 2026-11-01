"""Source-grounded data/causality workflow, separate from model architecture."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
INK='#252C31'; GRAY='#65717B'; LIGHT='#F2F4F5'; TEAL='#007F78'; PALE='#EDF6F4'; ORANGE='#CE651D'; OPALE='#FFF2E8'
STYLE={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}

def main():
    with mpl.rc_context(STYLE):
        fig=plt.figure(figsize=(177/25.4,195/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,177),ylim=(195,0));ax.axis('off');nodes=[]
        def text(x,y,t,fs=8.3,ha='center',c=INK,b=False):return ax.text(x,y,t,fontsize=max(fs,8.0),ha=ha,va='center',color=c,weight='bold' if b else 'normal',linespacing=1.16)
        def box(id,x,y,w,h,t,c=INK,fill='white',fs=8.3,dash=False):
            ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=1.1',ec=c,fc=fill,lw=.75,ls='--' if dash else '-'));text(x,y,t,fs,c=c)
            nodes.append({'id':id,'x':x*5,'y':y*5,'width':w*5,'height':h*5,'label':t,'fill':fill,'stroke':c,'font_size':16})
        def arrow(points,c=INK,dash=False):
            for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],color=c,lw=.75,ls='--' if dash else '-')
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,lw=.75,color=c,ls='--' if dash else '-',shrinkA=0,shrinkB=0))
        def line(points,c=GRAY,lw=.65,dash=False):ax.plot([p[0] for p in points],[p[1] for p in points],color=c,lw=lw,ls='--' if dash else '-')
        def grid(cx,cy,w,n=8,c=GRAY):
            ax.add_patch(Rectangle((cx-w/2,cy-w/2),w,w,fc='white',ec=c,lw=.6))
            for i in range(1,n):
                z=-w/2+w*i/n;line([(cx+z,cy-w/2),(cx+z,cy+w/2)],c,.25);line([(cx-w/2,cy+z),(cx+w/2,cy+z)],c,.25)
        text(3,5,'(a)  Same-time COT value: observed images at target time τ',9.5,ha='left',b=True)
        box('obs',20,27,34,19,'Observed AGRI\n13 spectral channels\n+ known geometry',fill=LIGHT,fs=8.0)
        box('Rnow',66,27,32,19,'Frozen R\n16 × 16 context',fill=LIGHT)
        box('Hnow',121,27,37,19,'Same-time GHI head\naligned 3 × 3 inputs',fill=PALE,c=TEAL,fs=8.0)
        box('Gnow',161,27,25,19,'k × mean(GHIcs)\n15-min GHI',fs=8.0)
        arrow([(37,27),(50,27)]);arrow([(82,27),(102.5,27)],TEAL)
        text(92,20,'COT',8.2,c=TEAL);text(92,34,'16² → 3²',8.0,c=TEAL)
        arrow([(139.5,27),(148.5,27)])
        arrow([(20,17.5),(20,13),(121,13),(121,17.5)],GRAY)
        text(75,12,'same-time AGRI + geometry: 3 × 3',8.0,c=GRAY)
        box('cpp',57,47,43,12,'CPP reference\nretrieval supervision only',fs=8.0,dash=True)
        arrow([(66,41),(66,36.5)],GRAY,True)
        box('ground',116,47,37,12,'Ground GHI\nhead supervision only',fs=8.0,dash=True)
        arrow([(116,41),(116,36.5)],GRAY,True)
        text(159,47,'Interval\nclear-sky GHI',8.0,c=GRAY)
        arrow([(161,41),(161,36.5)],GRAY)
        line([(3,58),(174,58)],'#CCD2D4',.7)

        text(3,64,'(b)  Causal forecasting: only observations available by initialization t',9.5,ha='left',b=True)
        box('history',23,85,39,22,'8 observed AGRI frames\n13 bands + 3 geometry\n256 × 256 region',fill=LIGHT,fs=8.0)
        box('simvp',72,85,34,22,'Fixed SimVP\n8 → 16 frames\n256 × 256',fill=LIGHT,fs=8.2)
        box('forecast_agri',119,85,38,22,'16 forecast AGRI frames\n13 bands\n+15 to +240 min',fill=LIGHT,fs=8.0)
        arrow([(42.5,85),(55,85)]);arrow([(89,85),(100,85)])
        # Future COT and future-only GHI head.
        box('Rfuture',156,85,26,22,'Frozen R\n16 × 16\nper frame',fill=OPALE,c=ORANGE,fs=8.0)
        arrow([(138,85),(143,85)],ORANGE)
        box('futurehead',139,115,63,24,'',c=ORANGE,fill=OPALE,fs=8.1)
        text(139,106.5,'Equal-capacity residual corrections',8.0,c=ORANGE,b=True)
        line([(139,110),(139,120)],ORANGE,.55)
        text(122,115,'Matched control\nfuture COT slots = 0',8.0,c=ORANGE)
        text(155,115,'With forecast COT\n18 COT/change slots',8.0,c=ORANGE)
        text(139,123.5,'Shared context; separately trained corrections',8.0,c=ORANGE)
        arrow([(156,96),(156,103)],ORANGE);text(165,101,'COT',8.0,c=ORANGE)
        arrow([(119,96),(119,103)],GRAY)
        text(156,70,'Known target geometry',8.0,c=GRAY)
        arrow([(156,72),(156,74)],GRAY)
        # Historical COT is a distinct model/data contract, not silently joined.
        box('Rhistory',23,115,39,22,'Frozen R on history\n8 retrieved COT maps\n16 × 16 → 3 × 3',fill=PALE,c=TEAL,fs=8.0)
        arrow([(23,96),(23,104)],TEAL)
        box('historyhead',70,115,39,22,'Historical-COT base\nforecast + history AGRI\n+ historical COT\nfrozen for corrections',fill=PALE,c=TEAL,fs=8.0)
        arrow([(42.5,115),(50.5,115)],TEAL)
        arrow([(109,96),(109,100),(70,100),(70,104)],GRAY)
        # Show observed history reaches this head independently of R.
        arrow([(36,96),(36,100),(56,100),(56,104)],GRAY)
        arrow([(89.5,115),(107.5,115)],TEAL)
        text(98.5,111,'base k',8.0,c=TEAL)
        text(43,140,'Paired no-COT reference:\nsame base network; history COT slots = 0',8.0,c=GRAY)
        box('forecast_GHI',139,140,63,15,'correction = sigmoid(gate) × signed residual\nk = max(0, base k + available × correction)\nGHI = k × mean(GHIcs)',fill=LIGHT,fs=8.0)
        arrow([(139,127),(139,132.5)],ORANGE)
        line([(3,151),(174,151)],'#CCD2D4',.7)

        text(3,157,'(c)  Forecast time and the GHI averaging interval',9.2,ha='left',b=True)
        x0,xt,xe,y=8,55,172,171
        line([(x0,y),(xe,y)],INK,.8)
        for i in range(8):
            x=x0+(xt-x0)*i/7;ax.plot(x,y,marker='o',ms=2.8,c=GRAY)
        for j in range(16):
            left=xt+(xe-xt)*j/16;right=xt+(xe-xt)*(j+1)/16
            ax.add_patch(Rectangle((left,y-3.2),right-left,6.4,fc=OPALE,ec=ORANGE,lw=.55))
            if j in [0,3,7,11,15]:text(right,179,str((j+1)*15),8.0,c=ORANGE)
        line([(xt,164),(xt,184)],INK,.8,True)
        text(x0,179,'−105',8.0);text(xt,179,'t = 0',8.0)
        text(24,164,'8 history snapshots',8.0,c=GRAY);text(118,164,'16 future intervals',8.0,c=ORANGE)
        text(3,189,'τk = t + 15k min;   GHI label = mean over (τk − 15 min, τk].',8.4,ha='left')
        text(174,189,'k = 1, …, 16',8.2,ha='right')
        spec={'canvas':{'width':885,'height':975},'style':{'font_family':'Times New Roman, serif','font_size':16,'bg_color':'#FFFFFF'},'nodes':nodes,
              'edges':[{'from':a,'to':b} for a,b in [('obs','Rnow'),('Rnow','Hnow'),('Hnow','Gnow'),('history','simvp'),('simvp','forecast_agri'),('forecast_agri','Rfuture'),('Rfuture','futurehead'),('history','Rhistory'),('Rhistory','historyhead'),('historyhead','futurehead'),('futurehead','forecast_GHI')]]}
        (HERE/'figure_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        renderer=Path('C:/Users/29728/.codex/aris_repo/skills/figure-spec/scripts/figure_renderer.py')
        result=subprocess.run([sys.executable,str(renderer),'render',str(HERE/'figure_spec.json'),'--output',str(HERE/'stock_spec_draft.svg')],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        (HERE/'figurespec_render.log').write_bytes(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('FigureSpec failure')
        for ext in ['pdf','svg','png']:fig.savefig(HERE/f'fig_workflow.{ext}',dpi=300)
        fig.savefig(HERE/'print_preview.png',dpi=150);plt.close(fig)
    rels=['refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/infer_S.py',
          'refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/train.py',
          'refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/history_base.py',
          'refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/prepare.py',
          'experiments/monthly_R_station16_20261013_v1/train_ghi_heads.py']
    manifest={'type':'task/input workflow review','width_mm':177,'height_mm':195,'font_pt_range':[8.0,9.5],
              'result_claim':'none','test_payloads_read':False,'future_observation_inference':False,
              'sources':{r:hashlib.sha256((ROOT/r).read_bytes()).hexdigest() for r in rels},
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()



