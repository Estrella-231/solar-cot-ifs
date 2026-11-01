"""Deterministic source-audited architecture, 177-mm manuscript review figure."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

HERE=Path(__file__).resolve().parent
INK='#252C31'; GRAY='#65717B'; LIGHT='#F2F4F5'; TEAL='#007F78'; PALE='#EDF6F4'; ORANGE='#CE651D'
STYLE={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8.4,
       'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}

def main():
    with mpl.rc_context(STYLE):
        fig=plt.figure(figsize=(177/25.4,188/25.4)); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,177),ylim=(188,0)); ax.axis('off')
        nodes=[]; edges=[]
        def text(x,y,t,fs=8.4,ha='center',va='center',c=INK,b=False):
            return ax.text(x,y,t,fontsize=max(8.0,fs),ha=ha,va=va,color=c,weight='bold' if b else 'normal',linespacing=1.16)
        def box(id,x,y,w,h,t,fs=8.4,c=INK,fill='white',dash=False):
            ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=1.1',lw=.75,ec=c,fc=fill,ls='--' if dash else '-'))
            text(x,y,t,fs,c=c);nodes.append({'id':id,'x':x*5,'y':y*5,'width':w*5,'height':h*5,'label':t,'fill':fill,'stroke':c,'font_size':16})
        def arrow(points,c=INK,dash=False,lw=.75):
            for p,q in zip(points[:-2],points[1:-1]): ax.plot([p[0],q[0]],[p[1],q[1]],color=c,lw=lw,ls='--' if dash else '-')
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,lw=lw,color=c,ls='--' if dash else '-',shrinkA=0,shrinkB=0))
        def line(points,c=GRAY,lw=.65,dash=False):
            ax.plot([p[0] for p in points],[p[1] for p in points],color=c,lw=lw,ls='--' if dash else '-')
        def plus(x,y):
            ax.add_patch(Circle((x,y),1.5,fc='white',ec=GRAY,lw=.65));text(x,y,'C',7,c=GRAY)
        def grid(x,y,w=12,c=GRAY,labels=False):
            ax.add_patch(Rectangle((x-w/2,y-w/2),w,w,fc='white',ec=c,lw=.7))
            for i in [1,2]:
                z=-w/2+w*i/3;line([(x+z,y-w/2),(x+z,y+w/2)],c,.4);line([(x-w/2,y+z),(x+w/2,y+z)],c,.4)
            ax.add_patch(Rectangle((x-w/6,y-w/6),w/3,w/3,fc=PALE,ec=c,lw=.65))
            if labels:
                for r in range(3):
                    for j in range(3):text(x-w/3+j*w/3,y-w/3+r*w/3,str(r*3+j+1),6.7,c=c)
            else:text(x,y,'+',9,c=c)
        text(3,4,'(a)  COT retrieval from a single observed frame',9.5,ha='left',b=True)
        text(174,4,'R frozen during GHI training',8.0,ha='right',c=GRAY)
        # U-Net: actual topology shared by audited regional implementations.
        box('input',16,24,27,20,'13 AGRI bands\n+ 3 geometry\n16 × 16',8.0,fill=LIGHT)
        box('enc1',48,24,24,15,'Conv block\nw × 16 × 16',8.1,fill=LIGHT)
        box('enc2',76,43,24,15,'Conv block\n2w × 8 × 8',8.1,fill=LIGHT)
        box('mid',104,62,24,15,'Conv block\n4w × 4 × 4',8.1,fill=LIGHT)
        box('dec2',132,43,24,15,'Conv block\n2w × 8 × 8',8.1,fill=LIGHT)
        box('dec1',161,24,24,15,'Conv block\nw × 16 × 16',8.1,fill=LIGHT)
        arrow([(29.5,24),(36,24)])
        arrow([(48,31.5),(48,43),(64,43)]);text(51,37,'pool',7.3,ha='left',c=GRAY)
        arrow([(76,50.5),(76,62),(92,62)]);text(79,56,'pool',7.3,ha='left',c=GRAY)
        arrow([(116,62),(132,62),(132,50.5)]);text(122,57,'up',7.3,c=GRAY)
        arrow([(144,43),(161,43),(161,31.5)]);text(151,38,'up',7.3,c=GRAY)
        # Skips reach concatenation points, not feature addition.
        line([(48,16.5),(48,12),(161,12),(161,14.8)],GRAY);plus(161,15)
        text(105,11,'skip concatenate',7.8,c=GRAY)
        line([(76,35.5),(76,32),(132,32),(132,33.8)],GRAY);plus(132,34)
        text(104,31,'skip concatenate',7.8,c=GRAY)
        box('out',160,62,29,22,'1 × 1 + Softplus\nℓ = log(1 + COT)\nCOT = expm1(ℓ)\n1 × 16 × 16',7.8,c=TEAL,fill=PALE)
        arrow([(173,24),(175,24),(175,62),(174.5,62)],TEAL)
        text(17,43,'Known geometry:\ncosSOZ, cosRAA,\nday mask',7.8,c=GRAY)
        box('cpp',22,62,34,15,'CPP COT reference\ntraining supervision',7.8,dash=True)
        arrow([(39,62),(45,62),(45,76),(160,76),(160,73)],GRAY,True)
        text(82,75,'masked Huber in log-COT',7.8,c=GRAY)
        text(3,82,'Block: [3 × 3 Conv–BN–SiLU] × 2     Pool: 2 × 2     Up: 2 × 2 transposed Conv',7.8,ha='left',c=GRAY)
        text(3,87,'Widths: w = 32 (Hunan); w = 16 (Hubei).  Batch dimension omitted throughout.',7.8,ha='left',c=GRAY)
        line([(3,92),(174,92)],'#CDD3D6',.7)

        text(3,98,'(b)  Aligned local fusion and physical GHI readout',9.5,ha='left',b=True)
        text(174,98,'Hunan single-COT-channel head',8.0,ha='right',c=GRAY)
        grid(13,112,12);text(13,123,'3 × 3 AGRI',7.8)
        box('local_agri',43,112,34,15,'13 AGRI + 2 geometry\n15 × 3 × 3',7.8,fill=LIGHT)
        box('agri_encoder',82,112,31,15,'1 × 1 Conv + SiLU\n15 → 32 → 32',8.0,fill=LIGHT)
        arrow([(19,112),(26,112)]);arrow([(60,112),(66.5,112)])
        grid(13,141,12,TEAL);text(13,152,'3 × 3 COT',7.8,c=TEAL)
        box('cot_norm',43,141,34,18,'log1p(COT)\ntrain standardization\n1 × 3 × 3',7.8,c=TEAL,fill=PALE)
        box('cot_encoder',82,141,31,15,'1 × 1 Conv + SiLU\n1 → 4',8.0,c=TEAL,fill=PALE)
        arrow([(19,141),(26,141)],TEAL);arrow([(60,141),(66.5,141)],TEAL)
        box('concat',116,126.5,24,23,'Concatenate\n36 × 3 × 3\n↓\nFlatten: 324',8.0)
        arrow([(97.5,112),(101,112),(101,121),(104,121)])
        arrow([(97.5,141),(101,141),(101,132),(104,132)],TEAL)
        box('mlp',153,126.5,32,23,'MLP + SiLU\n324 → 64 → 32 → 1\nSoftplus → k',7.9)
        arrow([(128,126.5),(137,126.5)])
        box('ghi',153,150,36,14,'GHI = k × mean(GHIcs)\n15-min interval mean',8.0,fill=LIGHT)
        arrow([(153,138),(153,143)])
        text(63,155,'All nine positions retained; no spatial pooling.',8.0,c=GRAY)
        text(115,150,'interval\nclear sky',7.5,c=GRAY)
        arrow([(125,150),(135,150)],GRAY)
        line([(3,162),(174,162)],'#CDD3D6',.7)

        text(3,168,'(c)  Spatial order and implementation-specific controls',9.0,ha='left',b=True)
        grid(12,179,12,GRAY,True);grid(29,179,12,TEAL,True)
        text(40,178,'Same station pixels\nFlatten: channel → row → column',7.7,ha='left')
        line([(100,173),(100,186)],'#CCD2D4',.65)
        text(104,176,'Hubei four-arm interface',8.0,ha='left',b=True)
        text(104,182,'[zCOT, zPcloud] → Conv 2→4 (no bias)\nCOT-only arm: [zCOT, 0]',7.7,ha='left')
        # The canonical spec is an editable stock draft; the final renderer uses
        # explicitly routed skips/grids needed for the real topology.
        spec={'canvas':{'width':885,'height':940},'style':{'font_family':'Times New Roman, serif','font_size':16,'bg_color':'#FFFFFF'},'nodes':nodes,
              'edges':[{'from':a,'to':b} for a,b in [('input','enc1'),('enc1','enc2'),('enc2','mid'),('mid','dec2'),('dec2','dec1'),('dec1','out'),('local_agri','agri_encoder'),('cot_norm','cot_encoder'),('agri_encoder','concat'),('cot_encoder','concat'),('concat','mlp'),('mlp','ghi')]]}
        (HERE/'figure_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        renderer=Path('C:/Users/29728/.codex/aris_repo/skills/figure-spec/scripts/figure_renderer.py')
        result=subprocess.run([sys.executable,str(renderer),'render',str(HERE/'figure_spec.json'),'--output',str(HERE/'stock_spec_draft.svg')],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        (HERE/'figurespec_render.log').write_bytes(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('FigureSpec validation/render failed; see log')
        for ext in ['pdf','svg','png']:fig.savefig(HERE/f'fig_core_architecture.{ext}',dpi=300)
        fig.savefig(HERE/'print_preview.png',dpi=150)
        plt.close(fig)
    payload={'type':'source-audited architecture review','width_mm':177,'height_mm':188,'font_points':[8.0,9.5],
             'result_claim':'none; verified architecture only; final Hubei corrected-geometry same-time result binding pending',
             'renderer':'Matplotlib deterministic vector diagram + validated FigureSpec stock topology',
             'diagram_scope':'single-frame; Hunan single-COT-channel main head, Hubei two-slot control inset',
             'no_test_or_training_payloads':True,'sources':json.loads((HERE/'CODE_WITNESS.json').read_text())['sources'],
             'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
    (HERE/'MANIFEST.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()

