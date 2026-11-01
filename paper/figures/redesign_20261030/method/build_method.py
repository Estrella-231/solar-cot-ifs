"""Deterministic method diagram; no experimental imagery is synthesized.

FigureSpec produces the stock draft. The final renderer adds true grid topology,
orthogonal branches, and an interval timeline, unsupported by the stock schema.
"""
from pathlib import Path
import json, hashlib, subprocess, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
INK, TEAL, GRAY = '#20272B', '#007F78', '#697983'
STYLE = {'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
         'font.size':9, 'pdf.fonttype':42, 'ps.fonttype':42,
         'svg.fonttype':'none', 'savefig.facecolor':'white'}

def spec():
    nodes = [
      dict(id='agri', label='Observed AGRI\n13 bands; 16 × 16', x=110,y=150,width=175,height=100),
      dict(id='crop', label='Local AGRI + geometry\n15 × 3 × 3',x=330,y=150,width=185,height=100),
      dict(id='agri_encoder',label='Pixelwise encoder\n15 → 32 → 32',x=535,y=150,width=170,height=70),
      dict(id='R',label='Frozen retriever R\n13 bands + geometry',x=110,y=325,width=185,height=70),
      dict(id='cot',label='Retrieved COT\n16 × 16 → local 3 × 3',x=330,y=325,width=185,height=100),
      dict(id='cot_encoder',label='COT encoder\n1 → 4',x=535,y=325,width=170,height=70),
      dict(id='merge',label='Concatenate\nthen ordered flatten',x=738,y=235,width=160,height=85),
      dict(id='head',label='MLP\n324 → 64 → 32 → 1',x=915,y=235,width=145,height=85),
      dict(id='ghi',label='Clear-sky scaling\n15-min mean GHI',x=890,y=385,width=190,height=75),
      dict(id='history',label='8 observed frames\nending at t',x=100,y=625,width=165,height=75),
      dict(id='simvp',label='Fixed SimVP\n256 × 256 region',x=310,y=625,width=165,height=75),
      dict(id='pred',label='16 predicted AGRI\nframes; 13 bands',x=520,y=625,width=165,height=75),
      dict(id='map',label='Frozen R + GHI head\naligned local inputs',x=735,y=625,width=180,height=75),
      dict(id='forecast',label='16 GHI forecasts\n+15 to +240 min',x=930,y=625,width=130,height=75),
    ]
    edges=[('agri','crop'),('crop','agri_encoder'),('agri','R'),('R','cot'),('cot','cot_encoder'),
           ('agri_encoder','merge'),('cot_encoder','merge'),('merge','head'),('head','ghi'),
           ('history','simvp'),('simvp','pred'),('pred','map'),('map','forecast')]
    return {'title':'COT-assisted GHI estimation and forecast input availability',
       'canvas':{'width':1020,'height':870},
       'style':{'font_family':'Times New Roman, serif','font_size':18,'bg_color':'#FFFFFF','palette':['#F3F6F6']},
       'nodes':nodes,'edges':[{'from':a,'to':b,'color':INK} for a,b in edges],
       'labels':[{'text':'(a) Aligned local estimation','x':20,'y':35,'font_size':22,'anchor':'start'},
                 {'text':'(b) Forecast inputs available at initialization','x':20,'y':535,'font_size':22,'anchor':'start'}]}

def main():
    s=spec(); (HERE/'method_spec.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    renderer=Path('C:/Users/29728/.codex/aris_repo/skills/figure-spec/scripts/figure_renderer.py')
    subprocess.run([__import__('sys').executable,str(renderer),'render',str(HERE/'method_spec.json'),
                    '--output',str(HERE/'stock_figurespec_draft.svg')],check=True,capture_output=True,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    with mpl.rc_context(STYLE):
      fig=plt.figure(figsize=(177/25.4,157/25.4))
      ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1020),ylim=(870,0)); ax.axis('off')
      def text(x,y,t,fs=9,ha='center',color=INK,weight='normal',va='center'):
          return ax.text(x,y,t,fontsize=fs,ha=ha,va=va,color=color,weight=weight,linespacing=1.25)
      def box(x,y,w,h,label,color=INK,fill='white',fs=9):
          ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=5',
                        linewidth=.8,edgecolor=color,facecolor=fill))
          text(x,y,label,fs,color=color)
      def line(points,color=INK,dash=False):
          for a,b in zip(points[:-2],points[1:-1]):
              ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.85,ls='--' if dash else '-')
          ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,
                       lw=.85,linestyle='--' if dash else '-',color=color,shrinkA=0,shrinkB=0))
      def grid(cx,cy,size,n,color=GRAY,focus=False,stack=False):
          if stack:
              for d in [12,6]:
                  ax.add_patch(Rectangle((cx-size/2+d,cy-size/2-d),size,size,facecolor='#F4F6F7',edgecolor=color,lw=.55))
          ax.add_patch(Rectangle((cx-size/2,cy-size/2),size,size,facecolor='white',edgecolor=color,lw=.7))
          for i in range(1,n):
              a=-size/2+size*i/n
              ax.plot([cx+a,cx+a],[cy-size/2,cy+size/2],lw=.28,color=color,alpha=.65)
              ax.plot([cx-size/2,cx+size/2],[cy+a,cy+a],lw=.28,color=color,alpha=.65)
          if focus:
              q=size/n; left=cx-size/2+7*q; top=cy-size/2+7*q
              ax.add_patch(Rectangle((left,top),3*q,3*q,fc='#C9E6E3',ec=TEAL,lw=1))
              text(left+1.5*q,top+1.5*q,'+',fs=8,color=INK)
          elif n==3:
              text(cx,cy,'+',fs=10,color=INK)
      text(20,30,'(a)  COT-assisted estimation from observed imagery',10.5,ha='left',weight='bold')
      grid(110,140,80,16,stack=True)
      text(110,209,'AGRI: 13 bands\n16 × 16 context')
      grid(330,140,67,3)
      text(330,209,'13 AGRI + 2 geometry\nchannels; 3 × 3')
      box(535,140,170,72,'1 × 1 encoders\n15 → 32 → 32')
      line([(162,140),(294,140)]); line([(365,140),(450,140)])
      box(110,318,180,66,'Frozen retriever R\n+ geometry / day flag',fs=8.5)
      line([(110,240),(110,285)])
      grid(330,318,80,16,color=TEAL,focus=True)
      text(330,394,'COT: 16 × 16\nselect the aligned 3 × 3',color=TEAL)
      line([(200,318),(288,318)],TEAL)
      box(535,318,170,100,'Train-standardized\nlog(1 + COT)\n1 × 1 encoder: 1 → 4',color=TEAL,fill='#F1F8F7',fs=8.5)
      line([(372,318),(450,318)],TEAL)
      box(737,228,154,88,'Concatenate\nordered flatten\n36 × 3 × 3',fs=9)
      line([(620,140),(638,140),(638,202),(660,202)])
      line([(620,318),(638,318),(638,254),(660,254)],TEAL)
      box(916,228,149,88,'MLP\n324 → 64 → 32\n→ clear-sky index',fs=8.5)
      line([(814,228),(841,228)])
      box(886,374,209,72,r'$\widehat G=\widehat k\,\overline{G^{cs}}$'+'\n15-min mean GHI',fill='#F4F6F7',fs=10)
      line([(916,272),(916,338)])
      text(110,417,'CPP COT\ntraining supervision only',fs=8.5)
      line([(110,393),(110,351)],dash=True)
      text(515,452,'Nine spatial positions retained; no global spatial pooling.',fs=9,color=GRAY)
      ax.plot([20,1000],[485,485],color='#CDD4D6',lw=.7)
      text(20,523,'(b)  Future inputs from a fixed cloud-image forecast',10.5,ha='left',weight='bold')
      xs=[105,315,525,745,940]
      box(xs[0],620,175,76,'8 observed frames\nending at t',fill='#F4F6F7')
      box(xs[1],620,170,76,'Fixed SimVP\n256 × 256 region')
      box(xs[2],620,170,76,'16 predicted AGRI\nframes; 13 bands')
      box(xs[3],620,190,88,'R → predicted COT\nAGRI + COT → GHI\n+ target geometry',color=TEAL,fill='#F1F8F7',fs=8.5)
      box(xs[4],620,130,76,'16 GHI\nforecasts')
      for x1,x2 in [(192,230),(400,440),(610,650),(840,875)]: line([(x1,620),(x2,620)])
      text(745,695,'Forecast-input GHI heads are trained separately.',8.3,color=GRAY)
      text(315,560,'Time/location → known solar geometry',8.7,color=GRAY)
      line([(315,574),(315,582)],GRAY)
      # Timeline: observed frames occupy points; future GHI occupies intervals.
      x0,xt,xf=65,310,980; y=760
      ax.plot([x0,xf],[y,y],color=INK,lw=.8)
      for i in range(8):
          x=x0+(xt-x0)*i/7
          ax.plot(x,y,'s',ms=3.3,color=GRAY)
      step=(xf-xt)/16
      for k in range(1,17):
          ax.add_patch(Rectangle((xt+(k-1)*step,y-8),step,16,fc='#E3F0EE',ec=TEAL,lw=.45))
      for k,t in [(0,'t'),(1,'+15'),(4,'+60'),(8,'+120'),(12,'+180'),(16,'+240 min')]:
          text(xt+k*step,795,t,8.1)
      text(165,795,'Observed history',8.2)
      text(510,841,r'Target $\tau_k$: mean GHI over $(\tau_k-15\,\mathrm{min},\,\tau_k]$',9)
      for ext in ['pdf','svg','png']:
          fig.savefig(HERE/f'method_workflow.{ext}',dpi=300)
      plt.close(fig)
    sources=[PAPER/'main.tex', PAPER/'WRITING_REQUIREMENTS.md']
    manifest={'kind':'architecture schematic, not an experimental cloud image',
      'width_mm':177,'height_mm':157,'font_pt_range':[8.1,10.5],
      'semantics':'Synchronous one-frame estimation. Forecast is a separately trained application of the mapping, not shared GHI weights.',
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
      'renderer':'FigureSpec stock draft plus Matplotlib orthogonal/grid/timeline renderer',
      'geometry':'R: cosSOZ,cosRAA,day_mask; GHI: cosSOZ,cosRAA',
      'reference_scope':'CPP labels only in retrieval fitting; ground GHI only in head fitting/scoring',
      'not_shown':'history-COT/residual-probe variants require their own experiment diagram; not implied by this single-frame main architecture',
      'matplotlib_version':mpl.__version__}
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created method_workflow.pdf/svg/png and source manifest')

if __name__=='__main__': main()
