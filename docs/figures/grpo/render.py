from pathlib import Path
import json, csv, hashlib, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='illustrated-grpo'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parent
OUT=ROOT; OUT.mkdir(exist_ok=True)
SRC=ROOT.parents[2]/'output'
ONLY={name for name in os.environ.get('GRPO_FIGURES','').split(',') if name}
SAVE_METADATA={
 'png':{'Software':'Matplotlib'},
 'svg':{'Date':None,'Creator':'Matplotlib'},
 'pdf':{'Creator':'Matplotlib','Producer':'Matplotlib','CreationDate':None,'ModDate':None},
}
labels=['3-digit addition | Full FT','5-digit addition | Full FT','6-digit addition | Full FT','6-digit addition | LoRA','2-digit multiplication | LoRA']
stems=['stage4','stage4b','stage4c','stage4d','stage4e']
ef=['500','5digit_500','6digit_500','6digit_lora_500','2digit_mul_lora_500']
logs=[json.loads((SRC/(s+'_trainer_state.json')).read_text())['log_history'] for s in stems]
ev=[json.loads((SRC/('eval_results_'+s+'.json')).read_text()) for s in ef]
BLUE='#087E8B'; ORANGE='#BD4B16'; GRAY='#707C8C'; INK='#202C3C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':GRAY,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
manifest=[]
def vals(i,key): return np.array([r[key] for r in logs[i]])
def setup(title,sub,size=(12,7)):
 f=plt.figure(figsize=size)
 f.text(.07,.955,title,fontsize=19,weight='bold',va='top')
 f.text(.07,.905,sub,fontsize=10.5,color=GRAY,va='top')
 return f
def foot(f,s): f.text(.07,.025,s,fontsize=9,color=GRAY,va='bottom')
def save(f,name,claim):
 if not ONLY or name in ONLY:
  for fmt in ['png','svg','pdf']:
   path=OUT/(name+'.'+fmt); f.savefig(path,dpi=300,metadata=SAVE_METADATA[fmt])
   if fmt=='svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 manifest.append({'id':name,'claim':claim,'size_inches':list(f.get_size_inches()),'formats':['png','svg','pdf']})
 plt.close(f)
def grid(a): a.grid(axis='y',color='#E3E8ED'); a.set_axisbelow(True)
def curve(a,i,key,color=BLUE,style='-',label=None):
 x=vals(i,'step'); y=vals(i,key)
 a.plot(x,y,color=color,alpha=.18,lw=.65)
 a.plot(x[19:],np.convolve(y,np.ones(20)/20,mode='valid'),color=color,ls=style,lw=2,label=label)
 a.set_xlim(1,500); grid(a)

# 1: Each row is an experiment, not a connected trajectory across tasks.
f=setup('Five experiments: legacy before/after observations','Greedy accuracy under the original evaluation protocol; 500 rows per question-generation partition')
gs=f.add_gridspec(1,2,left=.30,right=.95,bottom=.17,top=.80,width_ratios=[2,1],wspace=.30)
a,b=[f.add_subplot(gs[0,j]) for j in range(2)]
for i,e in enumerate(ev):
 x=100*e['base_model']['mean']; y=100*e['trained_model']['mean']
 a.plot([x,y],[i,i],color='#BBC3CC',lw=3)
 a.scatter(x,i,c=GRAY,marker='o',s=65,zorder=3,label='Base' if i==0 else None)
 a.scatter(y,i,c=BLUE,marker='D',s=55,zorder=3,label='Trained' if i==0 else None)
 a.text(x,i-.15,f'{x:.2f}',ha='center',fontsize=10,color=GRAY)
 a.text(y,i+.24,f'{y:.2f}',ha='center',fontsize=10,color=BLUE)
 d=y-x; b.scatter(d,i,c=INK,s=65,marker='D'); b.annotate(f'{d:+.2f}',(d,i),xytext=(10 if d>=0 else -10,0),textcoords='offset points',ha='left' if d>=0 else 'right',va='center')
a.set_yticks(range(5),labels); a.set_ylim(4.6,-.6); a.set_xlim(60,100); a.set_xlabel('Accuracy (%)'); a.legend(loc='upper left',bbox_to_anchor=(0,1.16),frameon=False,ncol=2)
b.set_ylim(a.get_ylim()); b.set_yticks([]); b.axvline(0,color=GRAY,ls='--',lw=1); b.set_xlim(-7,10); b.set_xlabel('Change (percentage points)')
foot(f,'Legacy protocol: train/evaluation operand pairs overlap; values are not held-out effects.\nOne saved training run per configuration; no independent-training uncertainty is available.')
save(f,'01_results_overview','Legacy descriptive differences across five configurations; the evaluation protocol overlaps training.')

# 2: Matched task, different multi-parameter training configurations.
f=setup('Training stability on 6-digit addition','Full FT and LoRA configurations on the same task; other training hyperparameters also differ',size=(12,9))
gs=f.add_gridspec(3,1,left=.10,right=.96,bottom=.14,top=.82,hspace=.20)
for j,(key,yl) in enumerate([('rewards/correctness_reward/mean','Correctness reward'),('kl','KL'),('grad_norm','Gradient norm')]):
 a=f.add_subplot(gs[j]); curve(a,2,key,ORANGE,'--','Full FT'); curve(a,3,key,BLUE,'-','LoRA'); a.set_ylabel(yl)
 if j==0: a.set_ylim(-.03,1.06); a.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.30))
 if j<2: a.tick_params(labelbottom=False)
 else: a.set_xlabel('Training step')
foot(f,'Thin lines: raw logged values. Thick lines: trailing 20-step mean (starts at step 20).\nGradient norm is shown as logged; comparisons do not isolate the effect of LoRA.')
save(f,'02_training_stability','Compare observed reward, KL and gradient trajectories on the same task.')

# 3: Log values and exact threshold share; no proxy renamed as gradient.
f=setup('When do rewards vary within a group?','Each row shows all 500 logged steps; the right column reports the stated threshold exactly')
gs=f.add_gridspec(1,2,left=.30,right=.96,bottom=.35,top=.79,width_ratios=[4,1.25],wspace=.10)
a=f.add_subplot(gs[0]); b=f.add_subplot(gs[1]); arr=np.array([vals(i,'frac_reward_zero_std') for i in range(5)])
im=a.imshow(arr,aspect='auto',interpolation='nearest',extent=[.5,500.5,4.5,-.5],cmap=ListedColormap([BLUE,'#E5E9EE']),vmin=0,vmax=1)
a.set_yticks(range(5),labels); a.set_xlabel('Training step'); a.set_xticks([1,100,200,300,400,500])
for y in np.arange(.5,4.5): a.axhline(y,color='white',lw=6)
b.set_xlim(0,1); b.set_ylim(4.5,-.5); b.axis('off'); b.set_title('Steps with\nzero-std fraction < 0.99',fontsize=10)
for i in range(5): b.text(.05,i,f'{np.mean(arr[i]<.99)*100:.1f}%',va='center',fontsize=16,weight='bold')
cb=f.colorbar(im,cax=f.add_axes([.30,.22,.48,.025]),orientation='horizontal',ticks=[0,1]); cb.ax.set_xticklabels(['0: reward variation','1: zero reward variation']); cb.set_label('Logged frac_reward_zero_std',fontsize=9)
assert set(np.unique(arr)).issubset({0.,1.}), 'Use continuous colormap for fractional log values'
foot(f,'Reward variation is not a measurement of useful gradients or learning quality. All five experiments are retained.')
save(f,'03_reward_signal','Distribution of logged zero-reward-standard-deviation fractions over training.')

# 4: Evaluation-batch pairing, no training replicate claim.
f=setup('Multiplication: five legacy evaluation partitions','Observed differences | one trained model | five question-generation seeds')
a=f.add_axes([.16,.18,.68,.61]); e=ev[4]
for i,(s,x,y) in enumerate(zip(e['config']['seeds'],e['base_model']['per_seed'],e['trained_model']['per_seed'])):
 x*=100; y*=100; a.plot([x,y],[i,i],color='#BCC5CE',lw=3); a.scatter(x,i,c=GRAY,s=75,marker='o',label='Base' if i==0 else None); a.scatter(y,i,c=BLUE,s=65,marker='D',label='LoRA' if i==0 else None)
 a.text(x-.5,i,f'{x:.1f}%',ha='right',va='center',color=GRAY); a.text(y+.5,i,f'{y:.1f}%',va='center',color=BLUE); a.text(81,i,f'+{y-x:.1f} pp',va='center',weight='bold')
a.set_yticks(range(5),[f'Question seed {s}' for s in e['config']['seeds']]); a.set_ylim(4.6,-.6); a.set_xlim(60,85); a.set_xlabel('Accuracy (%)'); a.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.16))
foot(f,'Legacy protocol overlaps training; seed 42 matches the training prefix 500/500.\nPaired by question-generation seed; not independent training runs or held-out inference.')
save(f,'04_evaluation_batches','All five legacy multiplication evaluation partitions show positive descriptive differences; the protocol overlaps training.')

# 5: Different metrics deliberately use separate axes and labels.
f=setup('Reward recovery does not guarantee evaluation gains','3-digit addition | Full FT | training reward and final greedy evaluation are different measurements')
gs=f.add_gridspec(1,2,left=.09,right=.94,bottom=.21,top=.78,width_ratios=[2,1],wspace=.38)
a=f.add_subplot(gs[0]); curve(a,0,'reward'); a.set_ylabel('Total training reward'); a.set_xlabel('Training step'); a.set_ylim(0,1.27); a.set_title('Training: reward recovers')
b=f.add_subplot(gs[1]); e=ev[0]
for j,k in enumerate(['base_model','trained_model']):
 y=e[k]['mean']*100; b.scatter(j,y,c=[GRAY,BLUE][j],s=95,marker=['o','D'][j]); b.text(j,y+1,f'{y:.2f}%',ha='center',fontsize=13)
b.plot([0,1],[e['base_model']['mean']*100,e['trained_model']['mean']*100],color='#BBC3CC',zorder=0); b.set_xticks([0,1],['Base','Trained']); b.set_xlim(-.5,1.5); b.set_ylim(88,99); b.set_ylabel('Final evaluation accuracy (%)'); b.set_title('Evaluation: −3.20 pp'); grid(b)
foot(f,'Training: raw trajectory + trailing 20-step mean; reward includes correctness and format components (maximum 1.2).\nEvaluation: five batches of 500 questions. No intermediate checkpoint evaluation is available.')
save(f,'05_reward_vs_evaluation','Late training reward does not establish an improvement in final evaluation accuracy.')

# 6: Components share a step axis, with explicitly different natural ranges.
f=setup('Separate correctness from the format bonus','LoRA configurations: 6-digit addition versus 2-digit multiplication',size=(12,9))
gs=f.add_gridspec(3,1,left=.11,right=.96,bottom=.14,top=.81,hspace=.26)
for j,(key,yl,limit) in enumerate([('rewards/correctness_reward/mean','Correctness reward',1.0),('rewards/format_reward/mean','Format reward',.2),('reward','Total reward',1.2)]):
 a=f.add_subplot(gs[j]); curve(a,3,key,ORANGE,'--','6-digit addition'); curve(a,4,key,BLUE,'-','2-digit multiplication'); a.set_ylabel(yl); a.set_ylim(-.04*limit,limit*1.08)
 if j==0: a.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.36))
 if j<2: a.tick_params(labelbottom=False)
 else: a.set_xlabel('Training step')
foot(f,'Thin lines: raw logs. Thick lines: trailing 20-step mean. Panel ranges follow component scales.\nTotal reward is checked against correctness + format reward; reward is not held-out accuracy.')
save(f,'06_reward_components','Correctness and format contributions to total reward are shown separately.')

# Portable source tables and focused data validation.
checks={'steps_complete':all(list(vals(i,'step'))==list(range(1,501)) for i in range(5)),
 'means_match_batches':all(abs(np.mean(e[k]['per_seed'])-e[k]['mean'])<1e-12 for e in ev for k in ['base_model','trained_model']),
 'reward_sum_matches':all(np.allclose(vals(i,'reward'),vals(i,'rewards/correctness_reward/mean')+vals(i,'rewards/format_reward/mean'),atol=1e-6) for i in range(5)),
 'signal_percentages':[float(np.mean(arr[i]<.99)*100) for i in range(5)]}
assert all(checks[k] for k in ['steps_complete','means_match_batches','reward_sum_matches'])
with (ROOT/'training_data.csv').open('w',newline='') as h:
 keys=['step','reward','rewards/correctness_reward/mean','rewards/format_reward/mean','kl','grad_norm','frac_reward_zero_std']; w=csv.writer(h,lineterminator='\n'); w.writerow(['experiment']+keys)
 for name,history in zip(labels,logs):
  for r in history: w.writerow([name]+[r[k] for k in keys])
with (ROOT/'evaluation_data.csv').open('w',newline='') as h:
 w=csv.writer(h,lineterminator='\n'); w.writerow(['experiment','evaluation_seed','base_accuracy','trained_accuracy'])
 for name,e in zip(labels,ev):
  for s,x,y in zip(e['config']['seeds'],e['base_model']['per_seed'],e['trained_model']['per_seed']): w.writerow([name,s,x,y])
(ROOT/'validation.json').write_text(json.dumps(checks,indent=2))
(ROOT/'manifest.json').write_text(json.dumps({'figures':manifest,'sources':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(SRC.glob('*.json'))},'uncertainty':'Legacy descriptive results; train/evaluation operand pairs overlap, with no training-replicate inference or paired question-level CI.','smoothing':'Trailing 20-step arithmetic mean; raw observations retained.','missing_policy':'All 500 steps and all five evaluation batches retained.','randomness':'No random sampling or jitter.'},indent=2))
print(json.dumps(checks)); print(f'Exported {len(ONLY) if ONLY else 6} figures x 3 formats')
