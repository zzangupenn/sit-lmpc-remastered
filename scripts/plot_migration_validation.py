"""Plot the bounded real-GPU migration checks, not a performance benchmark."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parents[1]


def main():
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    records={}
    for column,(folder,label) in enumerate([('validation-st-smoke','Known-dynamics SIT-LMPC'),('validation-bnn-chance-smoke','BNN posterior + chance')]):
        run=R/'results'/folder
        laps=json.loads((run/'lap_metrics.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        paths=sorted((run/'trajectories/success').glob('*.npz'))
        for i,path in enumerate(paths):
            with np.load(path) as data:
                xy=data['cartesian'][:,:2]
                axes[0,column].plot(xy[:,0],xy[:,1],lw=1,label=f'Lap {i+1}')
        axes[0,column].set(title=label,xlabel='x (m)',ylabel='y (m)',aspect='equal')
        ids=[v['attempt_id'] for v in laps]
        axes[1,column].bar(ids,[v['lap_time'] for v in laps],color=['#888888','#888888','#2464ad'])
        axes[1,column].set(xlabel='Trial',ylabel='Lap time (s)',xticks=ids,title=f"{summary['learned_steps']} learned-control steps; {summary['failed_attempts']} failures")
        axes[0,column].legend(fontsize=8)
        for panel in axes[:,column]:panel.grid(alpha=.2)
        records[label]={key:summary[key] for key in ('status','successful_laps','failed_attempts','applied_steps','learned_steps','training_generations','dynamics_belief_generation')}
        records[label]['elapsed_seconds']=summary['elapsed']
        records[label]['lap_seconds']=[v['lap_time'] for v in laps]
    fig.suptitle('Figure 1. Bounded migration checks: warm-up, value learning and learned control')
    fig.tight_layout(rect=(0,0,1,.95))
    target=R/'docs/figures';target.mkdir(exist_ok=True)
    fig.savefig(target/'migration-racecar-functional-checks.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (R/'docs/REAL_GPU_VALIDATION.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
