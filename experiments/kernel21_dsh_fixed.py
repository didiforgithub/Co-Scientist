#!/usr/bin/env python3
import argparse,json,os,shutil,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--task',required=True); p.add_argument('--gpu',type=int,default=0); p.add_argument('--agents',type=int,default=1); p.add_argument('--hours',type=float,default=4); p.add_argument('--evaluator-root',required=True); p.add_argument('--run-root',required=True); a=p.parse_args()
 src=Path(a.evaluator_root)/a.task; run=Path(a.run_root)/a.task; run.mkdir(parents=True,exist_ok=True)
 if not (run/'verifier.py').exists():
  for x in src.iterdir():
   if x.name=='__pycache__': continue
   (shutil.copytree if x.is_dir() else shutil.copy2)(x,run/x.name,dirs_exist_ok=True) if x.is_dir() else shutil.copy2(x,run/x.name)
  seed=next(Path(a.evaluator_root).parents[2].joinpath('20260922_kernel21_search42_r1/runs').glob(f'*__gt__{a.task}/bootstrap_ws/seed_solution.json'))
  shutil.copy2(seed,run/'solution_out.json')
 if not (run/'solution_out.json').exists():
  seed=next(Path(a.evaluator_root).parents[2].joinpath('20260922_kernel21_search42_r1/runs').glob(f'*__gt__{a.task}/bootstrap_ws/seed_solution.json')); shutil.copy2(seed,run/'solution_out.json')
  v=run/'verifier.py'; s=v.read_text(); needle='            cache_dir.mkdir(mode=0o777)\n';
  if 'cache_dir / "extensions"' not in s: s=s.replace(needle,needle+'            (cache_dir / "extensions").mkdir(mode=0o777)\n',1); v.write_text(s)
 docker='/mdata/zhangjiayi/cosci-h200-runtime/bin/docker'; end=time.time()+a.hours*3600; best=None; best_score=float('-inf'); gen=0
 state=run/'blackboard.json'
 if state.exists():
  try:
   prev=json.loads(state.read_text()); gen=int(prev.get('generation',0)); best_score=float(prev.get('best_score',best_score))
   if (run/'solution_out.json').exists(): best=json.loads((run/'solution_out.json').read_text())
  except (OSError,ValueError,TypeError): pass
 def ev(ws):
  q=subprocess.run([docker,"run","--rm","--gpus",f"device={a.gpu}","-v",f"{ws}:/work","-w","/work","cosci-k16-h200:20260907","python3","/work/run_verify.py","/work/solution_out.json"],capture_output=True,text=True,timeout=3500,env={**os.environ,"DOCKER_HOST":"unix:///var/run/cosci-h200-docker.sock"}); out_text=q.stdout.strip()
  try:
   return json.loads(out_text)
  except Exception:
   decoder=json.JSONDecoder()
   for pos in [j for j,c in enumerate(out_text) if c==chr(123)]:
    try: return decoder.raw_decode(out_text[pos:])[0]
    except Exception: pass
   return {"raw":None,"feasible":False,"error":(q.stderr or q.stdout)[-1000:]}
 while time.time()<end-20:
  gen+=1; cand=[]
  for i in range(a.agents):
   ws=run/f'solver_ws_{i}'; shutil.rmtree(ws,ignore_errors=True); shutil.copytree(run,ws,ignore=shutil.ignore_patterns('solver_ws_*','*.log','blackboard.json'))
   (ws/'BLACKBOARD.md').write_text(json.dumps({'generation':gen,'best_score':best_score,'best':best}))
   prompt=f'You are DSH Solver agent {i+1}/{a.agents} for fixed-GT Kernel task {a.task}, generation {gen}. Read README.md, sample_solution.json, seed_solution.json, and BLACKBOARD.md. Improve candidate_py with legitimate CUDA/Triton/PyTorch optimization preserving ModelNew API. Write {{"candidate_py":<full source>}} to solution_out.json and run ./container-eval solution_out.json. Record score/reasoning in NOTES_FOR_PEERS.md. Do not inspect or modify verifier.py or checker internals. Current best score: {best_score}.'
   env={**os.environ,'PATH':'/root/.local/bin:'+os.environ.get('PATH',''),'HOME':'/root','DEEPSEEK_API_KEY':os.environ.get('DEEPSEEK_API_KEY',''),'DEEPSEEK_BASE_URL':'http://api.llm.will/v1','HTTP_PROXY':'http://172.169.24.13:18118','HTTPS_PROXY':'http://172.169.24.13:18118','ALL_PROXY':'http://172.169.24.13:18118'}
   try: subprocess.run(['/root/.local/bin/dsh','--profile','headless','--',prompt],cwd=ws,env=env,capture_output=True,text=True,timeout=min(900,max(30,end-time.time())))
   except subprocess.TimeoutExpired: pass
   r=ev(ws); score=r.get('raw'); score=float(score) if score is not None else float('-inf'); cand.append((score,ws,r))
  score,ws,r=max(cand,key=lambda z:z[0])
  if score>best_score and (ws/'solution_out.json').exists(): best_score=score; best=json.loads((ws/'solution_out.json').read_text()); (run/'solution_out.json').write_text(json.dumps(best))
  (run/'blackboard.json').write_text(json.dumps({'generation':gen,'best_score':best_score,'last':r},indent=2)); (run/'dsh_solver.log').open('a').write(json.dumps({'generation':gen,'best_score':best_score,'last':r})+'\n')
 print(json.dumps({'task':a.task,'generations':gen,'best_score':best_score,'run':str(run)}))
if __name__=='__main__': main()
