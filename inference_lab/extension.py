"""Counterbalanced model-size/prefix-reuse serving study; RunPod execution only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,os,platform,signal,subprocess,sys,time
from pathlib import Path
from . import pilot
from .benchmark import stream_request

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='inference-lab-factorial-serving-v1'
MODELS=(('qwen-1.5b','Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
        ('qwen-7b','Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28'))
HARDWARE={'rtx4090':'NVIDIA GeForce RTX 4090','a10080':'NVIDIA A100-SXM4-80GB'}
ORDERS=((1,4,16,32),(32,16,4,1),(4,32,1,16),(16,1,32,4))
SEEDS=(31,47,73,101)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(encode(value)).hexdigest()
def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(encode(value)+b'\n');tmp.replace(p)


def workload(replicate,kind,concurrency,warmup=False):
    if kind not in ('shared','unique'):raise ValueError('Unknown prefix condition')
    words=' '.join(pilot.SENTENCES).split();rows=[]
    for length in (64,192,448,896):
        context=' '.join((words*(length//len(words)+1))[:length])
        for index in range(1 if warmup else 16):
            identity=f'r{replicate}-{kind}-c{concurrency}-w{length}-{index}'
            namespace=f'warmup-{identity}' if warmup else identity
            # Fresh prefix across stages/replicates; shared only within this stage/length.
            prefix=f'Warmup reference {namespace}.\n' if warmup else (
                f'Shared reference r{replicate} c{concurrency} w{length}.\n' if kind=='shared'
                else f'Independent reference {hashlib.sha256(namespace.encode()).hexdigest()}.\n')
            rows.append({'id':namespace,'prompt':prefix+context+'\n\n'+f'Task {index}: summarize the supplied operations, records and checks. Use only these notes.'})
    return rows


def cells():
    rows=[]
    for rep,seed in enumerate(SEEDS,1):
        models=MODELS if rep in (1,4) else tuple(reversed(MODELS))
        modes=(False,True) if rep in (1,4) else (True,False)
        kinds=('shared','unique') if rep%2 else ('unique','shared')
        for alias,model,revision in models:
            for kind in kinds:
                for cache in modes:
                    rows.append({'id':f'r{rep}-{alias}-{kind}-cache-{int(cache)}','replicate':rep,'seed':seed,
                                 'model_alias':alias,'model':model,'revision':revision,'prefix_kind':kind,
                                 'prefix_cache':cache,'concurrency':list(ORDERS[rep-1])})
    return rows


def plan():
    rows=cells()
    hashes={f'r{r}-{k}-c{c}':{'measured':sha(workload(r,k,c)),'warmup':sha(workload(r,k,c,True))}
            for r in range(1,5) for k in ('shared','unique') for c in (1,4,16,32)}
    return {'schema':SCHEMA,'image':pilot.IMAGE,'vllm_version':'0.10.2','models':[{'id':m,'revision':r,'license':'Apache-2.0'} for _,m,r in MODELS],
            'hardware':HARDWARE,'cells_per_hardware':rows,'workload_hashes':hashes,'measured_requests_per_stage':64,'warmups_per_stage':4,
            'planned_measured_requests_per_hardware':len(rows)*4*64,'planned_measured_requests_total':len(rows)*4*64*2,
            'max_output_tokens':128,'temperature':0,'server_seed':17,'max_model_len':4096,'gpu_memory_utilization':.8,
            'request_timeout_seconds':90,'server_ready_timeout_seconds':720,'experiment_wall_seconds':10800,
            'allocation_wall_seconds':14400,'campaign_gpu_cost_cap_usd':18,
            'cache_policy':'Fresh server per model/prefix/replicate/cache cell. Stage-specific prefixes prevent cross-stage text reuse except shared chat-template tokens. Warmups use disjoint prompts.',
            'order':'Models/cache order ABBA across four repeats; prefix order alternates; paired modes share stage/workload/seed.',
            'metrics':['task_request_success','first_text_chunk_latency','p50_p95_request_latency','completion_tokens_per_second','token_usage','device_utilization','device_memory','paired_output_hash_agreement'],
            'limitations':['Fixed-length synthetic summaries, not production traffic or answer-quality evaluation.','One allocated instance per hardware class; repeated requests are not independent GPU installations.','Requested deterministic decoding does not guarantee identical text; report output differences.','Two sizes in one model family, no cross-family or quantization/speculative-decoding claim.','All missing/failing cells remain explicit; no selective repetitions after measured outcomes.']}


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'inference_lab').glob('*.py'))}


def freeze(output,commit):
    value={'schema':SCHEMA,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,'sources':source_hashes(),'configuration':plan()}
    value['sha256']=sha(value)
    with output.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    return value


def server_command(cell):
    args=pilot.server_command(cell['prefix_cache'])
    for key,value in [('--model',cell['model']),('--revision',cell['revision']),('--tokenizer-revision',cell['revision'])]:args[args.index(key)+1]=value
    return args


def ready(server,model,deadline):
    end=min(deadline,time.monotonic()+720)
    while time.monotonic()<end:
        if server.poll() is not None:raise RuntimeError('Model server exited before readiness')
        try:
            if any(v.get('id')==model for v in json.loads(pilot.local_get('/v1/models')).get('data',[])):return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise TimeoutError('Model readiness timeout')


def run(output,frozen,hardware):
    if platform.system()!='Linux' or not os.environ.get('RUNPOD_POD_ID'):raise ValueError('Real inference runs only on the reviewed RunPod pod')
    if frozen['sha256']!=sha({k:v for k,v in frozen.items() if k!='sha256'}) or frozen['sources']!=source_hashes() or frozen['configuration']!=plan():raise ValueError('Frozen inputs/source/configuration changed')
    if importlib.metadata.version('vllm')!='0.10.2':raise ValueError('Wrong vLLM runtime')
    if hardware not in HARDWARE:raise ValueError('Unreviewed hardware')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'freeze.json',frozen)
    devices=pilot.capture(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],output/'gpu-identity.json')
    if devices.returncode or len(devices.stdout.strip().splitlines())!=1 or HARDWARE[hardware] not in devices.stdout:raise ValueError('Actual GPU differs from protocol')
    versions={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','huggingface-hub']}
    write(output/'environment.json',{'packages':versions,'python':platform.python_version(),'hardware':hardware,'source_hashes':source_hashes()})
    try:pilot.local_get('/health')
    except OSError:pass
    else:raise RuntimeError('Unknown server already occupies loopback port8000')
    manifest={'schema':SCHEMA,'hardware':hardware,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'cells':[],'active_cell':None,'active_stage':None}
    write(output/'manifest.json',manifest);deadline=time.monotonic()+10800
    server=telemetry=client=None
    previous_alarm=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Experiment wall limit')))
    signal.alarm(10800)
    try:
        for cell in frozen['configuration']['cells_per_hardware']:
            if time.monotonic()>=deadline:raise TimeoutError('Experiment deadline')
            folder=output/cell['id'];folder.mkdir();manifest.update(active_cell=cell['id'],active_stage=None);write(output/'manifest.json',manifest)
            command=server_command(cell);metadata={**cell,'hardware':hardware,'image_expected':pilot.IMAGE,'launch_command':command,'packages_measured':versions,'gpu_identity_measured':devices.stdout}
            write(folder/'server-settings.json',metadata)
            with (folder/'server.log').open('w') as log,(folder/'gpu.csv').open('w') as gpu,(folder/'gpu.stderr').open('w') as err:
                gpu.write(pilot.GPU_FIELDS+'\n');gpu.flush()
                telemetry=subprocess.Popen(['nvidia-smi',f'--query-gpu={pilot.GPU_FIELDS}','--format=csv,noheader,nounits','-lms','500'],stdout=gpu,stderr=err,start_new_session=True,env={**os.environ,'TZ':'UTC'})
                server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1'})
                ready(server,cell['model'],deadline)
                if telemetry.poll() is not None:raise RuntimeError('GPU telemetry unavailable')
                for concurrency in cell['concurrency']:
                    stage=folder/f'c{concurrency}';stage.mkdir();manifest['active_stage']=concurrency;write(output/'manifest.json',manifest)
                    data=workload(cell['replicate'],cell['prefix_kind'],concurrency)
                    warm=workload(cell['replicate'],cell['prefix_kind'],concurrency,True)
                    (stage/'workload.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
                    (stage/'warmup-workload.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in warm))
                    with (stage/'warmups.jsonl').open('w') as handle:
                        for row in warm:
                            result=stream_request('http://127.0.0.1:8000/v1',cell['model'],row['prompt'],128,90,None,True)
                            handle.write(json.dumps({'workload_id':row['id'],'warmup':True,**result})+'\n');handle.flush()
                    (stage/'metrics-before.txt').write_bytes(pilot.local_get('/metrics'))
                    args=[sys.executable,'-m','inference_lab.benchmark','--model',cell['model'],'--workload',str(stage/'workload.jsonl'),'--concurrency',str(concurrency),'--requests','64','--warmup','0','--max-tokens','128','--fixed-output-tokens','--timeout','90','--seed',str(cell['seed']),'--server-metadata',str(folder/'server-settings.json'),'--output',str(stage/'client'),'--execute']
                    with (stage/'client.log').open('w') as client_log:
                        client=subprocess.Popen(args,cwd=ROOT,stdout=client_log,stderr=subprocess.STDOUT,start_new_session=True)
                        code=client.wait(timeout=max(1,deadline-time.monotonic()))
                    if code not in (0,1) or not (stage/'client/summary.json').exists():raise RuntimeError('Client did not retain complete stage records')
                    (stage/'metrics-after.txt').write_bytes(pilot.local_get('/metrics'))
                    if telemetry.poll() is not None:raise RuntimeError('GPU telemetry stopped during measured stage')
                    print(json.dumps({'hardware':hardware,'cell':cell['id'],'concurrency':concurrency,'client_status':code}),flush=True)
                pilot.stop_owned(server);server=None;pilot.stop_owned(telemetry);telemetry=None
            manifest['cells'].append(cell['id']);manifest.update(active_cell=None,active_stage=None);write(output/'manifest.json',manifest)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='incomplete',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous_alarm)
        for process in (client,server,telemetry):pilot.stop_owned(process)
        manifest['finished_at']=datetime.now(timezone.utc).isoformat();write(output/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--freeze',type=Path);g.add_argument('--execute',action='store_true')
    p.add_argument('--frozen',type=Path);p.add_argument('--hardware',choices=HARDWARE);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.freeze:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for rel,expected in source_hashes().items():
            if hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)).hexdigest()!=expected:raise ValueError('Uncommitted source')
        value=freeze(a.freeze,commit);print(json.dumps({'frozen':True,'sha256':value['sha256'],'planned_requests':value['configuration']['planned_measured_requests_total']}));return
    if not a.execute:print(json.dumps(plan(),indent=2));return
    if not a.frozen or not a.hardware or not a.output:p.error('Frozen protocol, exact hardware and new output required')
    def interrupted(*_):raise KeyboardInterrupt('Supervisor requested stop')
    signal.signal(signal.SIGTERM,interrupted)
    run(a.output.resolve(),json.loads(a.frozen.read_text()),a.hardware)
if __name__=='__main__':main()
