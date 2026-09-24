from collections import Counter
import unittest
from inference_lab import extension

class ExtensionTests(unittest.TestCase):
 def test_factorial_matrix_and_counterbalancing(self):
  p=extension.plan();cells=p['cells_per_hardware']
  self.assertEqual(len(cells),32);self.assertEqual(p['planned_measured_requests_total'],16384)
  self.assertEqual(len({c['id'] for c in cells}),32)
  self.assertEqual(set(Counter((c['model'],c['prefix_kind'],c['prefix_cache']) for c in cells).values()),{4})
  for rep in range(1,5):
   for alias,_,_ in extension.MODELS:
    for kind in ['shared','unique']:
     pair=[c for c in cells if c['replicate']==rep and c['model_alias']==alias and c['prefix_kind']==kind]
     self.assertEqual(len(pair),2);self.assertEqual(pair[0]['concurrency'],pair[1]['concurrency']);self.assertEqual(pair[0]['seed'],pair[1]['seed'])
     self.assertEqual([c['prefix_cache'] for c in pair],[False,True] if rep in [1,4] else [True,False])
 def test_unique_prefixes_and_disjoint_warmup_stage_and_replication(self):
  seen=set()
  for rep in range(1,5):
   for kind in ['shared','unique']:
    for concurrency in [1,4,16,32]:
     rows=extension.workload(rep,kind,concurrency);warm=extension.workload(rep,kind,concurrency,True)
     prompts={r['prompt'] for r in rows};warm_prompts={r['prompt'] for r in warm}
     self.assertEqual(len(prompts),64);self.assertFalse(prompts&warm_prompts);self.assertFalse(prompts&seen);seen.update(prompts)
     prefixes={r['prompt'].split('\n')[0] for r in rows}
     self.assertEqual(len(prefixes),4 if kind=='shared' else 64)
 def test_only_model_and_cache_arguments_change_and_no_remote_host(self):
  for cell in extension.cells():
   cmd=extension.server_command(cell)
   self.assertEqual(cmd[cmd.index('--model')+1],cell['model'])
   self.assertEqual(cmd[cmd.index('--revision')+1],cell['revision'])
   self.assertEqual(cmd[cmd.index('--host')+1],'127.0.0.1')
   self.assertEqual(cmd[-1],'--enable-prefix-caching' if cell['prefix_cache'] else '--no-enable-prefix-caching')
 def test_frozen_plan_is_deterministic_and_bounded(self):
  self.assertEqual(extension.plan(),extension.plan())
  self.assertEqual(extension.plan()['allocation_wall_seconds'],14400)
  self.assertEqual(extension.plan()['campaign_gpu_cost_cap_usd'],18)
  self.assertEqual(len(extension.plan()['workload_hashes']),32)
if __name__=='__main__':unittest.main()
