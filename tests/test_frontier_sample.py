import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import run_frontier_sample as m


def task(i=0,native=None,controller=None):
    return dict(task_id=str(i),item_id=str(i),benchmark='IHEval' if native else 'XSTest',domain='test',gold_label='safe',input_text='Test user text',native_system=native,controller=controller,prompt_family='No-controller',clarity_level='none',prompt_variant='control',model_name=m.MODEL)

class FrontierSampleTests(unittest.TestCase):
    def test_advance_waits_for_existing_batch(self):
        with patch.object(m,'poll',return_value={}),patch.object(m,'state',return_value={'batches':[{'reconciled':False}]}),patch.object(m,'submit') as submit:
            self.assertEqual(m.advance({'tasks':[]})['next_action'],'wait')
            submit.assert_not_called()

    def test_advance_stops_on_billing_error(self):
        with patch.object(m,'poll',return_value={}),patch.object(m,'state',return_value={'batches':[]}),patch.object(m,'rows',return_value=[{'error_code':'credit_balance_exhausted'}]),patch.object(m,'submit') as submit:
            self.assertEqual(m.advance({'tasks':[]})['next_action'],'billing_attention_required')
            submit.assert_not_called()

    def test_advance_sizes_chunk_to_remaining_cap(self):
        with patch.object(m,'poll',return_value={}),patch.object(m,'state',return_value={'batches':[]}),patch.object(m,'rows',return_value=[]),patch.object(m,'pending',return_value=[task(i) for i in range(5)]),patch.object(m,'CAP',.25),patch.object(m,'reservation',return_value=.1),patch.object(m,'submit',return_value={}) as submit:
            m.advance({'tasks':[]})
            self.assertEqual(submit.call_args.args[1],2)

    def test_baseline_role_semantics(self):
        self.assertEqual([x['role'] for x in m.payload(task())['input']],['user'])
        p=m.payload(task(native='Native task rules'))
        self.assertEqual(p['input'][0],{'role':'system','content':'Native task rules'})
        p=m.payload(task(native='Native task rules',controller='Registry controller'))
        self.assertEqual(p['input'][0]['content'],'Native task rules\n\nRegistry controller')

    def test_batch_cost_and_reservation(self):
        self.assertAlmostEqual(m.usage_cost({'input_tokens':1000,'output_tokens':200}),.01)
        self.assertGreater(m.reservation(task()),2048*25/1e6)

    def test_ambiguous_post_cannot_duplicate(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m,'OUT',Path(d)),patch.object(m,'api',side_effect=[{'id':'file-test'},TimeoutError()]):
            with self.assertRaises(TimeoutError):m.submit({'tasks':[task()]},1)
            self.assertGreater(m.total(m.state()),0)
            with self.assertRaisesRegex(RuntimeError,'Reconcile'):m.submit({'tasks':[task()]},1)

    def test_interrupted_upload_requeues_without_inference(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)),patch.object(m,'api',side_effect=TimeoutError) as api:
            manifest={'tasks':[task()]}
            with self.assertRaises(TimeoutError):m.submit(manifest,1)
            m.poll(manifest)
            self.assertEqual(api.call_count,1)
            self.assertEqual(m.total(m.state()),0)
            self.assertEqual(len(m.pending(manifest,m.state())),1)

    def test_budget_before_network(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)),patch.object(m,'CAP',.001),patch.object(m,'api') as api:
            with self.assertRaisesRegex(RuntimeError,'cap'):m.submit({'tasks':[task()]},1)
            api.assert_not_called()

    def test_export_omits_incomplete_and_errors_from_judge(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)):
            rr=[{'task_id':'0','benchmark':'XSTest','item_id':'0','status':'completed','model_output':'Answer'},
                {'task_id':'1','benchmark':'XSTest','item_id':'1','status':'incomplete','model_output':'Partial'},
                {'task_id':'2','benchmark':'XSTest','item_id':'2','status':'api_error','error_code':'cyber_policy'}]
            (Path(d)/'responses.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
            report=m.export({'tasks':[task(i) for i in range(3)]})
            self.assertEqual(report['ready_for_original_judge'],1)
            self.assertNotIn('Answer',(Path(d)/'usage.jsonl').read_text())

if __name__=='__main__':unittest.main()
