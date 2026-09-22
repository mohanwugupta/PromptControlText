import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('micro',Path(__file__).resolve().parents[1]/'experiments/run_judge_micro_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MicroTests(unittest.TestCase):
    def test_payload_is_output_only_and_strict(self):
        t=dict(model=m.MODEL,system_prompt='Classify visible behavior.',input_text='assistant_response:\nHello.',reasoning_effort='none',max_output_tokens=512)
        p=m.payload(t)
        self.assertEqual(p['input'][1]['content'],t['input_text'])
        self.assertEqual(p['reasoning'],{'effort':'none'})
        self.assertTrue(p['text']['format']['strict'])
        self.assertGreater(m.reservation(t),512*1.25/1e6)

    def test_missing_ledger_blocks_archive_not_empty_preparation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m,'OUT',Path(d)):
            (Path(d)/'usage.jsonl').write_text('')
            m.validate_resume_state()
            (Path(d)/'usage.jsonl').write_text('{}\n')
            with self.assertRaises(RuntimeError):m.validate_resume_state()

if __name__=='__main__':unittest.main()
