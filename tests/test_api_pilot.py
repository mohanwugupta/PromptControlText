"""Offline checks for paid-request controls; never sends network requests."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('api_pilot', Path(__file__).resolve().parents[1] / 'experiments/run_api_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def task(i=0):
    return dict(task_id=str(i), model=pilot.MODELS[1], system_prompt=None,
                input_text='Explain how rainbows form.', reasoning_effort='low', max_output_tokens=2048)


class PilotTests(unittest.TestCase):
    def test_public_archive_cannot_repeat_paid_requests(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            (directory / 'usage.jsonl').write_text('{}\n')
            with patch.object(pilot, 'OUT', directory):
                with self.assertRaisesRegex(RuntimeError, 'Refusing to repeat'):
                    pilot.execute({}, 1)

    def test_baseline_omits_system(self):
        self.assertEqual([x['role'] for x in pilot.payload(task())['input']], ['user'])

    def test_prompt_preserved(self):
        t = task(); t['system_prompt'] = 'Ask for clarification.'
        self.assertEqual(pilot.payload(t)['input'][0], {'role': 'system', 'content': t['system_prompt']})

    def test_concurrent_reservations_enforce_cap(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = pilot.Ledger(Path(d))
            with ThreadPoolExecutor(max_workers=16) as pool:
                accepted = list(pool.map(ledger.reserve, [task(i) for i in range(200)]))
            self.assertLessEqual(ledger.total(), 10)
            self.assertLess(sum(accepted), 200)
            self.assertEqual(pilot.Ledger(Path(d)).total(), ledger.total())

    def test_unknown_failure_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = pilot.Ledger(Path(d)); ledger.reserve(task())
            amount = ledger.total()
            ledger.settle(task(), {'status': 'transport_or_parse_error'})
            self.assertEqual(ledger.total(), amount)
            self.assertFalse(ledger.reserve(task(), retry_rejected=True))

    def test_credit_rejection_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = pilot.Ledger(Path(d)); ledger.reserve(task())
            ledger.settle(task(), {'status': 'http_error', 'http_status': 429, 'error_code': 'credit_balance_exhausted'})
            self.assertEqual(ledger.total(), 0)
            self.assertFalse(ledger.reserve(task()))
            self.assertTrue(ledger.reserve(task(), retry_rejected=True))

    def test_completed_requests_not_duplicated(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = pilot.Ledger(Path(d)); ledger.reserve(task())
            ledger.settle(task(), {'status': 'completed', 'estimated_cost_usd': 0.01})
            self.assertEqual(ledger.total(), 0.01)
            self.assertFalse(pilot.Ledger(Path(d)).reserve(task(), retry_rejected=True))


if __name__ == '__main__':
    unittest.main()
