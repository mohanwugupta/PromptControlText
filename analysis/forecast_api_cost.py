"""Offline planning scenarios; no credentials, network, or paid requests."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/frontier_pilot'
COUNTS = {'HarmBench': 400, 'XSTest': 450, 'IHEval': 2520}


def forecast(rows):
    paid = [r for r in rows if 'usage' in r]
    models = sorted({r['model'] for r in paid})
    # Hypothetical GPT-5.4 mini judge. Token counts are assumptions, not measurements.
    judge_per_response = (1200 * 0.75 + 200 * 4.50) / 1e6 * 3.2
    scenarios = []
    for model in models:
        rr = [r for r in paid if r['model'] == model]
        weighted = sum(n * 73 * mean(r['estimated_cost_usd'] for r in rr if r['benchmark'] == b)
                       for b, n in COUNTS.items())
        pooled = sum(COUNTS.values()) * 73 * mean(r['estimated_cost_usd'] for r in rr)
        judges = sum(COUNTS.values()) * 73 * judge_per_response
        scenarios.append(dict(model=model, planned_responses=246010,
                              generation_benchmark_weighted_usd=weighted,
                              generation_pooled_sensitivity_usd=pooled,
                              assumed_judging_usd=judges,
                              total_with_25_percent_buffer_usd=[(weighted+judges)*1.25,(pooled+judges)*1.25],
                              batch_total_with_buffer_usd=[(weighted+judges)*0.625,(pooled+judges)*0.625]))
    return {'full_items_by_benchmark':COUNTS,'conditions':73,
            'judge_assumptions':{'model':'gpt-5.4-mini','input_tokens_per_call':1200,
                'output_tokens_per_call':200,'input_usd_per_million':0.75,
                'output_usd_per_million':4.5,'calls_per_response':3.2,
                'explanation':'Three judge prompts plus one adjudication for an assumed 20% of responses; unvalidated.'},
            'scenarios':scenarios,
            'reduced_100_items_both_models':{
                'planned_responses':14600,
                'generation_pooled_usd':sum(mean(r['estimated_cost_usd'] for r in paid if r['model']==m)*7300 for m in models),
                'assumed_judging_usd':14600*judge_per_response},
            'limitations':['These scenarios are not confidence intervals or price guarantees.',
                'Correcting omitted dataset context may increase token use materially.',
                'Policy blocks and transport failures bias observed response lengths.',
                'Batch discount assumes eligible models and endpoints; runner does not implement Batch.',
                'No human labor, storage, or self-hosted judge costs included.']}


if __name__ == '__main__':
    result=forecast([json.loads(s) for s in (OUT/'usage.jsonl').read_text().splitlines()])
    (OUT/'cost_forecast.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
