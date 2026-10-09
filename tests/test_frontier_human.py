"""SYNTHETIC SOFTWARE FIXTURES ONLY. No human annotations or study findings."""
import csv
import json
import math
import shutil
import sqlite3
from pathlib import Path

import pytest

from audit.frontier.common import (MODEL, VERSION, LABELS, ROOT, Store, sha, stamp, read,
                                   verified, immutable, check_bundle, rubric_hashes, validate_export)
from audit.frontier.sampling import (practice, representative, targeted, adjudication_bundle,
                                    import_judge, judge_input, make_kit, package, allocation)
from audit.frontier.analysis import agreement, confusion, analyze, cluster_intervals


def frame(complete=True):
    rows=[]
    for provider in ['p0','p1','p2']:
        for c in range(4):
            for i in range(12):
                text='SYNTHETIC response '+provider+' '+str(c)+' '+str(i)
                if c==0 and i in (0,1):text='SYNTHETIC identical response'
                rows.append({'request_id':sha(provider+str(c)+':'+str(i)), 'response_text':text,'response_sha256':sha(text),
                             'provider':provider,'benchmark':'benchmark'+str(i%2),'stratum':'stratum'+str(i%2),
                             'item_id':'item'+str(i),'parent_id':'benchmark'+str(i%2)+':item'+str(i),
                             'condition_id':'condition'+str(c),'family':'family','truncated':False})
    return stamp({'version':VERSION,'complete':complete,'rows':rows,'rubric':rubric_hashes(),'provenance':{'fixture':'synthetic'}})


def rating(text, label='compliance', uncertain=False):
    return {'primary_label':label,'confidence':4,'uncertain':uncertain,'evidence':text[:40],
            'reason':'SYNTHETIC unit-test rating','elapsed_seconds':12}


def exports(tmp_path, bundle, disagree=False):
    results=[]
    for coder in ['a','b']:
        st=Store(tmp_path/(coder+'.sqlite'),bundle,coder)
        for i,r in enumerate(bundle['rows']):
            label=LABELS[(i+(1 if coder=='b' and disagree else 0))%len(LABELS)]
            st.save(r['audit_id'],rating(r['response_text'],label))
        st.seal();results.append(st.export());st.db.close()
    return results


def judges(f):
    return stamp({'model':MODEL,'frame_checksum':f['checksum'],'rubric':f['rubric'],
                  'rows':[{'request_id':r['request_id'],'response_sha256':r['response_sha256'],
                           'primary_label':'compliance','confidence':.7,'needs_review':False,
                           'votes':['compliance','refusal','compliance']} for r in f['rows']]})


def test_deterministic_separation_inclusion_coverage():
    f=frame();cal,b=practice(f,'cal',6);m,rb=representative(f,cal,'eval',12)
    assert (m,rb)==representative(f,cal,'eval',12)
    reverse=dict(f,rows=list(reversed(f['rows'])));reverse.pop('checksum');reverse=stamp(reverse)
    assert [r['request_id'] for r in m['rows']]==[r['request_id'] for r in representative(reverse,cal,'eval',12)[0]['rows']]
    assert len(m['rows'])==36
    assert {r['response_sha256'] for r in cal['rows']}.isdisjoint(r['response_sha256'] for r in m['rows'])
    for provider in ['p0','p1','p2']:
        subset=[r for r in m['rows'] if r['provider']==provider]
        assert len(subset)==12
        assert len({r['condition_id'] for r in subset})==4
    for row in m['rows']:
        h=next(x for x in m['settings']['strata'] if x['provider']==row['provider'] and x['condition_id']==row['condition_id'])
        assert row['inclusion_probability']==h['n']/h['N']
        assert row['weight']*row['inclusion_probability']==pytest.approx(1)
    assert sum(r['weight'] for r in m['rows'])==pytest.approx(m['settings']['eligible_N'])


def test_representative_rejects_partial_and_impossible_size():
    f=frame(False);c,_=practice(f,'x',3)
    with pytest.raises(ValueError,match='final complete'):representative(f,c,'x',12)
    with pytest.raises(ValueError):allocation({'a':2,'b':4},1)
    with pytest.raises(ValueError):allocation({'a':2},3)
    assert allocation({'a':1,'b':4},5)=={'a':1,'b':4}


def test_benchmark_sample_can_be_smaller_than_condition_count():
    f=frame();cal,_=practice(f,'cal',6)
    m,b=representative(f,cal,'small',3,stratify='benchmark')
    assert len(m['rows'])==9 and m['settings']['stratify']=='benchmark'
    assert all(set(r)=={'audit_id','response_text'} for r in b['rows'])
    for provider in ['p0','p1','p2']:
        rows=[r for r in m['rows'] if r['provider']==provider]
        assert len(rows)==3 and len({r['benchmark'] for r in rows})==2
        assert m['settings']['coverage'][provider]['missing_conditions']
        for r in rows:
            h=next(h for h in m['settings']['strata'] if h['provider']==provider and h['benchmark']==r['benchmark'])
            assert r['inclusion_probability']==h['n']/h['N']
            assert r['weight']==h['N']/h['n']
    assert sum(r['weight'] for r in m['rows'])==pytest.approx(m['settings']['eligible_N'])
    assert (m,b)==representative(f,cal,'small',3,stratify='benchmark')
    reverse=dict(f,rows=list(reversed(f['rows'])));reverse.pop('checksum');reverse=stamp(reverse)
    other,_=representative(reverse,cal,'small',3,stratify='benchmark')
    assert [r['request_id'] for r in m['rows']]==[r['request_id'] for r in other['rows']]
    with pytest.raises(ValueError,match='stratum'):representative(f,cal,'small',3,stratify='condition')
    with pytest.raises(ValueError,match='stratification'):representative(f,cal,'small',3,stratify='unknown')


def test_reduced_practice_preserves_prior_exclusions_and_targeted_provenance():
    f=frame();old,_=practice(f,'practice',12);cal,_=practice(f,'practice',6)
    assert {r['request_id'] for r in cal['rows']} < {r['request_id'] for r in old['rows']}
    m,_=representative(f,cal,'small',6,stratify='benchmark',prior_practice=[old])
    exposed={r['response_sha256'] for r in old['rows']}
    assert exposed.isdisjoint(r['response_sha256'] for r in m['rows'])
    t,_=targeted(f,cal,m,judges(f),'target',20,prior_practice=[old])
    assert exposed.isdisjoint(r['response_sha256'] for r in t['rows'])
    assert {r['response_sha256'] for r in m['rows']}.isdisjoint(r['response_sha256'] for r in t['rows'])
    assert t['sources']['prior_practice']==[old['checksum']]
    with pytest.raises(ValueError,match='Prior practice'):targeted(f,cal,m,judges(f),'target',20)
    bad=json.loads(json.dumps(old));bad['rows'][0]['request_id']='foreign';bad.pop('checksum')
    with pytest.raises(ValueError,match='map'):representative(f,cal,'bad',6,'benchmark',[stamp(bad)])
    bad=json.loads(json.dumps(old));bad['rows'][0]['response_text']='SYNTHETIC altered';bad.pop('checksum')
    with pytest.raises(ValueError,match='map'):representative(f,cal,'bad',6,'benchmark',[stamp(bad)])


def test_benchmark_sampling_rejects_unrepresented_provider_after_exclusions():
    f=frame();old,_=practice(f,'old',3)
    rows=[r for r in f['rows'] if r['provider']=='p0']
    all_p0,_=package(rows,'practice','all-p0',f,{})
    with pytest.raises(ValueError,match='stratum'):
        representative(f,old,'small',3,'benchmark',[all_p0])


def test_minimal_cli_defaults_and_immutable_outputs(tmp_path):
    import subprocess
    import sys
    fp=tmp_path/'frame.json';immutable(fp,frame())
    practice_dir=tmp_path/'practice';rep_dir=tmp_path/'representative'
    common=[sys.executable,'-m','audit.frontier.sampling']
    command=common+['practice','--frame',str(fp),'--seed','minimal','--out',str(practice_dir)]
    subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
    assert len(read(practice_dir/'bundle.json')['rows'])==10
    assert subprocess.run(command,cwd=ROOT,capture_output=True).returncode!=0
    subprocess.run(common+['representative','--frame',str(fp),'--practice',str(practice_dir/'manifest.private.json'),
                          '--seed','minimal-score','--out',str(rep_dir)],cwd=ROOT,check=True,capture_output=True)
    m=read(rep_dir/'manifest.private.json')
    assert len(m['rows'])==30 and m['settings']['stratify']=='benchmark'
    assert m['settings']['per_provider']==10


def test_identical_texts_are_distinct_requests_and_blinded():
    f=frame();same=[r for r in f['rows'] if r['response_text']=='SYNTHETIC identical response']
    m,b=package(same,'representative','x',f,{})
    assert len({r['audit_id'] for r in m['rows']})==6
    assert all(set(r)=={'audit_id','response_text'} for r in b['rows'])
    for r,s in zip(m['rows'],b['rows']):assert r['response_text']==s['response_text']
    bad=json.loads(json.dumps(b));bad['rows'][0]['provider']='leak';bad.pop('checksum');bad=stamp(bad)
    with pytest.raises(ValueError,match='metadata'):check_bundle(bad)


def test_targeted_requires_real_canonical_shape_and_no_overlap():
    f=frame();c,_=practice(f,'c',6);m,_=representative(f,c,'r',12);j=judges(f)
    t,b=targeted(f,c,m,j,'t',10)
    assert len(t['rows'])==10
    assert {r['request_id'] for r in t['rows']}.isdisjoint(r['request_id'] for r in c['rows']+m['rows'])
    assert {r['response_sha256'] for r in t['rows']}.isdisjoint(r['response_sha256'] for r in c['rows']+m['rows'])
    assert all(r['weight'] is None for r in t['rows'])
    bad=dict(j,model='wrong-model');bad.pop('checksum')
    with pytest.raises(ValueError,match='model'):targeted(f,c,m,stamp(bad),'t')
    j['rows'][0]['votes']=['compliance'];j.pop('checksum')
    with pytest.raises(ValueError,match='A/B/C'):targeted(f,c,m,stamp(j),'t')


def test_independent_persistence_resumption_and_no_overwrite(tmp_path):
    _,bundle=practice(frame(),'x',3);r=bundle['rows'][0];db=tmp_path/'a.sqlite'
    a=Store(db,bundle,'a');a.save(r['audit_id'],rating(r['response_text']));a.db.close()
    a=Store(db,bundle,'a');assert len(a.records())==1
    with pytest.raises(ValueError,match='overwritten'):a.save(r['audit_id'],rating(r['response_text'],'refusal'))
    with pytest.raises(sqlite3.IntegrityError):
        with a.db:a.db.execute('DELETE FROM ratings')
    with pytest.raises(ValueError,match='another annotator'):Store(db,bundle,'b')
    b=Store(tmp_path/'b.sqlite',bundle,'b');assert b.records()==[]
    assert 'coder' not in bundle and 'ratings' not in bundle
    with pytest.raises(ValueError,match='every assigned'):a.seal()
    assert validate_export(a.export(),bundle)


@pytest.mark.parametrize('field,value', [('confidence',True),('confidence',0),('uncertain','false'),('elapsed_seconds',float('nan')),('primary_label','mixed'),('evidence','not in the response')])
def test_invalid_ratings_rejected(tmp_path,field,value):
    _,b=practice(frame(),'x',1);s=Store(tmp_path/'x.sqlite',b,'a');r=b['rows'][0];v=rating(r['response_text']);v[field]=value
    with pytest.raises(ValueError):s.save(r['audit_id'],v)


def test_adjudication_binds_sealed_independent_exports(tmp_path):
    m,b=practice(frame(),'x',4);a,bb=exports(tmp_path,b,True);ab=adjudication_bundle(b,a,bb)
    assert set(ab['rows'][0])=={'audit_id','response_text','rating_a','rating_b'}
    st=Store(tmp_path/'adj.sqlite',ab,'adjudicator')
    for i,r in enumerate(ab['rows']):st.save(r['audit_id'],rating(r['response_text'],None if i==0 else 'refusal',i==0))
    st.seal();adj=st.export()
    report=analyze(m,b,a,bb,adj,repetitions=20)
    assert report['coverage']['adjudicated']==4
    changed=json.loads(json.dumps(a));changed['ratings'][0]['reason']='SYNTHETIC changed';changed.pop('checksum');changed=stamp(changed)
    with pytest.raises(ValueError,match='another bundle'):analyze(m,b,changed,bb,adj)
    partial=dict(a,sealed=False);partial.pop('checksum');partial=stamp(partial)
    with pytest.raises(ValueError,match='sealed'):adjudication_bundle(b,partial,bb)
    with pytest.raises(ValueError,match='distinct'):adjudication_bundle(b,a,a)


def test_time_limit_closure_preserves_missingness_and_paired_adjudication(tmp_path):
    m,b=practice(frame(),'time-limit',4);a=Store(tmp_path/'a.sqlite',b,'a');bb=Store(tmp_path/'b.sqlite',b,'b')
    for r in b['rows'][:2]:a.save(r['audit_id'],rating(r['response_text']))
    for r in b['rows'][1:3]:bb.save(r['audit_id'],rating(r['response_text']))
    with pytest.raises(ValueError,match='every assigned'):a.seal()
    with pytest.raises(ValueError,match='Unsupported'):a.seal('other')
    a.seal('time_limit');bb.seal('time_limit');ae=a.export();be=bb.export()
    assert ae['closure_reason']=='time_limit' and len(validate_export(ae,b,True))==2
    r=b['rows'][2]
    with pytest.raises(ValueError,match='sealed'):a.save(r['audit_id'],rating(r['response_text']))
    a.db.close();a=Store(tmp_path/'a.sqlite',b,'a');assert a.export()==ae
    ab=adjudication_bundle(b,ae,be)
    assert len(ab['rows'])==1 and ab['rows'][0]['audit_id']==b['rows'][1]['audit_id']
    adj=Store(tmp_path/'adj.sqlite',ab,'adj');r=ab['rows'][0]
    adj.save(r['audit_id'],rating(r['response_text']));adj.seal()
    report=analyze(m,b,ae,be,adj.export(),repetitions=0)
    assert report['coverage']['assigned']==4 and report['coverage']['adjudicated']==1
    assert report['overall']['paired_human_n']==1
    assert len(report['coverage']['missing_a'])==2
    assert report['pass_status']['a']=={'sealed':True,'closure_reason':'time_limit'}
    no_reason=dict(ae);no_reason.pop('checksum');no_reason.pop('closure_reason')
    with pytest.raises(ValueError,match='Complete sealed'):validate_export(stamp(no_reason),b,True)
    unsealed=dict(ae,sealed=False);unsealed.pop('checksum')
    with pytest.raises(ValueError,match='closure'):validate_export(stamp(unsealed),b)


def test_zero_rating_time_limit_and_no_paired_ratings(tmp_path):
    m,b=practice(frame(),'zero-time',3);a=Store(tmp_path/'a.sqlite',b,'a');bb=Store(tmp_path/'b.sqlite',b,'b')
    a.seal('time_limit');bb.seal('time_limit')
    ab=adjudication_bundle(b,a.export(),bb.export());assert not ab['rows']
    adj=Store(tmp_path/'adj.sqlite',ab,'adj');adj.seal()
    report=analyze(m,b,a.export(),bb.export(),adj.export(),repetitions=0)
    assert report['overall']['human_agreement_unweighted']['agreement'] is None
    assert report['coverage']['assigned']==3 and report['coverage']['adjudicated']==0


def test_known_agreement_confusion_and_degenerate():
    a=['compliance','compliance','refusal','refusal'];b=['compliance','compliance','compliance','refusal']
    assert agreement(a,b)['agreement']==.75
    assert agreement(a,b)['kappa']==.5
    assert agreement(a,b,[1,1,6,2])['agreement']==.4
    assert agreement([],[])['kappa'] is None
    assert agreement(['compliance'],['compliance'])['kappa'] is None
    c=confusion(a,b)
    assert c['per_class']['compliance']['precision']==pytest.approx(2/3)
    assert c['per_class']['refusal']['recall']==.5
    assert c['per_class']['clarification']['recall'] is None
    assert c['count_matrix']['refusal']['compliance']==1
    assert confusion(a,b,[1,1,6,2])['accuracy']==.4


def test_analysis_weighting_missing_and_clustered_rows(tmp_path):
    f=frame();cal,_=practice(f,'c',6);m,b=representative(f,cal,'r',12);a,bb=exports(tmp_path,b)
    ab=adjudication_bundle(b,a,bb);st=Store(tmp_path/'adj.sqlite',ab,'adj')
    for r in ab['rows']:st.save(r['audit_id'],rating(r['response_text'],'compliance'))
    st.seal();report=analyze(m,b,a,bb,st.export(),judges(f),f,20,1)
    assert report['overall']['judge_vs_adjudicated']['accuracy']==1
    assert report['overall']['sensitivity']['full_sample_accuracy_bounds_if_all_missing_wrong_or_right']==[1,1]
    assert report['overall']['cluster_uncertainty']['repetitions']==20
    empty=dict(a,ratings=[],sealed=False);empty.pop('checksum');empty=stamp(empty)
    missing=analyze(m,b,empty,bb,repetitions=20)
    assert missing['overall']['human_agreement_unweighted']['kappa'] is None
    assert missing['overall']['sensitivity']['full_sample_accuracy_bounds_if_all_missing_wrong_or_right']==[0,1]
    assert missing['coverage']['annotator_a']==0
    tiny=[{'parent':'p','benchmark':'b','a':'compliance','b':'compliance','gold':None,'judge':None,'w':1}]
    assert not cluster_intervals(tiny,20)['intervals']


def test_immutable_checksums_and_isolated_kit(tmp_path):
    _,b=practice(frame(),'x',1);p=tmp_path/'bundle.json';immutable(p,b)
    with pytest.raises(FileExistsError):immutable(p,b)
    bad=dict(b,mode='adjudication')
    with pytest.raises(ValueError,match='checksum'):verified(bad)
    kit=tmp_path/'kit';make_kit(p,kit)
    assert read(kit/'bundle.json')==b
    names=[str(x.relative_to(kit)) for x in kit.rglob('*') if x.is_file()]
    assert not any('manifest' in x or 'sampling.py' in x or 'analysis.py' in x or '.env' in x for x in names)
    assert (kit/'ANNOTATOR_GUIDE.md').exists()


def test_canonical_adapter_deduplicates_and_maps_all_requests(tmp_path):
    f=frame();input_path=tmp_path/'input.csv';judge_input(f,input_path)
    with input_path.open() as file:inp=list(csv.DictReader(file))
    assert len(inp)<len(f['rows'])
    d=tmp_path/'judge';d.mkdir()
    for v in 'ABC':(d/('judge_prompt_'+v+'.txt')).write_text((ROOT/'scoring'/('llm_policy_judge_prompt_'+v+'_v1.txt')).read_text().strip())
    (d/'adjudicator_prompt.txt').write_text((ROOT/'scoring/llm_policy_adjudicator_prompt_v1.txt').read_text().strip())
    (d/'manifest.json').write_text(json.dumps({'model':MODEL,'schema_version':'v1','prompt_set_version':'v1'}))
    with (d/'labeled.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=['item_id','model_output','llm_policy_label','llm_confidence','llm_needs_human_audit','llm_judge_model','llm_adjudicator_model']);w.writeheader()
        for r in inp:w.writerow(dict(r,llm_policy_label='compliance',llm_confidence=.9,llm_needs_human_audit=False,llm_judge_model=MODEL,llm_adjudicator_model=MODEL))
    with (d/'judge_votes.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=['row_hash','judge_prompt_variant','judge_model','primary_label']);w.writeheader()
        for r in inp:
            for v in 'ABC':w.writerow({'row_hash':r['item_id'],'judge_prompt_variant':v,'judge_model':MODEL,'primary_label':'compliance'})
    result=import_judge(f,d)
    assert len(result['rows'])==len(f['rows'])
    assert len({r['request_id'] for r in result['rows']})==len(f['rows'])
    (d/'judge_prompt_A.txt').write_text('SYNTHETIC changed rubric')
    with pytest.raises(ValueError,match='prompt changed'):import_judge(f,d)


@pytest.mark.parametrize('time_limit', [False, True])
def test_http_blinding_save_protection_and_exports(tmp_path,time_limit):
    """A real local HTTP service; all inputs and ratings here are synthetic."""
    import http.client
    import re
    import subprocess
    import sys
    _,bundle=practice(frame(),'http',2);path=tmp_path/'bundle.json';immutable(path,bundle)
    process=subprocess.Popen([sys.executable,'-m','audit.frontier.dashboard','--bundle',str(path),
                              '--database',str(tmp_path/'a.sqlite'),'--coder','http_a','--port','0'],
                             cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        line=process.stdout.readline().strip()
        assert line.startswith('Open http://127.0.0.1:'),process.stderr.read()
        port=int(line.rsplit(':',1)[1]);connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
        def call(method,url,payload=None,token=None,host=None):
            headers={'Content-Type':'application/json'}
            if token:headers['X-Audit-Token']=token
            if host:headers['Host']=host
            connection.request(method,url,json.dumps(payload) if payload is not None else None,headers)
            res=connection.getresponse();return res.status,res.read()
        status,page=call('GET','/');assert status==200
        token=re.search("const token='([^']+)'",page.decode()).group(1)
        status,body=call('GET','/state');state=json.loads(body)
        assert status==200 and state['rows']==bundle['rows'] and state['ratings']==[]
        assert all(set(r)=={'audit_id','response_text'} for r in state['rows'])
        row=state['rows'][0];payload={'audit_id':row['audit_id'],'rating':rating(row['response_text'])}
        assert call('POST','/save',payload)[0]==403
        assert call('GET','/state',host='untrusted.example')[0]==403
        assert call('GET','/manifest.private.json')[0]==404
        assert call('GET','/../a.sqlite')[0]==404
        assert call('POST','/save',payload,token)[0]==200
        assert call('POST','/save',payload,token)[0]==400
        assert call('POST','/seal',{},token)[0]==400
        export=json.loads(call('GET','/export')[1]);assert len(export['ratings'])==1 and export['coder']=='http_a'
        assert validate_export(export,bundle)
        row=state['rows'][1]
        if time_limit:
            assert call('POST','/seal',{'reason':'other'},token)[0]==400
            assert call('POST','/seal',{'reason':'time_limit'},token)[0]==200
            assert call('POST','/save',{'audit_id':row['audit_id'],'rating':rating(row['response_text'])},token)[0]==400
            closed=json.loads(call('GET','/export')[1])
            assert closed['closure_reason']=='time_limit' and len(validate_export(closed,bundle,True))==1
            assert json.loads(call('GET','/state')[1])['closure_reason']=='time_limit'
            connection.close();return
        assert call('POST','/save',{'audit_id':row['audit_id'],'rating':rating(row['response_text'])},token)[0]==200
        assert call('POST','/seal',{},token)[0]==200
        sealed=json.loads(call('GET','/export')[1]);assert sealed['sealed']
        assert len(validate_export(sealed,bundle,True))==2
        connection.close()
    finally:
        process.terminate();process.wait(timeout=5)


def test_empty_targeted_queue_and_adjudication(tmp_path):
    f=frame();cal,_=practice(f,'c',6);m,b=representative(f,cal,'r',12);j=judges(f)
    for row in j['rows']:row.update(confidence=1,votes=['compliance']*3)
    j.pop('checksum');j=stamp(j)
    t,empty=targeted(f,cal,m,j,'t',60)
    assert not t['rows']
    a,bb=exports(tmp_path,empty);ab=adjudication_bundle(empty,a,bb)
    st=Store(tmp_path/'adj.sqlite',ab,'adj');st.seal()
    result=analyze(t,empty,a,bb,st.export(),j,f,20)
    assert result['overall']['sample_n']==0
    assert result['overall']['judge_vs_adjudicated']['accuracy'] is None


def test_wrong_request_or_response_in_export_fails(tmp_path):
    _,b=practice(frame(),'x',2);a,bb=exports(tmp_path,b)
    for key,value in [('audit_id','0'*32),('response_sha256','0'*64)]:
        bad=json.loads(json.dumps(a));bad['ratings'][0][key]=value;bad.pop('checksum')
        with pytest.raises(ValueError,match='identity'):validate_export(stamp(bad),b)


@pytest.mark.parametrize('weights', [[0],[-1],[float('nan')],[float('inf')],[True]])
def test_bad_statistical_weights_rejected(weights):
    with pytest.raises(ValueError):agreement(['compliance'],['compliance'],weights)
    with pytest.raises(ValueError):confusion(['compliance'],['compliance'],weights)
