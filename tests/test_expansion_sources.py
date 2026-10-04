import importlib.util
from pathlib import Path
import pandas as pd
import pytest
from types import SimpleNamespace
import json

spec = importlib.util.spec_from_file_location('expansion', Path(__file__).parents[1]/'scripts/expand_event_data.py')
expansion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)


def test_paginated_snapshot_and_resume_require_matching_request(tmp_path, monkeypatch):
    offsets = []
    def query(endpoint, **params):
        offsets.append(params['offset'])
        return pd.DataFrame({'ts_code':['A','B'], 'ann_date':[20260130]*2}) if params['offset']==0 else pd.DataFrame()
    monkeypatch.setattr(expansion,'query',query)
    path = tmp_path/'20260130.csv'
    params = {'ann_date':'20260130','fields':'ts_code,ann_date'}
    assert len(expansion.fetch('forecast',path,params,limit=2))==2
    assert offsets==[0,2]
    assert len(expansion.fetch('forecast',path,params,limit=2))==2
    assert offsets==[0,2]
    with pytest.raises(ValueError,match='mismatch'):
        expansion.fetch('forecast',path,dict(params,ann_date='20260131'))


def test_bad_date_or_schema_cannot_be_certified_complete(tmp_path, monkeypatch):
    path = tmp_path/'bad.csv'
    monkeypatch.setattr(expansion,'query',lambda *a,**kw:pd.DataFrame({'ts_code':['A'],'ann_date':[20260129]}))
    with pytest.raises(ValueError,match='another date'):
        expansion.fetch('forecast',path,{'ann_date':'20260130','fields':'ts_code,ann_date'})
    assert not path.exists() and not path.with_suffix('.json').exists()
    with pytest.raises(ValueError,match='Missing requested fields'):
        expansion.fetch('daily_basic',path,{'fields':'ts_code,turnover_rate'})


def test_failed_page_is_not_a_complete_snapshot(tmp_path, monkeypatch):
    def query(endpoint,**params):
        if params['offset']:
            raise RuntimeError('network unavailable')
        return pd.DataFrame({'ts_code':['A','B']})
    monkeypatch.setattr(expansion,'query',query)
    path=tmp_path/'partial.csv'
    with pytest.raises(RuntimeError):
        expansion.fetch('forecast',path,{'fields':'ts_code'},limit=2)
    assert not path.exists() and not path.with_suffix('.json').exists()


def test_report_pages_survive_rate_limit_and_are_not_labeled_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion,'DATA',tmp_path)
    monkeypatch.setenv('TUSHARE_TOKEN','test-only')
    monkeypatch.setattr(expansion.time,'sleep',lambda _:None)
    def report(**params):
        if params['offset']:
            raise RuntimeError('频率超限(10次/小时)')
        return pd.DataFrame({'ts_code':['A'],'report_date':['20251210'],
            'quarter':['2025Q4'],'org_name':['B'],'report_title':['R']})
    monkeypatch.setattr(expansion.ts,'pro_api',lambda *a,**kw:SimpleNamespace(report_rc=report))
    expansion.reports(['2025-12'])
    progress=json.loads((tmp_path/'report_expansion_progress.json').read_text(encoding='utf-8'))
    assert progress['months']['202512']['offset']==1
    assert not progress['months']['202512']['complete']
    assert not (tmp_path/'report_rc_complete/202512.csv').exists()
    assert (tmp_path/'report_rc_pages/202512/000000.csv').exists()


def test_daily_quota_error_prevents_another_same_day_request(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion,'DATA',tmp_path)
    monkeypatch.setenv('TUSHARE_TOKEN','test-only')
    calls=[]
    def report(**params):
        calls.append(params)
        raise RuntimeError('频率超限(10次/天)')
    monkeypatch.setattr(expansion.ts,'pro_api',lambda *a,**kw:SimpleNamespace(report_rc=report))
    expansion.reports(['2025-12'])
    expansion.reports(['2025-12'])
    assert len(calls)==1
