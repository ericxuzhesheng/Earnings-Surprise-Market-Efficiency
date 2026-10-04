"""Verify downloaded partitions and independently reconstruct saved event returns."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_raw/expansion_20260930'
OUT=ROOT/'outputs/expanded_20260930'


def verify(sources_only=False):
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert not manifest['failures']
    expected=pd.date_range(manifest['start'],manifest['end'],freq='D')
    source_counts={}
    for endpoint in ['forecast','express']:
        rows=0
        for date in expected:
            path=DATA/f'announcement_{endpoint}'/f'{date:%Y%m%d}.csv'
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            frame=pd.read_csv(path)
            assert meta['endpoint']==endpoint and meta['params']['ann_date']==f'{date:%Y%m%d}'
            assert meta['rows']==len(frame)
            if not frame.empty:
                assert pd.to_numeric(frame.ann_date).eq(int(date.strftime('%Y%m%d'))).all()
                assert frame.end_date.notna().all()
            rows+=len(frame)
        source_counts[endpoint]={'calendar_partitions':len(expected),'rows':rows}
    market=pd.read_csv(DATA/'market/market_with_warmup.csv')
    market.index=pd.to_datetime(market.trade_date.astype(str))
    market=market.sort_index()
    assert not market.index.duplicated().any()
    calendar=pd.read_csv(DATA/'market/calendar_with_warmup.csv')
    dates=pd.to_datetime(calendar.loc[calendar.is_open.eq(1),'cal_date'].astype(str)).sort_values()
    assert np.array_equal(dates.to_numpy(),market.index.to_numpy())
    assert market.close.gt(0).all()
    receipt={'source_checks':source_counts,'market_sessions':len(market),'status':'sources passed'}
    if sources_only:
        (OUT/'source_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt,indent=2));return
    events=pd.read_csv(OUT/'events.csv',parse_dates=['ann_date','event_trade_date','latest_report_date'])
    assert (events.latest_report_date<events.ann_date).all()
    assert (events.ann_date<events.event_trade_date).all()
    assert not events.duplicated(['ts_code','period_end']).any()
    market_return=market.close/market.close.shift()-1
    checked=0
    missing_factors=0
    market_receipt=json.loads((DATA/'market_sources_receipt.json').read_text(encoding='utf-8'))
    assert market_receipt['complete'] and not market_receipt['failures']
    assert set(market_receipt['codes'])==set(events.ts_code)
    market_rows={endpoint:0 for endpoint in ['daily','daily_basic','adj_factor','income']}
    for code,group in events.groupby('ts_code'):
        snapshots={}
        for endpoint in market_rows:
            path=DATA/f'source_{endpoint}/{code}.csv'
            frame=pd.read_csv(path)
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert meta['endpoint']==endpoint and meta['params']['ts_code']==code
            assert meta['params']['start_date']==manifest['price_start']
            assert meta['params']['end_date']==manifest['end']
            assert set(meta['params']['fields'].split(',')).issubset(frame.columns)
            assert meta['rows']==len(frame) and frame.ts_code.eq(code).all()
            date_field='ann_date' if endpoint=='income' else 'trade_date'
            assert pd.to_numeric(frame[date_field]).between(int(manifest['price_start']),int(manifest['end'])).all()
            market_rows[endpoint]+=len(frame)
            snapshots[endpoint]=frame
        daily=snapshots['daily']
        factors=snapshots['adj_factor']
        daily.index=pd.to_datetime(daily.trade_date.astype(str))
        factors.index=pd.to_datetime(factors.trade_date.astype(str))
        assert not daily.index.duplicated().any() and not factors.index.duplicated().any()
        assert daily.close.gt(0).all()
        factor=factors.adj_factor.reindex(market.index)
        close=daily.close.reindex(market.index)
        missing_factors+=int((close.notna()&factor.isna()).sum())
        adjusted=close*factor.where(factor>0)
        abnormal=adjusted/adjusted.shift()-1-market_return
        for event in group.itertuples():
            pos=market.index.get_loc(event.event_trade_date)
            values=abnormal.iloc[pos+1:pos+11]
            wanted=values.sum() if len(values)==10 and values.notna().all() else np.nan
            np.testing.assert_allclose(event.CAR_1_10,wanted,rtol=0,atol=1e-12,equal_nan=True)
            checked+=1
    assert market_rows==market_receipt['row_counts']
    assert len(pd.read_parquet(DATA/'financial_announcements.parquet'))==market_rows['income']
    clean=events.loc[events.eligible&~events.contaminated_10].dropna(subset=['CAR_1_10','surprise_pct'])
    positive=clean.loc[clean.surprise_pct>0,'CAR_1_10']
    negative=clean.loc[clean.surprise_pct<0,'CAR_1_10']
    primary=pd.read_csv(OUT/'inference.csv').query('primary_test').iloc[0]
    assert primary.n==len(positive)+len(negative)
    np.testing.assert_allclose(primary.estimate,positive.mean()-negative.mean(),rtol=0,atol=1e-12)
    receipt.update(status='passed',events_checked=checked,headline_n=int(primary.n),
        market_snapshots_checked=len(market_receipt['codes'])*len(market_rows),market_rows=market_rows,
        matched_firms=int(events.ts_code.nunique()),missing_factor_on_traded_session=missing_factors,
        positive_minus_negative=float(primary.estimate))
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only',action='store_true')
    verify(**vars(parser.parse_args()))
