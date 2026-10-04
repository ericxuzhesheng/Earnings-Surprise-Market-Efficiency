"""Resumable all-A-share expansion; licensed raw data stays under data_raw/."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data_raw/expansion_20260930'
OUT = ROOT / 'outputs/expanded_20260930'
START, END = '20200101', '20260930'
PRICE_START = '20190101'  # Pre-event history for liquidity, listing age and leakage windows.
LOCK = threading.Lock()
NEXT = {}
BLOCKED = {}


def query(endpoint, **params):
    # Global endpoint pacing, shared by all workers. Never rotate credentials.
    interval = 361.0 if endpoint == 'report_rc' else .5
    with LOCK:
        if endpoint in BLOCKED:
            raise RuntimeError(f'{endpoint} blocked for this run: {BLOCKED[endpoint]}')
        delay = max(0., NEXT.get(endpoint, 0.) - time.monotonic())
        NEXT[endpoint] = time.monotonic() + delay + interval
    if delay:
        time.sleep(delay)
    for attempt in range(3):
        try:
            if endpoint in BLOCKED:
                raise RuntimeError(f'{endpoint} blocked for this run: {BLOCKED[endpoint]}')
            return ts.pro_api(os.environ['TUSHARE_TOKEN'], timeout=35).query(endpoint, **params)
        except Exception as exc:
            if any(s in str(exc) for s in ['频次','频率','权限','积分','每小时','每天']):
                BLOCKED[endpoint] = str(exc)
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch(endpoint, path, params, limit=3000, force_empty=False):
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        meta = json.loads(receipt.read_text(encoding='utf-8'))
        if meta['endpoint'] != endpoint or meta['params'] != params:
            raise ValueError(f'Cache request mismatch: {path}')
        return pd.read_csv(path)
    parts, offset, previous = [], 0, None
    while True:
        page = query(endpoint, **params, limit=limit, offset=offset)
        if page.empty:
            break
        required = set(params.get('fields','').split(',')) - {''}
        if not required.issubset(page.columns):
            raise ValueError(f'Missing requested fields for {endpoint}: {required-set(page.columns)}')
        if 'ann_date' in params and not pd.to_numeric(page.ann_date).eq(int(params['ann_date'])).all():
            raise ValueError(f'Announcement-date query returned another date: {params}')
        if previous is not None and previous.equals(page):
            raise ValueError(f'Repeated page: {endpoint} {params}')
        parts.append(page)
        previous = page
        offset += len(page)
        if len(page) < limit and not force_empty:
            break
    frame = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=params.get('fields','ts_code,ann_date,end_date').split(','))
    temporary = path.with_suffix('.partial.csv')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    receipt.write_text(json.dumps({'endpoint':endpoint,'params':params,'limit':limit,
        'rows':len(frame),'pages':len(parts),'retrieved_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'pagination':'until empty' if force_empty else 'until below requested page limit'},indent=2),encoding='utf-8')
    return frame


def refresh_market():
    fetch('index_daily', DATA/'market/market_with_warmup.csv',{'ts_code':'399300.SZ','start_date':PRICE_START,'end_date':END},limit=6000)
    fetch('trade_cal', DATA/'market/calendar_with_warmup.csv',{'exchange':'SSE','start_date':PRICE_START,'end_date':END},limit=6000)


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    universe = []
    for state, oldname in [('L','listed'),('D','delisted'),('P','paused')]:
        existing = DATA / f'{oldname}.csv'
        if existing.exists():
            frame = pd.read_csv(existing)
        else:
            frame = fetch('stock_basic', existing, {'list_status':state,
                'fields':'ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date'},limit=6000)
        universe.append(frame)
    universe = pd.concat(universe,ignore_index=True).drop_duplicates('ts_code')
    # CNY removes B-shares; preserve all listing states, including delisted names.
    universe = universe.loc[universe.curr_type.eq('CNY')].copy()
    universe = universe.loc[pd.to_numeric(universe.list_date,errors='coerce') <= int(END)]
    folder = DATA/'stock_basic'; folder.mkdir(exist_ok=True)
    universe.to_csv(folder/'listed.csv',index=False)
    print(f'Universe: {len(universe)} CNY listed/delisted/suspended stocks',flush=True)
    refresh_market()
    tasks = []
    # The ordinary endpoints support ann_date. This account has no VIP access;
    # use the documented daily-announcement query, including weekend releases.
    for day in pd.date_range(START, END, freq='D')[::-1]:
        date_text = day.strftime('%Y%m%d')
        for endpoint, fields in [
            ('forecast','ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net,first_ann_date,summary,change_reason,update_flag'),
            ('express','ts_code,ann_date,end_date,n_income')]:
            tasks.append((endpoint,DATA/f'announcement_{endpoint}'/f'{date_text}.csv',{'ann_date':date_text,'fields':fields}))
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch,*task):task for task in tasks}
        for i, future in enumerate(as_completed(futures),1):
            task = futures[future]
            try:
                frame = future.result()
                if i % 200 == 0:
                    print(f'Financial partitions {i}/{len(tasks)}; {task[0]} {task[1].stem}: {len(frame)} rows',flush=True)
            except Exception as exc:
                failures.append({'endpoint':task[0],'period':task[1].stem,'error':str(exc)})
    report_progress_path = DATA/'report_expansion_progress.json'
    report_progress = json.loads(report_progress_path.read_text(encoding='utf-8')) if report_progress_path.exists() else {}
    report_errors = report_progress.get('errors', [])
    manifest = {'start':START,'end':END,'price_start':PRICE_START,'universe_rows':len(universe),
        'universe':'All available CNY listed, delisted and suspended stocks; historic identifiers may remain incomplete',
        'forecast_partitions':f'Every calendar announcement date {START} through {END}, including weekends',
        'analyst_reports':'Existing monthly caches through 2026-04; incomplete/capped. Additional completed partitions only.',
        'report_api_limit_observed':report_errors[-1]['error'] if report_errors else 'Not probed by prepare; reports is a separate collection stage',
        'partition_layout':{f'{endpoint}_vip':f'Aggregated ordinary {endpoint}(ann_date=...) snapshots, not VIP access' for endpoint in ['forecast','express']},
        'market_file':'market/market_with_warmup.csv','calendar_file':'market/calendar_with_warmup.csv',
        'failures':failures,'point_in_time_vintage':False}
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} financial partitions failed; see manifest.json')
    for endpoint in ['forecast','express']:
        frames = []
        for path in sorted((DATA/f'announcement_{endpoint}').glob('*.csv')):
            frame = pd.read_csv(path)
            if not frame.empty and not pd.to_numeric(frame.ann_date).eq(int(path.stem)).all():
                raise ValueError(f'Wrong announcement date in {path}')
            frames.append(frame)
        combined = pd.concat([f for f in frames if not f.empty], ignore_index=True).drop_duplicates()
        if combined.end_date.isna().any():
            raise ValueError(f'Missing fiscal period in {endpoint}')
        folder = DATA / f'{endpoint}_vip'  # Existing analysis partition layout, source is ordinary endpoint.
        folder.mkdir(exist_ok=True)
        for period, group in combined.groupby('end_date'):
            group.to_csv(folder/f'{int(period)}.csv',index=False)
        print(f'{endpoint}: {len(combined):,} unique source rows',flush=True)
    print('Announcement-date partitions complete; ready for expanded matching.',flush=True)


def prices():
    audit = pd.read_csv(OUT/'matching_audit.csv',usecols=['ts_code','surprise_pct'])
    codes = sorted(audit.loc[audit.surprise_pct.notna(),'ts_code'].unique())
    print(f'Matched stocks: {len(codes)}; fetching prices, turnover, factors and income statements',flush=True)
    specs = [('daily','ts_code,trade_date,open,high,low,close,vol,amount'),
             ('daily_basic','ts_code,trade_date,turnover_rate'),
             ('adj_factor','ts_code,trade_date,adj_factor'),
             ('income','ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_income_attr_p,n_income,basic_eps,revenue,total_revenue')]
    tasks = [(endpoint,DATA/f'source_{endpoint}'/f'{code}.csv',
              {'ts_code':code,'start_date':PRICE_START,'end_date':END,'fields':fields})
             for code in codes for endpoint,fields in specs]
    failures = []
    totals = {name:0 for name,_ in specs}
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(fetch,*task,limit=100 if task[0]=='income' else 6000):task for task in tasks}
        for i,future in enumerate(as_completed(futures),1):
            task = futures[future]
            try:
                frame = future.result(); totals[task[0]] += len(frame)
                if i % 100 == 0:
                    print(f'Market snapshots {i}/{len(tasks)}; failures={len(failures)}',flush=True)
            except Exception as exc:
                failures.append({'endpoint':task[0],'code':task[1].stem,'error':str(exc)})
    receipt = {'codes':codes,'start':PRICE_START,'end':END,'row_counts':totals,'failures':failures,
               'retrieved_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'complete':not failures}
    (DATA/'market_sources_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stocks':len(codes),'rows':totals,'failures':len(failures)}),flush=True)
    if failures:
        raise RuntimeError('Some market snapshots failed; resume rather than silently dropping their events')
    financial = pd.concat([pd.read_csv(DATA/'source_income'/f'{code}.csv') for code in codes],ignore_index=True)
    financial.to_parquet(DATA/'financial_announcements.parquet',index=False)


def reports(months=None):
    """Use at most eight requests, >=61 seconds apart; persist every report page.

    Prioritize December snapshots before the main annual preannouncement season.
    Stop immediately on access/rate errors; a partial month is never complete.
    """
    progress_path = DATA/'report_expansion_progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {'requests':[], 'months':{}, 'errors':[]}
    def save():
        progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    service_day=str(pd.Timestamp.now(tz='Asia/Shanghai').date())
    for error in progress['errors']:
        error_day=str(pd.Timestamp(error['at_utc']).tz_convert('Asia/Shanghai').date())
        if error_day==service_day and any(word in error['error'] for word in ['次/天','每天']):
            progress['blocked_service_day']=service_day;save()
            print('Report daily quota already exhausted; no additional request sent.',flush=True)
            return
    used = 0
    priority=[f'{year}-12' for year in range(2025,2018,-1)]
    remaining=[period.strftime('%Y-%m') for period in pd.period_range('2019-01','2026-09',freq='M')[::-1]
               if period.strftime('%Y-%m') not in priority]
    for month in (months or priority+remaining):
        period = pd.Period(month,freq='M')
        key = period.strftime('%Y%m')
        state = progress['months'].setdefault(key,{'offset':0,'complete':False,'rows':0})
        if state['complete']:
            continue
        page_dir = DATA/'report_rc_pages'/key
        page_dir.mkdir(parents=True,exist_ok=True)
        saved_pages=sorted(page_dir.glob('*.csv'))
        # Recover a page saved immediately before an interrupted progress write.
        state['offset']=sum(len(pd.read_csv(p)) for p in saved_pages)
        state['rows']=state['offset'];save()
        while used < 8:
            now = time.time()
            recent = [t for t in progress['requests'] if now-t < 3600]
            if len(recent) >= 8:
                save(); print('Report request budget retained for this rolling hour.',flush=True); return
            if progress['requests']:
                time.sleep(max(0.,61-(now-progress['requests'][-1])))
            progress['requests'].append(time.time()); used += 1; save()
            params={'start_date':period.start_time.strftime('%Y%m%d'),
                    'end_date':period.end_time.strftime('%Y%m%d'),'limit':3000,'offset':state['offset']}
            try:
                page=ts.pro_api(os.environ['TUSHARE_TOKEN'],timeout=35).report_rc(**params)
            except Exception as exc:
                progress['errors'].append({'at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'params':params,'error':str(exc)})
                save(); print(f'Report collection stopped: {exc}',flush=True); return
            if page.empty:
                pieces=[pd.read_csv(p) for p in sorted(page_dir.glob('*.csv'))]
                full=pd.concat(pieces,ignore_index=True).drop_duplicates() if pieces else pd.DataFrame(columns=['ts_code'])
                complete_dir=DATA/'report_rc_complete';complete_dir.mkdir(exist_ok=True)
                full.to_csv(complete_dir/f'{key}.csv',index=False)
                state['complete']=True;state['rows']=len(full);save()
                print(f'Reports {key} COMPLETE: {len(full)} rows',flush=True)
                break
            dates=pd.to_datetime(page.report_date.astype(str))
            if not dates.between(period.start_time,period.end_time).all():
                raise ValueError('Report query returned dates outside its requested month')
            destination=page_dir/f'{state["offset"]:06d}.csv'
            if destination.exists():
                raise ValueError('Report progress would overwrite an existing page')
            previous_files=sorted(page_dir.glob('*.csv'))
            if previous_files:
                previous=pd.read_csv(previous_files[-1],dtype=str).fillna('')
                keys=['ts_code','report_date','quarter','org_name','report_title']
                current=page[keys].fillna('').astype(str).reset_index(drop=True)
                if previous[keys].reset_index(drop=True).equals(current):
                    raise ValueError('Repeated report page')
            page.to_csv(destination,index=False)
            state['offset']+=len(page);state['rows']+=len(page);save()
            print(f'Reports {key}: {state["rows"]} saved rows, completeness pending',flush=True)
        if used>=8:
            break
    save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','prices','reports'])
    args=parser.parse_args()
    {'prepare':prepare,'prices':prices,'reports':reports}[args.stage]()
