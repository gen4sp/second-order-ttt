"""Strict downloaded FRS-B interchange for the fixed Family-A tests."""
from . import frs_cohorts as cohorts


def validate_family_inputs(value, *, smoke=False):
    if not isinstance(value, dict) or type(value.get('schema',1)) is not int or value.get('schema',1) != 1:
        raise ValueError('FRS-B per-seed evidence is required (--frsb)')
    mode=value.get('smoke_only',value.get('smoke',value.get('manifest',{}).get('smoke',smoke)))
    if type(mode) is not bool or mode != smoke:
        raise ValueError('FRS-B explicit mode mismatch')
    family = value.get('family_inputs', {})
    for name in ('R-LIN', 'R-ADDR'):
        source = family.get(name)
        if not isinstance(source, dict) or any(not isinstance(source.get(k),list) for k in ('seeds','test','comparator')):
            raise ValueError(f'{name}: parallel seeds/test/comparator arrays required')
        seeds=source['seeds']
        for key in ('test','comparator','diverged'):
            if key not in source:continue
            if not isinstance(source[key],list) or len(source[key])!=len(seeds) or any(type(v) is not bool for v in source[key]):
                raise ValueError(f'{name}: invalid {key} vector')
        if any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds) or seeds != sorted(seeds):
            raise ValueError(f'{name}: duplicates or unsorted seeds')
        if not smoke and not set(seeds)<=set(range(200, 212)):
            raise ValueError(f'{name}: production seeds must be in 200–211')
        if smoke and any(not 0 <= s <= 9 for s in seeds):
            raise ValueError(f'{name}: smoke requires legacy seeds')
        missing=source.get('missing',[])
        if not isinstance(missing,list) or any(type(s) is not int for s in missing) or set(missing)&set(seeds) or \
                len(set(missing))!=len(missing) or missing!=sorted(missing):
            raise ValueError(f'{name}: malformed missing seeds')
        if not smoke and 'missing' in source and set(missing)!=set(range(200,212))-set(seeds):
            raise ValueError(f'{name}: missing must be the registered complement')
    return value


def family_pair(source, *, smoke=False):
    units = [dict(seed=s,test=t,comparator=c,diverged=d) for s,t,c,d in zip(source['seeds'],
        source['test'],source['comparator'],source.get('diverged',[False]*len(source['seeds'])))]
    out = cohorts.exact_mcnemar([u['test'] for u in units], [u['comparator'] for u in units])
    clean = [u for u in units if not u['diverged']]
    out.update(units=units, missing=source.get('missing',[] if smoke else sorted(set(range(200,212))-set(source['seeds']))))
    if 'diverged' in source:
        out['without_divergence']=cohorts.exact_mcnemar([u['test'] for u in clean], [u['comparator'] for u in clean])
    return out
