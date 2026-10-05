"""Check selected final/inner/outer fits without changing saved models or choices."""
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import no_neoadjuvant_core as core

OUT=core.OUT

def main():
    results=json.loads((OUT/'03_models/model_results.json').read_text(encoding='utf-8'))
    rows=[]
    with threadpool_limits(limits=1):
        data=core.load_sources()
        def checked_fit(name,train,test,penalty,scope):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                xt,xv,_,_=core.make_design(data,train,test,name)
                fit=core.fit_cox(xt,data.time[train],data.event[train],penalty)
            messages=[str(x.message) for x in caught]
            failed=any('failed to converge' in x.lower() for x in messages)
            assert np.isfinite(fit.params_.to_numpy()).all()
            rows.append({'model':name,'scope':scope,'penalty':penalty,'n_train':len(train),
                         'optimizer_failed_warning':failed,'warning_categories':[x.category.__name__ for x in caught],
                         'messages':messages})
        def checked_inner(name,indices,penalty,label):
            split=StratifiedKFold(n_splits=3,shuffle=True,random_state=core.SEED)
            for k,(a,b) in enumerate(split.split(indices,data.event[indices]),1):
                checked_fit(name,indices[a],indices[b],penalty,f'{label}/inner{k}')
        for name,item in results['models'].items():
            use=core.model_mask(data,name);tr=np.where(use&(data.center==core.CFG['development_center']))[0]
            te=np.where(use&np.isin(data.center,core.CFG['external_centers']))[0]
            checked_fit(name,tr,te,item['penalizer'],'final')
            checked_inner(name,tr,item['penalizer'],'selected_final_tuning')
            print(json.dumps({'audit_selected_model':name}),flush=True)
        for name in ['M10','M12']:
            indices=np.where(core.model_mask(data,name)&(data.center==core.CFG['development_center']))[0]
            outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=core.SEED)
            for k,(a,b) in enumerate(outer.split(indices,data.event[indices]),1):
                penalty=results['nested_cv'][name]['folds'][k-1]['penalizer']
                tr,te=indices[a],indices[b]
                checked_fit(name,tr,te,penalty,f'outer{k}')
                checked_inner(name,tr,penalty,f'outer{k}/selected_tuning')
    report={'selected_fits_checked':len(rows),'selected_optimizer_failure_count':sum(x['optimizer_failed_warning'] for x in rows),
            'rows':rows,'models_changed':False}
    (OUT/'04_validation/selected_fit_convergence_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    assert report['selected_optimizer_failure_count']==0,'Selected fits contain optimizer failures; do not finalize.'

if __name__=='__main__':main()
