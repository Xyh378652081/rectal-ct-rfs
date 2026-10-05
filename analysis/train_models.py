"""Run the downstream model update, with numerical and artifact checks."""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lifelines.exceptions import ConvergenceWarning
import no_neoadjuvant_core as core

OUT=core.OUT

def main():
    audit=json.loads((OUT/'00_protocol/input_audit.json').read_text(encoding='utf-8'))
    assert core.sha256(core.TABLE)==audit['analysis_table_sha256']
    for key,path in [('pyradiomics',core.RAD),('engineered_spatial',core.ENG),('cnn',core.CNN)]:
        assert core.sha256(path)==core.CFG['expected_feature_hashes'][key]
    assert len(core.ClinicalTransformer.continuous+core.ClinicalTransformer.categorical)==13
    removed=set(core.CFG['removed_predictors'])
    assert not removed.intersection(core.ClinicalTransformer.continuous+core.ClinicalTransformer.categorical)
    # Sanity check: perfectly ordered cases and controls should yield a time-dependent AUC of 1.
    t=np.array([10.,20.,40.,50.]);e=np.array([1,1,0,0]);s=np.array([2.,3.,0.,1.])
    km=core.km_censor(t,e)
    np.testing.assert_allclose(core.td_auc(t,e,s,36.,km),1.0,atol=1e-14)
    started=time.monotonic()
    (OUT/'08_logs/run_status.json').write_text(json.dumps({'status':'TRAINING','analysis_id':core.CFG['analysis_id']}),encoding='utf-8')
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always',ConvergenceWarning)
        data,results,stores=core.main()
    rows=[]
    for name,item in stores.items():
        for i,index in enumerate(item['train_idx']):
            rows.append({'patient_id':data.ids[index],'center':data.center[index],'model':name,
                         'time_months':data.time[index],'event':data.event[index],
                         'risk_score':item['train_score'][i],
                         'risk12':item['train_prob'][i,0],'risk24':item['train_prob'][i,1],'risk36':item['train_prob'][i,2]})
    pd.DataFrame(rows).to_csv(OUT/'03_models/development_patient_predictions.csv',index=False,encoding='utf-8-sig')
    primary=stores['M12']
    np.savez_compressed(OUT/'03_models/M12_design_matrices.npz',train=primary['train_design'],external=primary['test_design'],
                        train_indices=primary['train_idx'],external_indices=primary['test_idx'])
    warning_rows=[{'category':w.category.__name__,'message':str(w.message)} for w in records]
    (OUT/'08_logs/training_warnings.json').write_text(json.dumps(warning_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    status={'status':'PRIMARY_MODELS_TRAINED_AND_EVALUATED','analysis_id':core.CFG['analysis_id'],
            'models':list(stores),'elapsed_seconds':time.monotonic()-started,'warnings':len(warning_rows),
            'removed_variables_absent':all(not any(any(x in f for x in removed) for f in r['block_metadata']['feature_names']) for r in results['models'].values()),
            'postprocessing_pending':True}
    assert status['removed_variables_absent']
    (OUT/'08_logs/run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
