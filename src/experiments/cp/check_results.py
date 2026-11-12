"""Validate compact paper archives and optionally replay-check a smoke run."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from run_cp import methods_for

HERE = Path(__file__).resolve().parent


def check_archive(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    current = json.loads((HERE / 'cp_tensor_experiments_normalized.ipynb').read_text())
    with zipfile.ZipFile(root / 'original_sources.zip') as sources:
        for group in ['filtered', 'unfiltered']:
            folder = root / group
            prov = json.loads((folder / 'provenance.json').read_text())
            for label, origin in prov['original_sources'].items():
                raw = sources.read(f'{label}/cp_tensor_experiments_normalized.ipynb')
                assert hashlib.sha256(raw).hexdigest() == origin['source_sha256']
                assert hashlib.sha256(sources.read(f'{label}/run_cp.py')).hexdigest() == origin['runner_sha256']
                original = json.loads(raw)
                for i in [1, 5, 7, 9, 11]:
                    assert current['cells'][i]['source'] == original['cells'][i]['source'], (label, i)
            cfg = json.loads((folder / 'settings.json').read_text())
            assert cfg['trials'] == 10 and cfg['dense_noise_relative_level'] == .02
            assert cfg['replacement_scale'] == 2 and cfg['replacement_quantile'] == .9
            summary = json.loads((folder / 'summary.json').read_text())
            names = {f'{case}__{method}__trial{t}' for case in ['exact','query_noise','approximate']
                     for method, _, _ in methods_for(case,cfg) for t in range(10)}
            assert set(summary) == names
            with np.load(folder/'histories.npz') as z:
                expected = {f'{case}__{method}__{metric}' for case in ['exact','query_noise','approximate']
                            for method,_,_ in methods_for(case,cfg)
                            for metric in ['target_error','observed_error','elapsed_seconds']}
                assert set(z.files) == expected
                for name in z.files:
                    case = name.split('__')[0]
                    assert z[name].shape == (10,cfg['outer_sweeps'][case]+1)
                    assert np.isfinite(z[name]).all(), name
    with np.load(root/'filtered/inputs.npz') as a,np.load(root/'unfiltered/inputs.npz') as b:
        assert set(a.files)==set(b.files)
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
        np.testing.assert_allclose(np.linalg.norm(a['dense_noise'])/np.linalg.norm(a['exact_target']),.02)
    print('PASS: archive checksums, original sources, numerical cells, paired inputs, 250 method/trial summaries.')


def check_smoke(run, reference):
    for group in ['filtered','unfiltered']:
        cfg=json.loads((run/group/'settings.json').read_text())
        prov=json.loads((run/group/'provenance.json').read_text())
        assert prov['status']=='complete' and cfg['trials']==1
        assert all(n==2 for n in cfg['outer_sweeps'].values())
        with np.load(run/group/'histories.npz') as actual,np.load(reference/group/'histories.npz') as expected:
            for name in actual.files:
                if name.endswith('elapsed_seconds'):
                    continue
                np.testing.assert_allclose(actual[name],expected[name][:1,:3],rtol=1e-12,atol=1e-14,err_msg=name)
        with np.load(run/group/'inputs.npz') as actual,np.load(reference/group/'inputs.npz') as expected:
            for key in actual.files:
                wanted=expected[key]
                if key.startswith('initial_factor_') or key in ['initialization_seeds','sampling_seeds','query_seeds']:
                    wanted=wanted[:1]
                np.testing.assert_array_equal(actual[key],wanted)
    print('PASS: all 25 smoke trajectories match the first two sweeps of the final paper runs; inputs match exactly.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=HERE/'saved_runs/paper')
    parser.add_argument('--smoke-run',type=Path)
    args=parser.parse_args()
    check_archive(args.archive)
    if args.smoke_run is not None:
        check_smoke(args.smoke_run,args.archive)
