"""VN-SpeechMix/Libri2Mix baseline/LTRR audit and real-audio smoke test (not full epochs)."""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--model', default='SepReformer_Base_VnSpeechMix_16K',
                   choices=['SepReformer_Base_VnSpeechMix_16K', 'SepReformer_LTRR_VnSpeechMix_16K',
                            'SepReformer_Base_Libri2Mix_16K', 'SepReformer_LTRR_Libri2Mix_16K'])
    p.add_argument('--config', help='Optional pilot config, relative to --root or absolute')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--full-audio', action='store_true')
    p.add_argument('--report', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from loguru import logger
    logger.remove()
    import importlib
    dataset_module = importlib.import_module(f'models.{args.model}.dataset')
    VnSpeechMixDataset = getattr(dataset_module, 'VnSpeechMixDataset', None) or dataset_module.Libri2MixDataset
    _collate, _seed_worker = dataset_module._collate, dataset_module._seed_worker
    Model = importlib.import_module(f'models.{args.model}.model').Model
    from utils.util_implement import CriterionFactory, OptimizerFactory, SchedulerFactory
    from utils.util_system import set_random_seed
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'checks': {}, 'limitations': []}
    started = time.time()
    try:
        config_path = root / args.config if args.config else root/'models'/args.model/'configs.yaml'
        cfg = yaml.safe_load(config_path.read_text())['config']
        set_random_seed(**cfg['experiment'])
        torch.set_num_threads(4)
        device = torch.device(args.device)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        assert 1024 <= args.samples <= cfg['dataset']['max_len'] and args.steps > 0
        assert cfg['dataset']['sampling_rate'] == 16000
        assert cfg['model']['num_spks'] == len(cfg['dataset']['source_dirs']) == 2
        stride = cfg['model']['module_audio_enc']['stride']
        assert cfg['dataset']['sample_length_multiple'] == stride
        assert cfg['model']['module_audio_dec']['stride'] == stride
        manifest_path = root / cfg['dataset']['manifest']
        with manifest_path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        report['checks']['manifest'] = {'rows': len(rows), 'columns': list(rows[0])}
        datasets = {}
        for split, expected in cfg['dataset']['expected_sizes'].items():
            ds = VnSpeechMixDataset(cfg['dataset'], split)
            datasets[split] = ds
            assert len(ds) == expected, (split, len(ds), expected)
            names = {x[0] for x in ds.examples}
            manifest_names = [(r['key'] if 'key' in r else r['mixture_id']+'.wav')
                              for r in rows if r['split'] == split]
            assert len(manifest_names) == len(set(manifest_names)) == expected
            assert set(manifest_names) == names, (split, 'manifest/file mismatch')
            archive_names = []
            if ds.storage == 'zip':
                with zipfile.ZipFile(ds.archive_path) as archive:
                    archive_names = archive.namelist()
            for folder in [cfg['dataset']['mixture_dir'], *cfg['dataset']['source_dirs']]:
                if ds.storage == 'directory':
                    actual = {x.name for x in (ds.extracted_root/ds.partition_dir/folder).glob('*.wav')}
                else:
                    prefix = f'{ds.archive_root}/{ds.partition_dir}/{folder}/'
                    actual = {x[len(prefix):] for x in archive_names
                              if x.startswith(prefix) and x.endswith('.wav')}
                assert actual == names, (split, folder, 'file sets differ')
            indices = range(len(ds)) if args.full_audio else sorted({0, len(ds)-1, *random.Random(0).sample(range(len(ds)), min(8, len(ds)))})
            worst = 0.0
            for i in indices:
                _, mixpath, sourcepaths = ds.examples[i]
                arrays = []
                for path in [mixpath, *sourcepaths]:
                    # The dataset reader also validates sample rate, mono and finiteness.
                    audio = ds._load_audio(path)
                    arrays.append(audio)
                assert len({len(a) for a in arrays}) == 1
                error = float(np.max(np.abs(arrays[0]-arrays[1]-arrays[2])))
                assert error <= 2/32768, (mixpath, error)
                worst = max(worst, error)
            first, second = ds[0], ds[0]
            if split != 'train':
                assert np.array_equal(first['mix'], second['mix']), 'Unstable validation/test crop'
            report['checks'][split] = {'mixtures':len(ds), 'waveforms_checked':len(indices), 'max_sum_error':worst}
            print(split, report['checks'][split], flush=True)
        loader = torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['dataloader']['batch_size'],
            num_workers=args.workers, collate_fn=_collate, worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(0), shuffle=True)
        model = Model(**cfg['model']).to(device)
        criteria = CriterionFactory(cfg['criterion'], device).get_criterions()
        spectral, temporal = criteria[:2]
        optimizer = OptimizerFactory(cfg['optimizer'], model.parameters()).get_optimizers()[0]
        schedulers = SchedulerFactory(cfg['scheduler'], [optimizer]).get_schedulers()
        losses = []
        model.train()
        iterator = iter(loader)
        for step in range(args.steps):
            lengths, mix, sources, keys = next(iterator)
            mix = mix[:, :args.samples].to(device)
            sources = [s[:, :args.samples] for s in sources]
            lengths = lengths.clamp(max=args.samples)
            optimizer.zero_grad(set_to_none=True)
            estimates, aux = model(mix, input_sizes=lengths)
            assert len(estimates) == 2 and len(aux) == cfg['model']['num_stages']
            assert all(x.shape == mix.shape and torch.isfinite(x).all() for x in estimates)
            targets = spectral.prepare_targets(sources, lengths)
            freq = [spectral(estims=x,idx=i,input_sizes=lengths,target_attr=sources,prepared_targets=targets) for i,x in enumerate(aux)]
            loss = (.6*temporal(estims=estimates,input_sizes=lengths,target_attr=sources)+.4*sum(freq)/len(freq))/2
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['engine']['clip_norm'],error_if_nonfinite=True)
            before = next(model.parameters()).detach().clone()
            optimizer.step()
            schedulers[1].step()
            assert not torch.equal(before,next(model.parameters()).detach()), 'No parameter update'
            assert all(torch.isfinite(x).all() for x in model.parameters())
            losses.append(float(loss.detach()))
            print('optimizer step',step+1,'loss',losses[-1],'grad',float(grad),flush=True)
        model.eval()
        with torch.inference_mode():
            example = datasets['valid'][0]
            x = torch.tensor(example['mix'][:args.samples], device=device)[None]
            output, _ = model(x, return_aux=False)
            assert all(torch.isfinite(t).all() and t.shape == x.shape for t in output)
        # Real AdamW and scheduler serialization in a temporary directory, no production run touched.
        with tempfile.TemporaryDirectory(prefix='vn_baseline_checkpoint_') as folder:
            path = Path(folder)/'state.pth'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'schedulers':[s.state_dict() for s in schedulers]},path)
            saved = torch.load(path,map_location=device)
            assert saved['optimizer']['state']
            model.load_state_dict(saved['model'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            for s,state in zip(schedulers,saved['schedulers']): s.load_state_dict(state)
            with torch.inference_mode():
                restored,_ = model(x,return_aux=False)
            for left,right in zip(output,restored): torch.testing.assert_close(left,right,rtol=0,atol=0)
        report['checks']['model'] = {'parameters':sum(p.numel() for p in model.parameters()),
            'name':args.model,
            'device':str(device),'samples':args.samples,'batch_size':cfg['dataloader']['batch_size'],
            'steps':args.steps,'losses':losses,'checkpoint_roundtrip':'passed','workers':args.workers}
        from importlib.metadata import version, PackageNotFoundError
        packages = {}
        for name in ['torch','soundfile','pesq','pystoi','mir-eval','tensorboard']:
            try: packages[name] = version(name)
            except PackageNotFoundError: packages[name] = None
        report['packages'] = packages
        report['disk_free_bytes'] = shutil.disk_usage(root).free
        report['status'] = 'smoke_passed'
        report['limitations'] = ['Not a full epoch or production engine end-to-end check.',
            'Does not establish convergence or optimal hyperparameters.',
            'Audio audit sampled unless --full-audio; no all-file content hash.',
            'Checkpoint test checks serialization and restored outputs, not resumed next-update equality.']
        if args.device == 'cpu': report['limitations'].append('CUDA and GPU memory not tested.')
        if args.samples != cfg['dataset']['max_len']: report['limitations'].append('Training crop shortened for smoke test.')
    except Exception as e:
        report['status'] = 'failed'
        report['error'] = repr(e)
        raise
    finally:
        report['elapsed_seconds'] = time.time()-started
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Report:',args.report,report['status'],flush=True)


if __name__ == '__main__':
    main()
