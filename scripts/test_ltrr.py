"""CPU regression checks: python -B -m scripts.test_ltrr."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from loguru import logger

from models.SepReformer_Base_VnSpeechMix_16K.model import Model as Baseline
from models.SepReformer_LTRR_VnSpeechMix_16K.model import Model
from models.SepReformer_LTRR_VnSpeechMix_16K.modules.ltrr import LTRR
from models.SepReformer_LTRR_VnSpeechMix_16K.modules.pairing import (
    BASELINE, CANDIDATE, apply_paired_initialization, baseline_artifact_config, make_run_contract,
)
from scripts.fixtures.legacy_nogate import SGLR
from utils.paired_initialization import apply_paired_initialization as pair_baseline
from utils.util_implement import CriterionFactory, OptimizerFactory, SchedulerFactory
from utils.util_engine import save_latest_checkpoint
from utils.runtime_state import restore_runtime_state
from models.SepReformer_LTRR_VnSpeechMix_16K.engine import _load_training_resume


ROOT = Path(__file__).resolve().parents[1]


class LTRRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT/'models'/CANDIDATE/'configs.yaml').read_text())["config"]
        cls.base = yaml.safe_load((ROOT/'models'/BASELINE/'configs.yaml').read_text())["config"]

    def test_shared_config_and_pipeline(self):
        self.assertEqual(baseline_artifact_config(self.config), self.base)
        for name in ('dataset.py', 'modules/module.py', 'modules/network.py'):
            self.assertEqual((ROOT/'models'/CANDIDATE/name).read_text(),
                             (ROOT/'models'/BASELINE/name).read_text())
        engine = (ROOT/'models'/CANDIDATE/'engine.py').read_text().replace(
            'from utils.run_contract import validate_run_contract, file_sha256\nfrom .modules.pairing import make_run_contract',
            'from utils.run_contract import make_run_contract, validate_run_contract, file_sha256')
        self.assertEqual(engine, (ROOT/'models'/BASELINE/'engine.py').read_text())

    def test_original_nogate_forward_and_gradients(self):
        options = self.config['model']['module_ltrr']
        for training in (False, True):
            old = SGLR(**options, use_speaker_gate=False).train(training)
            new = LTRR(**options).train(training)
            new.load_state_dict(old.state_dict(), strict=True)
            self.assertEqual(sum(p.numel() for p in new.parameters()), 89031)
            x = torch.randn(4, 128, 137, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            torch.manual_seed(21)
            expected = old(x)
            torch.manual_seed(21)
            actual = new(y)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            for (n1, p1), (n2, p2) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=0)

    def test_pairing_both_orders_and_config_rejection(self):
        for candidate_first in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                config = copy.deepcopy(self.config)
                config['paired_initialization']['directory'] = folder
                base_config = baseline_artifact_config(config)
                torch.manual_seed(0)
                base = Baseline(**base_config['model'])
                torch.manual_seed(0)
                model = Model(**config['model'])
                extra_before = {k:v.clone() for k,v in model.state_dict().items() if k.startswith('ltrr.')}
                if candidate_first:
                    meta = apply_paired_initialization(model, CANDIDATE, config, ROOT)
                    base_meta = pair_baseline(base, BASELINE, base_config, ROOT)
                else:
                    base_meta = pair_baseline(base, BASELINE, base_config, ROOT)
                    meta = apply_paired_initialization(model, CANDIDATE, config, ROOT)
                self.assertEqual(meta['initialization_sha256'], base_meta['initialization_sha256'])
                for key,value in base.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                for key,value in extra_before.items():
                    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                wrong = copy.deepcopy(config)
                wrong['optimizer']['AdamW']['lr'] *= 2
                with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                    apply_paired_initialization(model, CANDIDATE, wrong, ROOT)

    def test_final_stage_only_and_true_lengths(self):
        base = Baseline(**self.base['model']).eval()
        model = Model(**self.config['model']).eval()
        model.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(2, 2048)
        with torch.inference_mode():
            out, aux = model(x)
            plain, plain_aux = base(x)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(out,plain)))
            for stage, other in zip(aux, plain_aux):
                for a,b in zip(stage,other): torch.testing.assert_close(a,b,rtol=0,atol=0)
            legacy = SGLR(**self.config['model']['module_ltrr'], use_speaker_gate=False).eval()
            legacy.load_state_dict(model.ltrr.state_dict())
            original = model.ltrr
            model.ltrr = legacy
            old_out, _ = model(x)
            for a,b in zip(out,old_out): torch.testing.assert_close(a,b,rtol=0,atol=0)
            model.ltrr = original
            mixed,_ = model(x, input_sizes=torch.tensor([2048,1024]), return_aux=False)
            single,_ = model(x[1:2,:1024], return_aux=False)
            for a,b in zip(mixed,single):
                torch.testing.assert_close(a[1:2,:1024],b,rtol=0,atol=0)
                self.assertEqual(torch.count_nonzero(a[1,1024:]),0)

    def test_training_and_resume_next_update(self):
        config = copy.deepcopy(self.config)
        # This synthetic regression test must work on a fresh clone without audio.
        config['dataset']['manifest'] = None
        def build():
            model = Model(**config['model'])
            model.run_contract = make_run_contract(model, config)
            optimizer = OptimizerFactory(config['optimizer'],model.parameters()).get_optimizers()[0]
            schedulers = SchedulerFactory(config['scheduler'],[optimizer]).get_schedulers()
            return model, optimizer, schedulers
        model, optimizer, schedulers = build()
        spectral, temporal, *_ = CriterionFactory(config['criterion'],torch.device('cpu')).get_criterions()
        sources = [torch.randn(2,2048),torch.randn(2,2048)]
        lengths = torch.tensor([2048,2048])
        def step(model, optimizer, schedulers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out, aux = model(sum(sources),input_sizes=lengths)
            prepared = spectral.prepare_targets(sources,lengths)
            freq = [spectral(estims=x,idx=i,input_sizes=lengths,target_attr=sources,prepared_targets=prepared)
                    for i,x in enumerate(aux)]
            loss = (.6*temporal(estims=out,input_sizes=lengths,target_attr=sources)+.4*sum(freq)/len(freq))/2
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for p in model.ltrr.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(torch.count_nonzero(model.ltrr.gcu.memory.out_proj.weight.grad),0)
            before = model.ltrr.alpha.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step(); schedulers[1].step()
            self.assertFalse(torch.equal(before,model.ltrr.alpha.detach()))
            return loss.detach()
        step(model,optimizer,schedulers)
        with tempfile.TemporaryDirectory() as folder:
            save_latest_checkpoint(1.,1.,1,model,optimizer,folder,schedulers=schedulers)
            uninterrupted = step(model,optimizer,schedulers)
            resumed, opt, sched = build()
            epoch, best, runtime = _load_training_resume(
                str(Path(folder)/'latest.pth'),resumed,opt,sched,torch.device('cpu'))
            self.assertEqual(epoch,2)
            restore_runtime_state(runtime)
            continued = step(resumed,opt,sched)
            torch.testing.assert_close(uninterrupted,continued,rtol=0,atol=0)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(any(k.endswith('modules/ltrr.py') for k in resumed.run_contract['source_files']))
            self.assertTrue(any(k.endswith('modules/pairing.py') for k in resumed.run_contract['source_files']))


if __name__ == '__main__':
    logger.remove()
    torch.set_num_threads(2)
    unittest.main()
