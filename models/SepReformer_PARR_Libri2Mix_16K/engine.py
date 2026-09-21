import os
import shutil
import torch
import time
import soundfile as sf
import librosa
from loguru import logger
from tqdm import tqdm
from utils import util_engine, functions
from utils.evaluation_metrics import SeparationMetricsEvaluator
from utils.runtime_state import restore_runtime_state
from utils.parr_diagnostics import collect_parr_diagnostics, core_gradient_norm
from utils.run_contract import make_run_contract, validate_run_contract, file_sha256
from utils.paired_initialization import (
    paired_initialization_enabled,
    validate_resume_initialization,
)
from utils.decorators import *
from torch.utils.tensorboard import SummaryWriter


_CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".pkl")


def _peak_normalize(signal, peak=0.9):
    maximum = max(abs(signal))
    return signal if maximum <= 1.0e-12 else peak * signal / maximum


def _checkpoint_files(directory):
    """Return supported checkpoint files, oldest epoch first."""
    files = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith(_CHECKPOINT_EXTENSIONS):
            continue
        try:
            epoch = int(filename.split(".")[1])
        except (IndexError, ValueError):
            epoch = -1
        path = os.path.join(directory, filename)
        files.append((epoch, os.path.getmtime(path), path))
    return [path for _, _, path in sorted(files)]


def _select_scratch_checkpoint(directory, engine_mode):
    """Use latest for resume and validation-best for held-out evaluation."""
    preferred_names = (
        ("best.pth", "latest.pth")
        if engine_mode != "train"
        else ("latest.pth", "best.pth")
    )
    for filename in preferred_names:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    legacy = [
        path
        for path in _checkpoint_files(directory)
        if os.path.basename(path) not in preferred_names
    ]
    return legacy[-1] if legacy else None


def _load_model_initialization(checkpoint_path, model, location):
    """Load only shape-compatible model tensors from a pretraining checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    skipped = len(source_state) - len(compatible_state)
    missing = len(target_state) - len(compatible_state)
    model.load_state_dict(compatible_state, strict=False)
    logger.info(
        f"Initialized model from {checkpoint_path}: "
        f"loaded={len(compatible_state)}, skipped={skipped}, missing={missing}. "
        "Optimizer and epoch were intentionally not restored."
    )


def _load_training_resume(checkpoint_path, model, optimizer, schedulers, location):
    """Restore a compatible PARR training run, including optimizer and epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_resume_initialization(checkpoint, model, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_states = checkpoint.get("scheduler_state_dicts")
    if scheduler_states is not None:
        if len(scheduler_states) != len(schedulers):
            raise RuntimeError(
                "Checkpoint scheduler count does not match the current configuration."
            )
        for scheduler, state in zip(schedulers, scheduler_states):
            scheduler.load_state_dict(state)
    else:
        logger.warning(
            "This legacy checkpoint has no scheduler state; optimizer learning "
            "rate is restored, but scheduler history starts fresh."
        )
    start_epoch = int(checkpoint["epoch"]) + 1
    logger.info(f"Resumed training from {checkpoint_path} at epoch {start_epoch}.")
    best_valid_loss = float(
        checkpoint.get("best_valid_loss", checkpoint.get("valid_loss", float("inf")))
    )
    return start_epoch, best_valid_loss, checkpoint.get("runtime_state")


@logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device, work_dir=None):
        
        ''' Default setting '''
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.model.run_contract = make_run_contract(self.model, config)
        self.evaluation_provenance = {}
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.work_dir = work_dir or os.path.dirname(os.path.abspath(__file__))
        self.resume_runtime_state = None
        if self.engine_mode == "train":
            self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
            self.main_optimizer = optimizers[0]
            self.main_scheduler, self.warmup_scheduler = schedulers
        
        self.pretrain_weights_path = os.path.join(self.work_dir, "log", "pretrain_weights")
        if not getattr(args, "run_dir", None):
            os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = (os.path.join(self.work_dir, "checkpoints") if getattr(args, "run_dir", None)
                                     else os.path.join(self.work_dir, "log", "scratch_weights"))
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        
        # Training checkpoints are always written to scratch_weights. A scratch
        # checkpoint means "resume" (model + optimizer + epoch); a pretraining
        # checkpoint is only a model initialization and starts a fresh run.
        self.checkpoint_path = self.scratch_weights_path
        scratch_checkpoint = _select_scratch_checkpoint(
            self.scratch_weights_path, self.engine_mode
        )
        pretrain_checkpoints = ([] if getattr(args, "run_dir", None)
                                else _checkpoint_files(self.pretrain_weights_path))
        if getattr(args, 'run_dir', None):
            scratch_checkpoint = getattr(args, 'resume', None) if self.engine_mode == 'train' else getattr(args, 'checkpoint', None)
            pretrain_checkpoints = []
        elif getattr(args, 'checkpoint', None):
            scratch_checkpoint = args.checkpoint
        if self.engine_mode != "train":
            selected = scratch_checkpoint or (pretrain_checkpoints[-1] if pretrain_checkpoints else None)
            if selected is None:
                raise FileNotFoundError("Evaluation requires a trained checkpoint.")
            checkpoint = torch.load(selected, map_location="cpu")
            validate_run_contract(checkpoint, self.model.run_contract, selected, evaluation=True)
            self.evaluation_provenance = {
                "checkpoint": os.path.basename(selected),
                "checkpoint_sha256": file_sha256(selected),
                "run_contract": checkpoint["run_contract"],
                "paired_initialization_metadata": checkpoint.get("paired_initialization_metadata"),
            }
            self.model.paired_initialization_metadata = checkpoint.get("paired_initialization_metadata")
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_valid_loss = float("inf")
            logger.info(f"Evaluation model weights loaded strictly from {selected}")
            del checkpoint
        elif scratch_checkpoint:
            self.start_epoch, self.best_valid_loss, self.resume_runtime_state = _load_training_resume(
                scratch_checkpoint,
                self.model,
                self.main_optimizer,
                (self.main_scheduler, self.warmup_scheduler),
                location=self.device,
            )
            # Carry forward the historical best when it is available and valid
            # at this resume boundary. Never substitute a later best for an old milestone.
            if getattr(args, 'run_dir', None):
                for candidate in (scratch_checkpoint, os.path.join(os.path.dirname(scratch_checkpoint), 'best.pth')):
                    if not os.path.isfile(candidate):
                        continue
                    saved = torch.load(candidate, map_location='cpu')
                    if (int(saved['epoch']) < self.start_epoch
                            and saved.get('valid_loss') == self.best_valid_loss
                            and saved.get('run_contract') == self.model.run_contract):
                        shutil.copy2(candidate, os.path.join(self.checkpoint_path, 'best.pth'))
                        break
                else:
                    logger.warning('Historical best is unavailable at this resume boundary; '
                                   'best.pth will only appear after a new global validation best. '
                                   'Use an explicit historical --checkpoint for evaluation.')
        elif pretrain_checkpoints:
            if paired_initialization_enabled(self.config):
                raise RuntimeError(
                    "Paired initialization is enabled, but log/pretrain_weights contains a "
                    "checkpoint. Move it aside: a trained/pretrained model must not override "
                    "the controlled epoch-0 state."
                )
            _load_model_initialization(
                pretrain_checkpoints[-1], self.model, location=self.device
            )
            self.start_epoch = 1
            self.best_valid_loss = float("inf")
        else:
            self.start_epoch = 1
            self.best_valid_loss = float("inf")
        
        # Logging 
        model_was_training = self.model.training
        self.model.eval()
        self.computation_summary = util_engine.model_params_mac_summary(
            model=util_engine.InferenceView(self.model),
            input=torch.randn(
                1, self.config['check_computations']['dummy_len'], device=self.device
            ),
            dummy_input=torch.rand(
                1, self.config['check_computations']['dummy_len'], device=self.device
            ),
            metrics=self.config['check_computations'].get(
                'metrics', ['ptflops', 'thop', 'torchinfo']
            ),
        )
        self.model.train(model_was_training)
        # Restore only after initialization/profiling have consumed random draws.
        if self.resume_runtime_state is not None:
            restore_runtime_state(self.resume_runtime_state, self.dataloaders)
        elif self.engine_mode == "train" and self.start_epoch > 1:
            logger.warning("Legacy checkpoint lacks RNG state; resume is not reproducible.")
        
        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")
    
    @logger_wraps()
    def _train(self, dataloader, epoch):
        self.model.train()
        gradient_total = 0.0
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch, num_utts = 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            nnet_input = mixture
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            batch_utts = len(input_sizes)
            num_utts += batch_utts
            pbar.update(1)
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()
            estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid,
                    module_kwargs={"input_sizes": input_sizes})
            prepared_targets = self.PIT_SISNR_mag_loss.prepare_targets(src, input_sizes)
            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src, prepared_targets=prepared_targets))
                tot_loss_freq[idx] += batch_utts * cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
            tot_loss_time += batch_utts * cur_loss_s.item() / self.config['model']['num_spks']
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            cur_loss = (1-alpha) * cur_loss_s + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn)
            cur_loss = cur_loss / self.config['model']['num_spks']
            if not torch.isfinite(cur_loss):
                raise FloatingPointError(f"Non-finite train loss at epoch {epoch}, batch {num_batch}")
            cur_loss.backward()
            # Measure before gradient clipping. Only scalar reductions are retained.
            gradient_norm = core_gradient_norm(self.model)
            if gradient_norm is not None:
                gradient_total += gradient_norm
            if self.config['engine']['clip_norm']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'], error_if_nonfinite=True)
            self.main_optimizer.step()
            # PyTorch requires optimizer.step() before scheduler.step(). The
            # shifted lambda preserves the original LR used by each update.
            if self.warmup_scheduler.last_epoch < self.warmup_scheduler.warmup_steps:
                self.warmup_scheduler.step()
            dict_loss = {"T_Loss": tot_loss_time / num_utts}
            dict_loss.update({'F_Loss_' + str(idx): loss / num_utts for idx, loss in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        self.parr_gradient_norm = gradient_total / num_batch
        return tot_loss_time / num_utts, tot_loss_freq / num_utts, num_batch
    
    @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch, num_utts = 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, _ in dataloader:
                nnet_input = mixture
                nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                batch_utts = len(input_sizes)
                num_utts += batch_utts
                pbar.update(1)
                estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid,
                    module_kwargs={"input_sizes": input_sizes})
                prepared_targets = self.PIT_SISNR_mag_loss.prepare_targets(src, input_sizes)
                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src, prepared_targets=prepared_targets))
                    tot_loss_freq[idx] += batch_utts * cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
                cur_loss_s_SDR = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                if not all(torch.isfinite(value) for value in [cur_loss_s_SDR, *cur_loss_s_bn]):
                    raise FloatingPointError(f"Non-finite validation loss at batch {num_batch}")
                tot_loss_time += batch_utts * cur_loss_s_SDR.item() / self.config['model']['num_spks']
                dict_loss = {"T_Loss":tot_loss_time / num_utts}
                dict_loss.update({'F_Loss_' + str(idx): loss / num_utts for idx, loss in enumerate(tot_loss_freq)})
                pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_utts, tot_loss_freq / num_utts, num_batch
    
    @logger_wraps()
    def _test(self, dataloader, wav_dir=None, evaluation_tag="latest"):
        self.model.eval()
        num_batch = 0
        evaluation_config = self.config.get("evaluation", {})
        evaluator = SeparationMetricsEvaluator(
            sampling_rate=self.config["dataset"]["sampling_rate"],
            num_spks=self.config["model"]["num_spks"],
            compute_bss_eval=evaluation_config.get("compute_bss_eval", True),
            compute_pesq=evaluation_config.get("compute_pesq", True),
            compute_stoi=evaluation_config.get("compute_stoi", True),
            compute_estoi=evaluation_config.get("compute_estoi", True),
            bootstrap_samples=evaluation_config.get("bootstrap_samples", 2000),
            confidence=evaluation_config.get("confidence", 0.95),
            bootstrap_seed=evaluation_config.get("bootstrap_seed", 0),
        )
        warmed_up = False
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, key in dataloader:
                if len(key) > 1:
                    raise RuntimeError("Test batch size must be one.")
                nnet_input = mixture.to(self.device)
                if not warmed_up and evaluation_config.get("warmup_first_utterance", True):
                    torch.nn.parallel.data_parallel(
                        self.model, nnet_input, device_ids=self.gpuid,
                        module_kwargs={"return_aux": False},
                    )
                    torch.cuda.synchronize(self.device)
                    warmed_up = True

                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
                inference_start = time.perf_counter()
                estim_src, _ = torch.nn.parallel.data_parallel(
                    self.model, nnet_input, device_ids=self.gpuid,
                    module_kwargs={"return_aux": False},
                )
                torch.cuda.synchronize(self.device)
                latency_seconds = time.perf_counter() - inference_start
                peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)

                length = int(input_sizes[0].item())
                row = evaluator.add_utterance(
                    key=os.path.splitext(key[0])[0],
                    mixture=mixture[0].cpu().numpy(),
                    references=[value[0].cpu().numpy() for value in src],
                    estimates=[value[0].detach().cpu().numpy() for value in estim_src],
                    length=length,
                    latency_seconds=latency_seconds,
                    peak_vram_mb=peak_vram_mb,
                )
                num_batch += 1
                pbar.update(1)

                if self.engine_mode == "test_save":
                    if wav_dir is None:
                        wav_dir = os.path.join(self.work_dir, "wav_out")
                    os.makedirs(wav_dir, exist_ok=True)
                    sampling_rate = self.config["dataset"]["sampling_rate"]
                    mixture_audio = mixture[0, :length].cpu().numpy()
                    sf.write(os.path.join(wav_dir, key[0][:-4] + '_mixture.wav'), _peak_normalize(mixture_audio), sampling_rate)
                    for speaker_index, estimate in enumerate(estim_src):
                        output_audio = estimate[0, :length].detach().cpu().numpy()
                        sf.write(
                            os.path.join(wav_dir, key[0][:-4] + f'_out_{speaker_index}.wav'),
                            _peak_normalize(output_audio),
                            sampling_rate,
                        )

                pbar.set_postfix({
                    "SI-SNRi": evaluator.running_mean("si_snri"),
                    "PESQ": evaluator.running_mean("pesq_wb"),
                    "STOI": evaluator.running_mean("stoi"),
                    "RTF": evaluator.running_mean("rtf"),
                })
                del estim_src, nnet_input
                if self.engine_mode == "test_save":
                    del estimate
        pbar.close()
        output_directory = os.path.join(self.work_dir, "evaluation", evaluation_tag)
        summary = evaluator.save(
            output_directory,
            model_parameters=sum(parameter.numel() for parameter in self.model.parameters()),
            metadata={
                **self.evaluation_provenance,
                "evaluation_config": self.config,
                "device": torch.cuda.get_device_name(self.device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "experiment": self.config.get("experiment", {}),
                "timing_scope": "final waveform forward; no auxiliary heads; synchronized; batch_size=1",
                "memory_scope": "peak allocated incl. resident model (aux weights included); optimizer excluded in standalone test",
                "parameter_counts": util_engine.parameter_counts(self.model),
                "computation_summary": self.computation_summary,
            },
        )
        metrics = summary["metrics"]
        logger.info(
            f"Evaluation written to {output_directory}: "
            f"SI-SNRi={metrics['si_snri']['mean']:.4f} dB, "
            f"BSS-SDRi={metrics['bss_sdri']['mean']:.4f} dB, "
            f"WB-PESQ={metrics['pesq_wb']['mean']:.4f}, "
            f"STOI={metrics['stoi']['mean']:.4f}, "
            f"ESTOI={metrics['estoi']['mean']:.4f}, "
            f"RTF={metrics['rtf']['mean']:.6f}."
        )
        return metrics['si_snri']['mean'], metrics['bss_sdri']['mean'], num_batch

    @logger_wraps()
    def _inference_sample(self, sample):
        self.model.eval()
        self.fs = self.config["dataset"]["sampling_rate"]
        mixture, _ = librosa.load(sample,sr=self.fs)
        mixture = torch.tensor(mixture, dtype=torch.float32)[None]
        self.stride = self.config["model"]["module_audio_enc"]["stride"]
        remains = mixture.shape[-1] % self.stride
        if remains != 0:
            padding = self.stride - remains
            mixture_padded = torch.nn.functional.pad(mixture, (0, padding), "constant", 0)
        else:
            mixture_padded = mixture

        with torch.inference_mode():
            nnet_input = mixture_padded.to(self.device)
            estim_src, _ = torch.nn.parallel.data_parallel(
                self.model, nnet_input, device_ids=self.gpuid,
                module_kwargs={"return_aux": False},
            )
            mixture = torch.squeeze(mixture).cpu().numpy()
            sf.write(sample[:-4]+'_in.wav', _peak_normalize(mixture), self.fs)
            for i in range(self.config['model']['num_spks']):
                src = torch.squeeze(
                    estim_src[i][..., :mixture.shape[-1]]
                ).detach().cpu().numpy()
                sf.write(sample[:-4]+'_out_'+str(i)+'.wav', _peak_normalize(src), self.fs)

    
    @logger_wraps()
    def run(self):
        with torch.cuda.device(self.device):
            with SummaryWriter(os.path.join(self.work_dir, "tensorboard"), purge_step=self.start_epoch) as writer_src:
                if "test" in self.engine_mode:
                    on_test_start = time.time()
                    test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(
                        self.dataloaders['test'],
                        self.out_wav_dir,
                        evaluation_tag=f"checkpoint_epoch_{max(self.start_epoch - 1, 0):04d}",
                    )
                    on_test_end = time.time()
                    logger.info(f"[TEST] \n - Epoch {self.start_epoch:2d}: SI-SNRi = {test_loss_src_time_1:.4f} dB | BSS-SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                    logger.info(f"Testing done!")
                else:
                    # Do not consume validation RNG again at resume.
                    valid_loss_best = self.best_valid_loss
                    for epoch in range(self.start_epoch, self.config['engine']['max_epoch'] + 1):
                        train_start_time = time.time()
                        train_loss_src_time, train_loss_src_freq, train_num_batch = self._train(self.dataloaders['train'], epoch)
                        train_end_time = time.time()
                        valid_start_time = time.time()
                        valid_loss_src_time, valid_loss_src_freq, valid_num_batch = self._validate(self.dataloaders['valid'])
                        valid_end_time = time.time()
                        if (epoch > self.config['engine']['start_scheduling']
                                and self.warmup_scheduler.last_epoch >= self.warmup_scheduler.warmup_steps):
                            self.main_scheduler.step(valid_loss_src_time)
                        logger.info(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {train_loss_src_time:.4f} dB | Loss_f = {train_loss_src_freq:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
                        logger.info(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {valid_loss_src_time:.4f} dB | Loss_f = {valid_loss_src_freq:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")
                        if epoch in self.config['engine']['test_epochs']:
                            on_test_start = time.time()
                            test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(
                                self.dataloaders['test'], evaluation_tag=f"epoch_{epoch:04d}"
                            )
                            on_test_end = time.time()
                            logger.info(f"[TEST] \n - Epoch {epoch:2d}: SI-SNRi = {test_loss_src_time_1:.4f} dB | BSS-SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                        valid_loss_best = util_engine.save_checkpoint_per_best(
                            valid_loss_best,
                            valid_loss_src_time,
                            train_loss_src_time,
                            epoch,
                            self.model,
                            self.main_optimizer,
                            self.checkpoint_path,
                            schedulers=(self.main_scheduler, self.warmup_scheduler),
                            dataloaders=self.dataloaders,
                        )
                        util_engine.save_latest_checkpoint(
                            valid_loss_src_time,
                            train_loss_src_time,
                            epoch,
                            self.model,
                            self.main_optimizer,
                            self.checkpoint_path,
                            schedulers=(self.main_scheduler, self.warmup_scheduler),
                            best_valid_loss=valid_loss_best,
                            dataloaders=self.dataloaders,
                            milestone_epochs=self.config['engine'].get('checkpoint_epochs', [10, 20, 50, 100, 150, 200]),
                        )
                        scalars = {
                            'loss/train_time': train_loss_src_time,
                            'loss/valid_time': valid_loss_src_time,
                            'loss/train_frequency': train_loss_src_freq,
                            'loss/valid_frequency': valid_loss_src_freq,
                            'learning_rate/main': self.main_optimizer.param_groups[0]['lr'],
                            'duration/train_seconds': train_end_time - train_start_time,
                            'duration/valid_seconds': valid_end_time - valid_start_time,
                        }
                        diagnostics = collect_parr_diagnostics(
                            self.model, self.dataloaders['valid'].dataset, self.device,
                            count=self.config['engine'].get('diagnostic_examples', 4),
                            mvn=self.config['engine']['mvn'])
                        if diagnostics:
                            diagnostics['parr/shared_core_gradient_norm'] = self.parr_gradient_norm
                            scalars.update(diagnostics)
                        for tag, value in scalars.items():
                            writer_src.add_scalar(tag, value, epoch)
                        writer_src.flush()
                    logger.info(f"Training for {self.config['engine']['max_epoch']} epoches done!")
