#!/usr/bin/env python3
"""Notebook-friendly HKL-VITS quality and performance testing harness.

This evaluator is designed for a notebook or Colab runtime where the training
and inference objects already exist in scope. It benchmarks both objective
speech quality and runtime performance.

The script intentionally keeps the notebook-facing API simple:
- `NotebookQualityConfig`
- `run_in_notebook_quality_eval(...)`
- `run_quality_testing_in_notebook(...)`

It computes a broad set of metrics when the required libraries are installed:
- latency / throughput / repeatability
- waveform and spectrum statistics
- pitch statistics (F0)
- PESQ, STOI, WER, CER and MCD when reference audio/text are available
- CPU, RAM and GPU resource usage
- optional plots and a JSON summary report
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

librosa = None
psutil = None
pd = None
plt = None
pesq_score_fn = None
stoi_score_fn = None
jiwer_wer = None
jiwer_cer = None
load_hf_metric = None
pw = None
parselmouth = None
mir_eval = None
pynvml = None
torch = None

for module_name in (
    "librosa",
    "psutil",
    "pandas",
    "matplotlib.pyplot",
    "pesq",
    "pystoi",
    "jiwer",
    "evaluate",
    "pyworld",
    "parselmouth",
    "mir_eval",
    "pynvml",
    "torch",
):
    try:
        module = importlib.import_module(module_name)
        if module_name == "librosa":
            librosa = module
        elif module_name == "psutil":
            psutil = module
        elif module_name == "pandas":
            pd = module
        elif module_name == "matplotlib.pyplot":
            plt = module
        elif module_name == "pesq":
            pesq_score_fn = module.pesq
        elif module_name == "pystoi":
            stoi_score_fn = module.stoi
        elif module_name == "jiwer":
            jiwer_wer = module.wer
            jiwer_cer = module.cer
        elif module_name == "evaluate":
            load_hf_metric = module.load
        elif module_name == "pyworld":
            pw = module
        elif module_name == "parselmouth":
            parselmouth = module
        elif module_name == "mir_eval":
            mir_eval = module
        elif module_name == "pynvml":
            pynvml = module
        elif module_name == "torch":
            torch = module
    except Exception:
        pass


# Notebook compatibility: allow the file to be imported without the training
# cell already having executed. The real notebook objects are optional here.
config: Any = globals().get("config")
pipeline: Any = globals().get("pipeline")

logger = logging.getLogger("notebook_quality_eval")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class NotebookQualityConfig:
    dataset_dir: str = "/content/kannada_tts_dataset"
    wav_dir: str = "/content/kannada_tts_dataset/wavs"
    text_dir: str = "/content/kannada_tts_dataset/text"
    output_dir: str = "/content/quality_eval"
    reference_wav_dir: str = "/content/kannada_tts_dataset/wavs"
    sample_rate: int = 16000
    sample_limit: int = 100
    repeat_per_sample: int = 3
    fallback_samples: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {"id": "sample_01", "name": "Simple greeting", "text": "ನಮಸ್ಕಾರ"},
            {"id": "sample_02", "name": "Number", "text": "2500"},
            {"id": "sample_03", "name": "Currency", "text": "₹2500"},
            {"id": "sample_04", "name": "Percentage", "text": "50%"},
            {"id": "sample_05", "name": "Mixed language", "text": "ಬೆಂಗಳೂರು City"},
            {"id": "sample_06", "name": "Question sentence", "text": "ನೀವು ಹೇಗಿದ್ದೀರಿ?"},
            {"id": "sample_07", "name": "Long paragraph", "text": "ಕನ್ನಡವು ಒಂದು ಪ್ರಾಚೀನ ಭಾಷೆಯಾಗಿದ್ದು, ಅದರಲ್ಲಿ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಂಸ್ಕೃತಿ ಬಹಳ ಆಳವಾಗಿಟ್ಟುಕೊಂಡಿದೆ. ಈ ಭಾಷೆಯ ಧ್ವವಿಮಾಧ್ಯಮದ ಗುಣಮಟ್ಟ ಮತ್ತು ಅರ್ಥಪೂರ್ಣತೆ ಬಹಳ ಮುಖ್ಯವಾಗಿದೆ."},
            {"id": "sample_08", "name": "Conjunct consonant", "text": "ಕ್ಷೇತ್ರ"},
        ]
    )

    def ensure_dirs(self) -> None:
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        Path(self.wav_dir).mkdir(parents=True, exist_ok=True)
        Path(self.text_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class SampleLoader:
    def __init__(self, config: NotebookQualityConfig):
        self.config = config

    def load(self) -> List[Dict[str, str]]:
        text_dir = Path(self.config.text_dir)
        if text_dir.exists():
            selected = []
            for txt_path in sorted(text_dir.glob("*.txt"))[: self.config.sample_limit]:
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    sample = {"id": txt_path.stem, "name": txt_path.stem, "text": text}
                    wav_path = Path(self.config.reference_wav_dir) / f"{txt_path.stem}.wav"
                    if wav_path.exists():
                        sample["reference_audio_path"] = str(wav_path)
                    selected.append(sample)
            if selected:
                return selected

        return self.config.fallback_samples[: self.config.sample_limit]


class AudioMetrics:
    @staticmethod
    def _read_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
        wav, sr = sf.read(audio_path)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1) if wav.shape[1] > 1 else wav[:, 0]
        wav = np.ascontiguousarray(wav.reshape(-1), dtype=np.float32)
        return wav, int(sr)

    @staticmethod
    def align_audio_lengths(reference: np.ndarray, generated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        reference = np.asarray(reference, dtype=np.float32)
        generated = np.asarray(generated, dtype=np.float32)

        if reference.ndim > 1:
            reference = np.mean(reference, axis=1) if reference.shape[1] > 1 else reference[:, 0]
        if generated.ndim > 1:
            generated = np.mean(generated, axis=1) if generated.shape[1] > 1 else generated[:, 0]

        reference = np.ascontiguousarray(reference.reshape(-1), dtype=np.float32)
        generated = np.ascontiguousarray(generated.reshape(-1), dtype=np.float32)

        if reference.size == 0 or generated.size == 0:
            return reference, generated

        target_len = min(reference.shape[0], generated.shape[0])
        if target_len <= 0:
            return reference, generated

        if reference.shape[0] != target_len:
            reference = reference[:target_len]
        if generated.shape[0] != target_len:
            generated = generated[:target_len]

        return reference, generated

    @staticmethod
    def audio_info(audio_path: Path) -> Dict[str, Any]:
        wav, sr = AudioMetrics._read_audio(audio_path)
        duration = len(wav) / sr
        rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        return {
            "sample_rate": int(sr),
            "duration_seconds": float(duration),
            "samples": int(len(wav)),
            "file_size_bytes": int(audio_path.stat().st_size),
            "rms": rms,
            "peak": peak,
        }

    @staticmethod
    def summarize(values: List[float]) -> Dict[str, float]:
        ordered = sorted(values)
        if not ordered:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "p95": 0.0}
        return {
            "min": float(min(ordered)),
            "max": float(max(ordered)),
            "avg": float(statistics.mean(ordered)),
            "p95": float(ordered[max(0, int(len(ordered) * 0.95) - 1)]),
        }

    @staticmethod
    def waveform_stats(audio_path: Path) -> Dict[str, float]:
        wav, sr = AudioMetrics._read_audio(audio_path)
        if wav.size == 0:
            return {"zero_crossing_rate": 0.0, "energy": 0.0, "rms": 0.0}
        zcr = float(np.mean(np.abs(np.diff(np.signbit(wav))))) if wav.size > 1 else 0.0
        energy = float(np.mean(np.square(wav)))
        rms = float(np.sqrt(energy))
        return {
            "zero_crossing_rate": zcr,
            "energy": energy,
            "rms": rms,
        }

    @staticmethod
    def spectral_stats(audio_path: Path) -> Dict[str, float]:
        if librosa is None:
            return {}
        wav, sr = AudioMetrics._read_audio(audio_path)
        if wav.size == 0:
            return {}
        spec = np.abs(librosa.stft(wav))
        cent = librosa.feature.spectral_centroid(y=wav, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=wav)
        return {
            "spectral_centroid_mean": float(np.mean(cent)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_flatness_mean": float(np.mean(flatness)),
        }

    @staticmethod
    def pitch_stats(audio_path: Path) -> Dict[str, float]:
        wav, sr = AudioMetrics._read_audio(audio_path)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        result: Dict[str, float] = {}

        if parselmouth is not None:
            try:
                sound = parselmouth.Sound(str(audio_path))
                pitch = sound.to_pitch()
                voicing = pitch.selected_array[:, 1]
                if voicing.size:
                    valid = voicing > 0
                    if np.any(valid):
                        result.update(
                            {
                                "f0_mean_hz": float(np.mean(voicing[valid])),
                                "f0_std_hz": float(np.std(voicing[valid])),
                                "f0_min_hz": float(np.min(voicing[valid])),
                                "f0_max_hz": float(np.max(voicing[valid])),
                            }
                        )
            except Exception:
                pass

        if not result and librosa is not None:
            try:
                f0 = librosa.yin(wav, fmin=50, fmax=300, sr=sr)
                valid = np.isfinite(f0)
                if np.any(valid):
                    result.update(
                        {
                            "f0_mean_hz": float(np.mean(f0[valid])),
                            "f0_std_hz": float(np.std(f0[valid])),
                            "f0_min_hz": float(np.min(f0[valid])),
                            "f0_max_hz": float(np.max(f0[valid])),
                        }
                    )
            except Exception as exc:
                logger.warning("librosa F0 fallback failed: %s", exc)

        if not result and pw is not None:
            try:
                if wav.ndim != 1 or wav.size == 0:
                    raise ValueError("waveform is empty or not one-dimensional")
                wav_1d = wav.astype(np.float64)
                f0, time_axis = pw.dio(wav_1d, sr)
                sp = pw.cheaptrick(wav_1d, f0, time_axis, sr)
                f0 = pw.stonemask(f0, time_axis, sp, sr)
                valid = f0 > 0
                if np.any(valid):
                    result.update(
                        {
                            "f0_mean_hz": float(np.mean(f0[valid])),
                            "f0_std_hz": float(np.std(f0[valid])),
                            "f0_min_hz": float(np.min(f0[valid])),
                            "f0_max_hz": float(np.max(f0[valid])),
                        }
                    )
            except Exception as exc:
                logger.warning("PyWorld pitch analysis failed: %s", exc)

        return result

# ==========================================================
# Speech Quality Metrics
# ==========================================================

class SpeechQualityMetrics:

    @staticmethod
    def compute_all(
        generated_audio_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        predicted_text: Optional[str] = None,
        sample_rate: int = 16000,
    ):

        results = {}

        ###################################################
        # Basic Audio Quality
        ###################################################

        results["waveform"] = AudioMetrics.waveform_stats(
            generated_audio_path
        )

        results["spectral"] = AudioMetrics.spectral_stats(
            generated_audio_path
        )

        results["pitch"] = AudioMetrics.pitch_stats(
            generated_audio_path
        )

        results["audio_info"] = AudioMetrics.audio_info(
            generated_audio_path
        )

        ###################################################
        # Objective Metrics
        ###################################################

        if reference_audio_path is not None and reference_audio_path.exists():

            results["pesq"] = ObjectiveQualityMetrics.pesq(
                reference_audio_path,
                generated_audio_path,
                sample_rate,
            )

            results["stoi"] = ObjectiveQualityMetrics.stoi(
                reference_audio_path,
                generated_audio_path,
                sample_rate,
            )

            results["mcd"] = ObjectiveQualityMetrics.mcd(
                reference_audio_path,
                generated_audio_path,
                sample_rate,
            )

        ###################################################
        # Intelligibility
        ###################################################

        if (
            reference_text is not None
            and predicted_text is not None
        ):

            results["wer"] = ObjectiveQualityMetrics.wer(
                reference_text,
                predicted_text,
            )

            results["cer"] = ObjectiveQualityMetrics.cer(
                reference_text,
                predicted_text,
            )

        ###################################################
        # Duration
        ###################################################

        info = results["audio_info"]

        results["duration"] = info["duration_seconds"]

        ###################################################
        # Speech Rate
        ###################################################

        if reference_text:

            words = len(reference_text.split())

            duration = max(
                info["duration_seconds"],
                0.01,
            )

            results["words_per_second"] = words / duration

            chars = len(reference_text)

            results["chars_per_second"] = chars / duration

        ###################################################
        # Silence Ratio
        ###################################################

        wav, sr = AudioMetrics._read_audio(
            generated_audio_path
        )

        silence = np.sum(
            np.abs(wav) < 0.01
        )

        results["silence_ratio"] = silence / len(wav)

        ###################################################
        # Dynamic Range
        ###################################################

        peak = np.max(np.abs(wav))

        rms = np.sqrt(np.mean(wav ** 2))

        results["dynamic_range_db"] = (
            20 * np.log10((peak + 1e-8) / (rms + 1e-8))
        )

        ###################################################
        # Clipping
        ###################################################

        clipping = np.sum(
            np.abs(wav) >= 0.99
        )

        results["clipping_percent"] = (
            clipping / len(wav)
        ) * 100

        return results
class PerformanceCounters:
    @staticmethod
    def cpu_ram_snapshot() -> Dict[str, float]:
        if psutil is None:
            return {}
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_percent": float(psutil.virtual_memory().percent),
            "process_rss_mb": float(memory_info.rss / (1024 * 1024)),
            "process_vms_mb": float(memory_info.vms / (1024 * 1024)),
        }

    @staticmethod
    def gpu_snapshot() -> Dict[str, Any]:
        if pynvml is None:
            return {}
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            devices = []
            for idx in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append(
                    {
                        "device_index": idx,
                        "gpu_util_percent": float(util.gpu),
                        "memory_util_percent": float((mem.used / mem.total) * 100.0) if mem.total else 0.0,
                        "memory_used_mb": float(mem.used / (1024 * 1024)),
                        "memory_total_mb": float(mem.total / (1024 * 1024)),
                    }
                )
            return {"gpu_devices": devices}
        except Exception as exc:
            logger.warning("GPU telemetry unavailable: %s", exc)
            return {}


class NotebookQualityEvaluator:
    """Uses the already-defined training notebook objects to evaluate TTS quality."""

    def __init__(self, config: NotebookQualityConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _invoke_synth_fn(
        synth_fn: Callable[..., Any],
        text: str,
        output_path: Path,
        config_obj: Optional[Any] = None,
        pipeline_obj: Optional[Any] = None,
    ) -> Any:
        signature = inspect.signature(synth_fn)
        parameters = signature.parameters

        if "config" in parameters and "pipeline" in parameters and "text" in parameters:
            return synth_fn(config_obj, pipeline_obj, text, output_path=str(output_path))

        if "text" in parameters and "output_path" in parameters:
            return synth_fn(text, output_path=str(output_path))

        if "text" in parameters:
            return synth_fn(text)

        raise TypeError(
            "The provided synth function must support either "
            "(config, pipeline, text, output_path=...) or (text, output_path=...)."
        )

    def _compute_quality_metrics(
        self,
        sample,
        generated_path,
        latency=None,
    ):

        reference_audio = None

        if "reference_audio_path" in sample:
            reference_audio = Path(sample["reference_audio_path"])

        reference_text = sample.get("text")

        # Replace with Whisper ASR output if available
        predicted_text = None

        metrics = SpeechQualityMetrics.compute_all(
            generated_audio_path=generated_path,
            reference_audio_path=reference_audio,
            reference_text=reference_text,
            predicted_text=predicted_text,
            sample_rate=self.config.sample_rate,
        )

        ##################################################
        # Runtime Metrics
        ##################################################

        if latency is not None:

            duration = metrics["duration"]

            metrics["latency"] = latency

            metrics["real_time_factor"] = (
                latency / duration
                if duration > 0
                else None
            )

            metrics["throughput_characters_per_second"] = (
                len(reference_text) / latency
                if latency > 0 else None
            )

            metrics["throughput_words_per_second_runtime"] = (
                len(reference_text.split()) / latency
                if latency > 0 else None
            )

            metrics["milliseconds_per_character"] = (
                latency * 1000 / len(reference_text)
                if len(reference_text) > 0 else None
            )

        return metrics

    def run_quality_suite(
        self,
        synth_fn: Callable[..., Any],
        samples: List[Dict[str, str]],
        repeat: int = 3,
        config_obj: Optional[Any] = None,
        pipeline_obj: Optional[Any] = None,
    ) -> Dict[str, Any]:
        all_latencies: List[float] = []
        sample_results: List[Dict[str, Any]] = []

        for sample in tqdm(samples, desc="Running notebook quality samples"):
            sample_dir = self.output_dir / sample["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)

            run_latencies: List[float] = []
            run_outputs: List[Dict[str, Any]] = []
            run_quality_metrics: List[Dict[str, Any]] = []

            for idx in range(1, repeat + 1):
                start = time.perf_counter()
                output_path = sample_dir / f"{sample['id']}_{idx}.wav"
                generated = self._invoke_synth_fn(
                    synth_fn=synth_fn,
                    text=sample["text"],
                    output_path=output_path,
                    config_obj=config_obj,
                    pipeline_obj=pipeline_obj,
                )
                latency = time.perf_counter() - start
                all_latencies.append(latency)
                run_latencies.append(latency)

                if isinstance(generated, tuple):
                    output_path = Path(generated[0])
                elif isinstance(generated, (str, os.PathLike)):
                    output_path = Path(generated)

                perf_start = time.perf_counter()
                quality_metrics = self._compute_quality_metrics(
                        sample,
                        output_path,
                        latency,
                    )
                perf_elapsed = time.perf_counter() - perf_start

                run_outputs.append(
                    {
                        "run": idx,
                        "latency_seconds": float(latency),
                        "metric_compute_seconds": float(perf_elapsed),
                        "audio_path": str(output_path),
                        "performance": PerformanceCounters.cpu_ram_snapshot(),
                        "gpu": PerformanceCounters.gpu_snapshot(),
                        "quality_metrics": quality_metrics,
                    }
                )
                run_quality_metrics.append(quality_metrics)

            sample_results.append(
                {
                    "id": sample["id"],
                    "name": sample["name"],
                    "text": sample["text"],
                    "latency_summary": AudioMetrics.summarize(run_latencies),
                    "quality_summary": self._SummarizeQualityMetrics(run_quality_metrics),
                    "runs": run_outputs,
                }
            )

        return {

    "sample_count": len(samples),

    "repeat_per_sample": repeat,

    "latency_summary": AudioMetrics.summarize(all_latencies),

    "performance_summary": self._overall_performance_summary(),

    "quality_summary": self._SummarizeQualityMetrics(
        [
            run["quality_metrics"]
            for sample in sample_results
            for run in sample["runs"]
        ]
    ),

    "samples": sample_results,

}

    @staticmethod
    def _SummarizeQualityMetrics(metrics_list):
        summary = {}
        metric_names = [

            "pesq",
            "stoi",
            "mcd",
            "wer",
            "cer",
            "duration",
            "words_per_second",
            "chars_per_second",
            "real_time_factor",
            "latency",
            "throughput_characters_per_second",
            "throughput_words_per_second_runtime",
            "milliseconds_per_character",
            "silence_ratio",
            "dynamic_range_db",
            "clipping_percent",

        ]

        for metric in metric_names:
            values = []
            for m in metrics_list:
                if metric in m and m[metric] is not None:
                    values.append(float(m[metric]))
            if values:
                summary[metric] = AudioMetrics.summarize(values)
        for section in (
            "waveform",
            "spectral",
            "pitch",

        ):
            section_values = {}
            for metrics in metrics_list:
                if section not in metrics:
                    continue
                for key, value in metrics[section].items():
                    if value is None:
                        continue
                    section_values.setdefault(key, []).append(float(value))
            if section_values:
                summary[section] = {
                    k: AudioMetrics.summarize(v)
                    for k, v in section_values.items()
                }
            return summary    
    @staticmethod
    def _overall_performance_summary() -> Dict[str, Any]:
        return {
            "cpu_ram": PerformanceCounters.cpu_ram_snapshot(),
            "gpu": PerformanceCounters.gpu_snapshot(),
        }

    def save_report(self, report: Dict[str, Any]) -> Path:

        def to_python(obj):
            import numpy as np

            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                return [to_python(v) for v in obj]

            elif isinstance(obj, tuple):
                return tuple(to_python(v) for v in obj)

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, np.generic):
                return obj.item()

            return obj

        report = to_python(report)

        report_path = self.output_dir / "quality_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Saved quality report to %s", report_path)

        if pd is not None:

            rows = []

            for sample in report.get("samples", []):

                for run in sample.get("runs", []):

                    row = {
                        "sample_id": sample["id"],
                        "sample_name": sample["name"],
                        "text": sample["text"],
                        "run": run["run"],
                        "latency_seconds": run["latency_seconds"],
                        "audio_path": run["audio_path"],
                    }

                    row.update(run["quality_metrics"].get("audio_info", {}))

                    q = run["quality_metrics"]

                    for metric in (
                        "pesq",
                        "stoi",
                        "mcd",
                        "wer",
                        "cer",
                        "duration",
                        "words_per_second",
                        "chars_per_second",
                        "silence_ratio",
                        "dynamic_range_db",
                        "clipping_percent",
                    ):
                        if metric in q:
                            row[metric] = q[metric]

                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(
                    self.output_dir / "quality_report.csv", encoding="utf-8-sig",
                    index=False,
                )

        return report_path
def build_notebook_quality_runner(config: NotebookQualityConfig) -> NotebookQualityEvaluator:
    config.ensure_dirs()
    return NotebookQualityEvaluator(config)


def run_in_notebook_quality_eval(
    config: NotebookQualityConfig,
    synth_fn: Callable[..., Any],
    repeat: int = 3,
    config_obj: Optional[Any] = None,
    pipeline_obj: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the notebook quality suite after training has already produced a model."""
    loader = SampleLoader(config)
    samples = loader.load()
    evaluator = build_notebook_quality_runner(config)
    report = evaluator.run_quality_suite(
        synth_fn=synth_fn,
        samples=samples,
        repeat=repeat,
        config_obj=config_obj,
        pipeline_obj=pipeline_obj,
    )
    evaluator.save_report(report)
    return report


@staticmethod
def summarize_quality(samples):

    metrics = [
        "pesq",
        "stoi",
        "mcd",
        "wer",
        "cer",
        "duration",
        "words_per_second",
        "chars_per_second",
        "silence_ratio",
        "dynamic_range_db",
        "clipping_percent",
    ]

    summary = {}

    for metric in metrics:

        values = []

        for sample in samples:

            for run in sample["runs"]:

                q = run["quality_metrics"]

                if metric in q and q[metric] is not None:
                    values.append(q[metric])

        if values:

            summary[metric] = AudioMetrics.summarize(values)

    return summary

def run_quality_testing_in_notebook(
    test_cfg: Optional[NotebookQualityConfig] = None,
    repeat: int = 3,
    sample_limit: int = 8,
    synth_fn: Optional[Callable[..., Any]] = None,
    config_obj: Optional[Any] = None,
    pipeline_obj: Optional[Any] = None,
) -> Dict[str, Any]:
    """Notebook-ready wrapper using the same notebook globals for config/pipeline."""
    if test_cfg is None:
        test_cfg = NotebookQualityConfig(
            dataset_dir="/content/kannada_tts_dataset",
            wav_dir="/content/kannada_tts_dataset/wavs",
            text_dir="/content/kannada_tts_dataset/text",
            output_dir="/content/quality_eval",
            sample_limit=sample_limit,
            repeat_per_sample=repeat,
        )

    if synth_fn is None:
        synth_fn = globals().get("synthesize_with_mms_backbone")
    if config_obj is None:
        config_obj = globals().get("config")
    if pipeline_obj is None:
        pipeline_obj = globals().get("pipeline")

    if synth_fn is None:
        raise RuntimeError("`synthesize_with_mms_backbone` is not available in the notebook scope.")
    if config_obj is None:
        raise RuntimeError("The real notebook `config` object is missing. Run the training cell first.")
    if pipeline_obj is None:
        raise RuntimeError("The real notebook `pipeline` object is missing. Run the inference setup first.")

    return run_in_notebook_quality_eval(
        config=test_cfg,
        synth_fn=synth_fn,
        repeat=repeat,
        config_obj=config_obj,
        pipeline_obj=pipeline_obj,
    )


cfg = NotebookQualityConfig(
    dataset_dir="/content/kannada_tts_dataset",
    wav_dir="/content/kannada_tts_dataset/wavs",
    text_dir="/content/kannada_tts_dataset/text",
    reference_wav_dir="/content/kannada_tts_dataset/wavs",
    output_dir="/content/quality_eval",
    sample_limit=100,
    repeat_per_sample=3,
)

report = run_quality_testing_in_notebook(
    test_cfg=cfg,
    repeat=3,
    sample_limit=8,
    synth_fn=synthesize_with_mms_backbone,
    config_obj=config,
    pipeline_obj=pipeline,
)

print(report["latency_summary"])
print(report["performance_summary"])
