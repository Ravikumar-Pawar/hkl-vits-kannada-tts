import os

import torch
from transformers import VitsModel, AutoTokenizer

from app.config.hkl_config import HKLConfig
from app.config.logger import setup_logger
from app.pipelines.training import HKLTrainerModel
from app.services.kannada_text_normalizer import KannadaTextNormalizer
logger = setup_logger(name=__name__)


class HKLVITSInference:
    def __init__(self, model, tokenizer, config: HKLConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.get_device()
        self.normalizer = KannadaTextNormalizer(config)
        logger.info("HKLVITSInference initialized on device: %s", self.device)

    @classmethod
    def load(cls):
        config = HKLConfig()
        device = config.get_device()
        logger.info("Loading HKL VITS model on device: %s", device)
        try:
            logger.info("Loading tokenizer: %s", config.mms_model_name)
            tokenizer = AutoTokenizer.from_pretrained(config.mms_model_name)
            logger.info("Loading MMS model")
            mms = VitsModel.from_pretrained(config.mms_model_name).to(device)
            logger.info("Wrapping MMS model with HKLTrainerModel")
            model = HKLTrainerModel(mms).to(device)
            if os.path.exists(config.checkpoint_path):
                logger.info("Loading checkpoint: %s", config.checkpoint_path)
                checkpoint = torch.load(config.checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model"])
                logger.info("Checkpoint loaded successfully")
            else:
                logger.error("Checkpoint missing: %s", config.checkpoint_path)
                raise FileNotFoundError(config.checkpoint_path)
            model.eval()
            logger.info("Model ready for inference")
            return cls(model, tokenizer, config)
        except Exception as e:
            logger.exception("Failed loading HKLVITS model: %s", str(e))
            raise

    def synthesize(self, text: str):
        try:
            if self.config.normalize_before_inference:
                text = self.normalizer.normalize(text)
            if self.config.verbose:
                logger.info(f"Normalized text: {text}")
            inputs = self.tokenizer(
                text, return_tensors=self.config.tokenizer_return_tensors
            ).to(self.device)
            logger.debug("Tokenization completed")
            with torch.no_grad():
                output = self.model.mms(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                )

                wav = output.waveform.squeeze().cpu().numpy()
            logger.info("Synthesis completed. Samples: %s", len(wav))
            return wav
        except Exception as e:
            logger.exception("Synthesis failed: %s", str(e))
            raise
