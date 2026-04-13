"""
Transcription Service Unit Tests
==================================
Tests inference, worker, and Redis client logic.

Run:
  pytest Test.py -v
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import app.Worker
import app.Redis_client


# =============================================================================
# Inference tests
# =============================================================================

class TestTranscribe:
    def test_transcribe_returns_text_and_segments(self):
        import torch
        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_waveform = torch.randn(1, 16000 * 5)

        mock_inputs = MagicMock()
        mock_inputs.input_features = torch.randn(1, 80, 3000)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_output = torch.tensor([[50258, 50259, 1234, 5678, 50257]])
        mock_model.generate.return_value = mock_output

        mock_processor.batch_decode.return_value = ["كيف حالك"]

        with patch("app.Inference.model", mock_model), \
             patch("app.Inference.processor", mock_processor), \
             patch("app.Inference.device", torch.device("cpu")), \
             patch("torchaudio.load", return_value=(mock_waveform, 16000)):
            from app.Inference import transcribe
            result = transcribe("chunk.wav", dialect="lebanese")

            assert result["text"] == "كيف حالك"
            assert result["dialect"] == "lebanese"
            assert len(result["segments"]) > 0

    def test_transcribe_raises_if_model_not_loaded(self):
        with patch("app.Inference.model", None):
            from app.Inference import transcribe
            with pytest.raises(RuntimeError, match="Model not loaded"):
                transcribe("chunk.wav")


class TestSaveResult:
    def test_saves_json_file(self):
        result = {
            "text": "كيف حالك",
            "segments": [{"start": 0.0, "end": 1.5, "text": "كيف حالك"}],
            "confidence": 0.92,
            "dialect": "lebanese",
        }

        with patch("builtins.open", MagicMock()) as mock_open, \
             patch("os.makedirs"):
            from app.Inference import save_result
            path = save_result(result, "results/j_123/chunk_0000.json")
            assert path == "results/j_123/chunk_0000.json"
            mock_open.assert_called_once()


class TestLoadModel:
    def test_loads_from_merged_path(self):
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=1000))]
        mock_processor = MagicMock()

        with patch("app.Inference.os.path.exists", return_value=True), \
             patch("app.Inference.WhisperForConditionalGeneration.from_pretrained", return_value=mock_model), \
             patch("app.Inference.WhisperProcessor.from_pretrained", return_value=mock_processor), \
             patch("app.Inference.torch.device", return_value=MagicMock()), \
             patch("app.Inference.torch.cuda.is_available", return_value=False):
            from app.Inference import load_model
            load_model()

    def test_falls_back_to_base_model(self):
        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(numel=MagicMock(return_value=1000))]
        mock_processor = MagicMock()

        with patch("app.Inference.os.path.exists", return_value=False), \
             patch("app.Inference.WhisperForConditionalGeneration.from_pretrained", return_value=mock_model) as mock_load, \
             patch("app.Inference.WhisperProcessor.from_pretrained", return_value=mock_processor), \
             patch("app.Inference.torch.device", return_value=MagicMock()), \
             patch("app.Inference.torch.cuda.is_available", return_value=False):
            from app.Inference import load_model
            load_model()
            call_args = mock_load.call_args[0][0]
            assert "whisper-small" in call_args


# =============================================================================
# Worker tests
# =============================================================================

@pytest.mark.asyncio
class TestTranscriptionWorker:
    async def test_successful_transcription(self):
        mock_push = AsyncMock()
        mock_result = {
            "text": "كيف حالك يا حبيبي",
            "segments": [{"start": 0.0, "end": 2.5, "text": "كيف حالك يا حبيبي"}],
            "confidence": 0.92,
            "dialect": "lebanese",
        }

        with patch("app.Worker.transcribe", return_value=mock_result), \
             patch("app.Worker.save_result", return_value="results/j_123/chunk_0005.json"), \
             patch("app.Worker.rc.push_completed", mock_push):
            from app.Worker import process_task
            await process_task({
                "task_id": "t_042",
                "job_id": "j_123",
                "chunk_path": "chunks/j_123/chunk_0005.wav",
                "dialect": "lebanese",
                "chunk_index": 5,
            })

            mock_push.assert_called_once()
            call_args = mock_push.call_args[0][0]
            assert call_args["status"] == "completed"
            assert call_args["type"] == "transcribe"
            assert call_args["job_id"] == "j_123"
            assert call_args["chunk_index"] == 5
            assert "كيف حالك" in call_args["text_preview"]

    async def test_failure_pushes_error(self):
        mock_push = AsyncMock()

        with patch("app.Worker.transcribe", side_effect=RuntimeError("CUDA OOM")), \
             patch("app.Worker.rc.push_completed", mock_push):
            from app.Worker import process_task
            await process_task({
                "task_id": "t_042",
                "job_id": "j_123",
                "chunk_path": "chunks/j_123/chunk_0005.wav",
                "chunk_index": 5,
            })

            mock_push.assert_called_once()
            call_args = mock_push.call_args[0][0]
            assert call_args["status"] == "failed"
            assert "CUDA OOM" in call_args["error"]
            assert call_args["chunk_index"] == 5

    async def test_default_dialect_is_auto(self):
        mock_push = AsyncMock()
        mock_transcribe = MagicMock(return_value={
            "text": "test", "segments": [], "confidence": 0.9, "dialect": "auto",
        })

        with patch("app.Worker.transcribe", mock_transcribe), \
             patch("app.Worker.save_result", return_value="results/j_123/chunk_0000.json"), \
             patch("app.Worker.rc.push_completed", mock_push):
            from app.Worker import process_task
            await process_task({
                "task_id": "t_001",
                "job_id": "j_123",
                "chunk_path": "chunks/j_123/chunk_0000.wav",
                "chunk_index": 0,
            })

            mock_transcribe.assert_called_once_with(
                "chunks/j_123/chunk_0000.wav", dialect="auto"
            )


# =============================================================================
# Redis client tests
# =============================================================================

@pytest.mark.asyncio
class TestRedisClient:
    async def test_pop_transcribe_task(self):
        mock_client = AsyncMock()
        mock_client.brpop.return_value = (
            "queue:transcribe",
            json.dumps({
                "task_id": "t_042",
                "job_id": "j_123",
                "chunk_path": "chunks/j_123/chunk_0005.wav",
                "chunk_index": 5,
            })
        )

        with patch("app.Redis_client.client", mock_client):
            from app.Redis_client import pop_transcribe_task
            result = await pop_transcribe_task()
            assert result["task_id"] == "t_042"
            assert result["chunk_index"] == 5

    async def test_pop_returns_none_on_timeout(self):
        mock_client = AsyncMock()
        mock_client.brpop.return_value = None

        with patch("app.Redis_client.client", mock_client):
            from app.Redis_client import pop_transcribe_task
            result = await pop_transcribe_task()
            assert result is None

    async def test_push_completed(self):
        mock_client = AsyncMock()

        with patch("app.Redis_client.client", mock_client):
            from app.Redis_client import push_completed
            await push_completed({
                "task_id": "t_042",
                "job_id": "j_123",
                "type": "transcribe",
                "status": "completed",
            })
            mock_client.lpush.assert_called_once()
            pushed_data = mock_client.lpush.call_args[0][1]
            parsed = json.loads(pushed_data)
            assert parsed["task_id"] == "t_042"