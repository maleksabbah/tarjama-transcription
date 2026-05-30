# transcription: app/Services/BatchWorkerService.py
"""
Batch transcription worker.
Consumes Kafka tasks, downloads audio from S3, transcribes, uploads transcript,
registers with storage, publishes completion. Commits offset on success.
"""
import os
import tempfile

from app.Repositories import (
    EventConsumer,
    EventPublisher,
    S3Client,
    StorageClient,
)
from app.Services.InferenceService import InferenceService


class BatchWorkerService:
    def __init__(
        self,
        consumer: EventConsumer,
        publisher: EventPublisher,
        s3: S3Client,
        storage: StorageClient,
        inference: InferenceService,
    ):
        self.consumer = consumer
        self.publisher = publisher
        self.s3 = s3
        self.storage = storage
        self.inference = inference

    async def run(self) -> None:
        await self.consumer.start()
        print("  [BATCH] Consumer started")
        try:
            async for message in self.consumer.messages():
                try:
                    await self.process(message)
                    await self.consumer.commit()
                except Exception as e:
                    print(f"  [BATCH] Handler error, will redeliver: {e}")
        finally:
            await self.consumer.stop()
            print("  [BATCH] Consumer stopped")

    async def process(self, message: dict) -> None:
        task_id = message["task_id"]
        job_id = message["job_id"]
        user_id = message.get("user_id", 0)
        audio_s3_key = message["audio_path"]

        print(f"  [BATCH] Job {job_id}: {audio_s3_key}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Download audio
            local_audio = os.path.join(tmp_dir, "full_audio.wav")
            self.s3.download_file(audio_s3_key, local_audio)

            # 2. Transcribe
            result = self.inference.transcribe_file(local_audio)

            # 3. Save and upload transcript
            local_result = os.path.join(tmp_dir, "transcript.json")
            self.inference.save_result(result, local_result)
            transcript_key = f"results/{job_id}/transcript.json"
            self.s3.upload_file(local_result, transcript_key)

            # 4. Register with storage (raises on failure → message redelivers)
            await self.storage.register_file(
                job_id=job_id,
                user_id=user_id,
                category="transcript",
                file_type="json",
                path=transcript_key,
                mime_type="application/json",
            )

            seg_count = len(result.get("segments", []))
            print(f"  [BATCH] Job {job_id} done: {seg_count} segments")

            # 5. Publish completion
            await self.publisher.publish_completion({
                "task_id": task_id,
                "job_id": job_id,
                "type": "transcribe",
                "status": "completed",
                "output": transcript_key,
                "text_preview": result["text"][:100],
            })