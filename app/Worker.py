"""
Transcription Worker (S3 version)
Pop task → download full audio from S3 → run Whisper pipeline → upload result to S3 → push completion.
Processes the full audio in one pass using the HuggingFace pipeline with internal 30s windowing.
"""
import os
import json
import tempfile
from app.Config import config
from app import Redis_client as rc
from app import S3_client as s3
from app.Inference import transcribe, save_result


async def process_task(message: dict):
    task_id = message["task_id"]
    job_id = message["job_id"]
    audio_s3_key = message["audio_path"]
    video_meta_path = message.get("video_meta_path")
    dialect = message.get("dialect", "auto")

    print(f"  [TRANSCRIBE] Job {job_id}: {audio_s3_key}")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Download full audio from S3
            local_audio = os.path.join(tmp_dir, "full_audio.wav")
            print(f"  [TRANSCRIBE] Downloading audio...")
            s3.download_file(audio_s3_key, local_audio)

            # Step 2: Run Whisper pipeline on full audio
            print(f"  [TRANSCRIBE] Running Whisper inference...")
            result = await transcribe(local_audio, dialect=dialect)

            # Step 3: Save result and upload to S3
            local_result = os.path.join(tmp_dir, "transcript.json")
            save_result(result, local_result)

            result_s3_key = f"results/{job_id}/transcript.json"
            s3.upload_file(local_result, result_s3_key)

            seg_count = len(result.get("segments", []))
            print(f"  [TRANSCRIBE] Job {job_id} done: {seg_count} segments, "
                  f"'{result['text'][:50]}...'")

            # Step 4: Push completion
            await rc.push_completed({
                "task_id": task_id,
                "job_id": job_id,
                "type": "transcribe",
                "status": "completed",
                "output": result_s3_key,
                "text_preview": result["text"][:100],
            })

    except Exception as e:
        print(f"  [TRANSCRIBE] Failed job {job_id}: {e}")
        await rc.push_completed({
            "task_id": task_id,
            "job_id": job_id,
            "type": "transcribe",
            "status": "failed",
            "error": str(e),
        })
