"""
Transcription Worker (S3 version)
Pop task → download chunk from S3 → run Whisper → upload result to S3 → push completion.
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
    chunk_s3_key = message["chunk_path"]  # e.g., "chunks/j_123/chunk_0005.wav"
    dialect = message.get("dialect", "auto")
    chunk_index = message.get("chunk_index", 0)

    print(f"  [TRANSCRIBE] Job {job_id} chunk {chunk_index}: {chunk_s3_key}")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Download chunk from S3
            local_chunk = os.path.join(tmp_dir, f"chunk_{chunk_index:04d}.wav")
            s3.download_file(chunk_s3_key, local_chunk)

            # Step 2: Run Whisper inference
            result = transcribe(local_chunk, dialect=dialect)

            # Step 3: Save result locally then upload to S3
            local_result = os.path.join(tmp_dir, f"chunk_{chunk_index:04d}.json")
            save_result(result, local_result)

            result_s3_key = f"results/{job_id}/chunk_{chunk_index:04d}.json"
            s3.upload_file(local_result, result_s3_key)

            print(f"  [TRANSCRIBE] Job {job_id} chunk {chunk_index} done: "
                  f"'{result['text'][:50]}...' conf={result.get('confidence', 'N/A')}")

            # Step 4: Push completion
            await rc.push_completed({
                "task_id": task_id,
                "job_id": job_id,
                "type": "transcribe",
                "status": "completed",
                "output": result_s3_key,
                "chunk_index": chunk_index,
                "text_preview": result["text"][:100],
            })

    except Exception as e:
        print(f"  [TRANSCRIBE] Failed job {job_id} chunk {chunk_index}: {e}")
        await rc.push_completed({
            "task_id": task_id,
            "job_id": job_id,
            "type": "transcribe",
            "status": "failed",
            "error": str(e),
            "chunk_index": chunk_index,
        })