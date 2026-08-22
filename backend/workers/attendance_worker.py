"""
workers/attendance_worker.py
-----------------------------
Defines and starts the three background worker threads.

Each worker runs an infinite loop:
    while True:
        job = frame_queue.get()
        process_frame(job)
        frame_queue.task_done()

Workers are daemon threads — they die automatically when the main
process exits, so there's no cleanup needed.

Call start_workers() exactly once from create_app().
"""
import threading
from workers.queue_manager import frame_queue, mark_processing, mark_completed, mark_failed
from workers.recognizer import process_frame

# Number of concurrent worker threads
NUM_WORKERS = 3

_workers_started = False
_start_lock = threading.Lock()


def _worker_loop(worker_name: str) -> None:
    """
    Infinite loop: dequeue a job, process it, repeat.
    Exceptions are caught and logged so workers never crash.
    """
    print(f"[{worker_name}] Started and waiting for jobs...")

    while True:
        # Blocks until a job is available
        job = frame_queue.get()
        mark_processing()

        try:
            process_frame(job, worker_name)
            mark_completed()
            print(f"[{worker_name}] Finished job for session={job.session_id}")

        except Exception as exc:
            mark_failed()
            print(f"[{worker_name}] Job failed — {exc} — continuing...")

        finally:
            # Always signal the queue that this job slot is done
            frame_queue.task_done()


def start_workers() -> None:
    """
    Spawn NUM_WORKERS daemon threads.

    Idempotent — safe to call multiple times (e.g. in testing),
    but threads are only created on the first call.
    """
    global _workers_started

    with _start_lock:
        if _workers_started:
            print("[Workers] Already started — skipping")
            return

        for i in range(1, NUM_WORKERS + 1):
            name = f"Worker-{i}"
            t = threading.Thread(
                target=_worker_loop,
                args=(name,),
                name=name,
                daemon=True,        # die with main process
            )
            t.start()
            print(f"[Workers] {name} launched (thread id={t.ident})")

        _workers_started = True
        print(f"[Workers] {NUM_WORKERS} worker(s) running")
