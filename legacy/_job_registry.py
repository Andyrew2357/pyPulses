from typing import Dict
import uuid, time

class ThreadJobRegistry():
    _jobs: Dict[str, object] = {}
    _meta: Dict[str, dict] = {}

    @classmethod
    def register(cls, job) -> str:
        jid = str(uuid.uuid4())
        cls._jobs[jid] = job
        cls._meta[jid] = {'created_at': time.time(), 'status': 'running'}
        return jid
    
    @classmethod
    def get(cls, jid: str):
        return cls._jobs.get(jid)
    
    @classmethod 
    def update_status(cls, jid: str, status: str):
        if jid in cls._meta:
            cls._meta[jid]['status'] = status

    @classmethod
    def cleanup(cls, jid: str):
        cls._jobs.pop(jid, None)
        cls._meta.pop(jid, None)

    @classmethod
    def list(cls):
        return {jid: {**meta, 'alive': job.is_alive() if hasattr(job, 'is_alive') else None}
                for jid, (job, meta) in zip(cls._jobs.keys(), zip(cls._jobs.values(), cls._meta.values()))}
