from datetime import datetime

# Global job store
jobs = {}

class Job:
    def __init__(self, job_id, filename):
        self.job_id = job_id
        self.filename = filename
        self.status = 'pending'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.created_at = datetime.now()
        self.output_path = None
        self.error = None
        self.transcript = None
        
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'filename': self.filename,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at.isoformat(),
            'output_path': self.output_path,
            'error': self.error,
            'transcript': self.transcript
        }
