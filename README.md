```python 
class Conflict: 
    score: float 
    reason: str 

class Event: 
    id: int 
    start: datetime 
    end: datetime
    summary: str 
    location: str 
    conflict: Conflict | None = None
``` 