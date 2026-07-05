from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Identity:
    global_id:  UUID                    = field(default_factory=dict)
    label:      Optional[str]           = None
    metadata:   Dict[str, Any]          = field(default_factory=dict)
    max_embeddings: int                 = 10     # cap per identity
