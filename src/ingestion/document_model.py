from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class LegalDocument:
    celex: Optional[str]
    language: str
    title: Optional[str]
    document_type: Optional[str]
    date_document: Optional[str]
    date_effect: Optional[str]
    eurovoc: List[str] = field(default_factory=list)

    recitals: List[str] = field(default_factory=list)
    articles: List[Dict] = field(default_factory=list)
    annexes: List[Dict] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)
