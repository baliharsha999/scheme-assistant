from typing import Literal
from pydantic import BaseModel , Field

# class of pydantic for retrival evlution scores

class RetrievalEvalution(BaseModel):
    """evalutes the relevance scores of the retrieved documents according to the user query"""
    score: float = Field(
        ge=0,
        le = 1,
        description = "retrival qualtiy score between 0 and 1"
    )

    reason : str = Field(
        description= "reason for assigning the retrieval quality score"
    )

# definting the structure of the refined documents
class RefinedDocuments(BaseModel):
    """passage extracted from a retrival document"""

    text: str = Field(
        description="Text of the extracted passage."
    )
    source: str = Field(
        description="original document name or source"
    )

    page: int | None = Field(
        default=None,
        description="Page number in the original document"
    )

    section: str | None = Field(
        default=None,
        description="section heading of the original document"
    )

    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="relevance score assigned to this passage"
    )

# we need to chcek if the documents that are refined and finalized are good for answering the query 

class EvidenceEvalution(BaseModel):
    """detemines wheather the evidance is sufficient for answering the query"""

    sufficient: bool = Field(
        description=(
            "wheather the available evidence sufficiently supports the answering the query"
        )
    )

    reason: str = Field(
        description="explain why the evidence is sufficient or insufficient"
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="information still neede to answer the query"
    )

#format of the final each doc
class Citation(BaseModel):
    """Source information supporting an answer."""

    source: str

    page: int | None = None

    section: str | None = None

    supporting_text: str | None = None

#structure of the final answers
class FinalAnswer(BaseModel):
    """Structured response returned to the user."""

    answer: str

    citations: list[Citation] = Field(
        default_factory=list
    )

    abstained: bool = False

    reason: str | None = None

# query rewriting

class RewrittenQuery(BaseModel):
    """structured output for query rewriting"""

    rewritten_query : str = Field(
        description = "imporives search query for retrieving relevant docs"
    )

    reason : str = Field(
        description = "why the original query needs rewritting"
    )
