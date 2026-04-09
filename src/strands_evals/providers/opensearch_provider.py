"""OpenSearch trace provider for retrieving agent evaluation data.

Requires the opensearch-genai-observability-sdk-py package:
    pip install opensearch-genai-observability-sdk-py[opensearch]
"""

import logging
from typing import Any

from strands_evals.mappers.opensearch_session_mapper import OpenSearchSessionMapper
from strands_evals.providers.exceptions import ProviderError, SessionNotFoundError
from strands_evals.providers.trace_provider import TraceProvider
from strands_evals.types.evaluation import TaskOutput
from strands_evals.types.trace import AgentInvocationSpan

logger = logging.getLogger(__name__)


class OpenSearchProvider(TraceProvider):
    """Retrieves agent traces from OpenSearch via genai-observability-sdk-py."""

    def __init__(
        self,
        host: str = "https://localhost:9200",
        index: str = "otel-v1-apm-span-*",
        auth: "tuple[str, str] | Any | None" = None,
        verify_certs: bool = True,
    ):
        try:
            from opensearch_genai_observability_sdk_py import OpenSearchTraceRetriever
        except ImportError:
            raise ImportError(
                "opensearch-genai-observability-sdk-py is required. "
                "Install with: pip install opensearch-genai-observability-sdk-py[opensearch]"
            )

        self._retriever = OpenSearchTraceRetriever(
            host=host, index=index, auth=auth, verify_certs=verify_certs
        )
        self._mapper = OpenSearchSessionMapper()

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Retrieve traces for a session and return evaluation data."""
        try:
            session_record = self._retriever.get_traces(session_id)
        except Exception as e:
            raise ProviderError(f"Failed to query OpenSearch: {e}") from e

        if not session_record.traces:
            raise SessionNotFoundError(f"No traces found for session: {session_id}")

        span_records = [span for trace in session_record.traces for span in trace.spans]
        session = self._mapper.map_to_session(span_records, session_id)
        output = self._extract_output(session)

        return TaskOutput(output=output, trajectory=session)

    @staticmethod
    def _extract_output(session):
        """Extract final agent response from the last AgentInvocationSpan."""
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan) and span.agent_response:
                    return span.agent_response
        return ""
