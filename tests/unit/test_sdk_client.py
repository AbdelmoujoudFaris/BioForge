"""SDK tests use httpx's MockTransport (no real network) to verify the
client builds requests correctly and parses responses into schema objects.
"""
import httpx

from bioforge.common.schemas import ADMETProfile
from bioforge.sdk.client import BioForgeClient


def _client_with_transport(handler) -> BioForgeClient:
    client = BioForgeClient(base_url="http://testserver")
    client._client = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    return client


def test_admet_profile_round_trip():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admet/profile"
        body = ADMETProfile(candidate_id="c1", method="rule_based_v1", off_target_hits=[]).model_dump(mode="json")
        return httpx.Response(200, json=body)

    with _client_with_transport(handler) as client:
        profile = client.admet_profile("c1", "CCO")
        assert profile.candidate_id == "c1"
        assert profile.method == "rule_based_v1"


def test_client_raises_on_http_error():
    import pytest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad smiles"})

    with _client_with_transport(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.admet_profile("c1", "not a smiles")
