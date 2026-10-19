"""Where a provider processes prompts, derived from its configuration.

Each provider module exposes `residency(config) -> str`, collected in
`providers.RESIDENCY`. The answer is one of "eu", "us", "global" (the vendor
may route anywhere) or "unknown" (nothing in the configuration says). It is
what the *configuration* selects, not an audit of the vendor: an endpoint the
library can't place is "unknown", never a guess.

`GatewayConfig.require_residency` (routing.py) refuses every provider whose
residency differs from the required value, so "unknown" is never allowed.
"""

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .config import GatewayConfig

EU = "eu"
US = "us"
GLOBAL = "global"
UNKNOWN = "unknown"

# Azure regions inside the EU / EEA, for `azure_region` and for endpoint hosts
# that carry the region (e.g. <resource>.swedencentral.api.cognitive.microsoft.com).
AZURE_EU_REGIONS = frozenset(
    {
        "francecentral",
        "francesouth",
        "germanywestcentral",
        "germanynorth",
        "italynorth",
        "northeurope",
        "norwayeast",
        "norwaywest",
        "polandcentral",
        "spaincentral",
        "swedencentral",
        "switzerlandnorth",
        "switzerlandwest",
        "westeurope",
    }
)


def vertex_location_residency(location: str) -> str:
    """Residency of a Vertex AI location: the `eu`/`us` multi-regions, a
    europe-*/us-* region, or `global`. Any other region is "unknown"."""
    value = (location or "").strip().lower()
    if value == "eu" or value.startswith("europe-"):
        return EU
    if value == "us" or value.startswith("us-"):
        return US
    if value == "global":
        return GLOBAL
    return UNKNOWN


def host_residency(url: str, *, eu_hosts: frozenset[str] = frozenset()) -> str:
    """ "eu" when the URL's host is one of `eu_hosts`, else "unknown"."""
    host = (urlparse(url).hostname or "").lower()
    return EU if host in eu_hosts else UNKNOWN


def azure_residency(*, data_zone: str, region: str, endpoint: str) -> str:
    """The operator-declared data zone wins, then an EU region name
    (`azure_region`, or a label of the endpoint host), else "unknown"."""
    zone = data_zone.strip().lower()
    if zone:
        return zone if zone in (EU, US, GLOBAL) else UNKNOWN
    if region.strip().lower() in AZURE_EU_REGIONS:
        return EU
    labels = (urlparse(endpoint).hostname or "").lower().split(".")
    if any(label in AZURE_EU_REGIONS for label in labels):
        return EU
    return UNKNOWN


def residency_report(config: "GatewayConfig", *, all_providers: bool = False) -> dict[str, str]:
    """`{provider: residency}` for the providers in `config.provider_order`, in
    order (an id the gateway doesn't know is "unknown"), or for every known
    provider with `all_providers=True`. Reads the configuration only."""
    from .providers import RESIDENCY

    names = list(RESIDENCY) if all_providers else config.provider_order_list
    report: dict[str, str] = {}
    for name in names:
        resolver = RESIDENCY.get(name)
        report[name] = resolver(config) if resolver else UNKNOWN
    return report
