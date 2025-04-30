# This appears to be necessary inside the AWS Cloud9 environment.

import ssl

import aiohttp.client_reqrep
import aiohttp.connector
import certifi

original_get_default_ssl_context = aiohttp.connector.TCPConnector._get_ssl_context

def patched_get_default_ssl_context(self: aiohttp.connector.TCPConnector,
                                    req: aiohttp.client_reqrep.ClientRequest) -> ssl.SSLContext | None:
    context = original_get_default_ssl_context(self, req)
    if not context:
        return context

    context.load_verify_locations(cafile=certifi.where())

    return context

aiohttp.connector.TCPConnector._get_ssl_context = patched_get_default_ssl_context  # type: ignore
