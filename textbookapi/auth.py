"""API key management and FastAPI authentication dependency."""

import json
import os
import secrets
import logging
from datetime import datetime, timezone

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("textbookapi.auth")
_security = HTTPBearer()


class APIKeyManager:
    def __init__(self, keys_file: str):
        self._keys_file = keys_file
        self._valid_keys: set[str] = set()
        self.reload()

    def reload(self):
        if not os.path.exists(self._keys_file):
            logger.warning(f"No API keys file found. Creating one at {self._keys_file}")
            self._bootstrap()
            return

        with open(self._keys_file) as f:
            data = json.load(f)
        self._valid_keys = set(data.get("keys", {}).keys())
        logger.info(f"Loaded {len(self._valid_keys)} API key(s)")

    def is_valid(self, key: str) -> bool:
        return key in self._valid_keys

    def _bootstrap(self):
        """Create api_keys.json with one auto-generated key."""
        key = f"ujjwal-{secrets.token_hex(24)}"
        data = {
            "keys": {
                key: {
                    "name": "default",
                    "created": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
        with open(self._keys_file, "w") as f:
            json.dump(data, f, indent=2)
        self._valid_keys = {key}
        logger.info(f"Generated default API key: {key}")
        print(f"\n  ** Your API key: {key} **")
        print(f"  ** Saved to: {self._keys_file} **\n")


def require_api_key(manager: APIKeyManager):
    """FastAPI dependency factory that validates Bearer tokens."""
    async def _verify(
        credentials: HTTPAuthorizationCredentials = Depends(_security),
    ) -> str:
        if not manager.is_valid(credentials.credentials):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return credentials.credentials
    return _verify
