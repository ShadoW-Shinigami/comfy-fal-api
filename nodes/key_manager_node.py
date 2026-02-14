import json
import math

from aiohttp import web
from server import PromptServer

from .fal_utils import FalConfig


# ── REST routes ──────────────────────────────────────────────────────────────

@PromptServer.instance.routes.post("/fal-api/set-key")
async def set_key_route(request):
    """Receive a key + name from the frontend and update the singleton."""
    try:
        data = await request.json()
        key = data.get("key", "").strip()
        name = data.get("name", "").strip()
        if not key:
            return web.json_response({"error": "No key provided"}, status=400)
        FalConfig().set_key(key, name or None)
        return web.json_response({"status": "ok", "active_key_name": FalConfig().get_key_name()})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/fal-api/active-key-info")
async def active_key_info_route(request):
    """Return the display name of the active key (never the key itself)."""
    return web.json_response({"active_key_name": FalConfig().get_key_name()})


# ── ComfyUI Node ─────────────────────────────────────────────────────────────

class FalApiKeyManager:
    """Visual node for switching between named FAL API keys.

    Keys are stored in browser localStorage — never serialised into the workflow.
    The frontend JS extension POSTs the selected key to the server before each
    queue execution, so all downstream nodes automatically use the right key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "selected_key_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FAL/Utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return math.nan

    def run(self, selected_key_name=""):
        active = FalConfig().get_key_name()
        PromptServer.instance.send_sync(
            "fal-key-status",
            {"active_key_name": active},
        )
        return {}


NODE_CLASS_MAPPINGS = {
    "FalApiKeyManager": FalApiKeyManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalApiKeyManager": "FAL API Key Manager",
}
