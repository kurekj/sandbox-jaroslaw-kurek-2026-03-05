#!/usr/bin/env python
import argparse
import asyncio
import os
from typing import Any, Optional

import httpx


async def call_prefill_endpoint(
    api_url: str = "http://0.0.0.0:8000/prefill_cache",
    api_key: Optional[str] = None,
    overwrite_visible_properties: bool = False,
    overwrite_pois: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Call the API endpoint to trigger cache prefill as a background task.

    Args:
        api_url (str): The API endpoint URL
        api_key (Optional[str]): The API key for authentication. If None, will try to get from environment.
        overwrite_visible_properties (bool): Whether to overwrite existing cached visible property data (default: False)
        overwrite_pois (bool): Whether to overwrite existing POI data (default: False)
        timeout (int): Request timeout in seconds (default: 30s)

    Returns:
        dict: The API response
    """
    # Get API key from argument or environment variable
    if api_key is None:
        api_key = os.environ.get("API_KEY", "<your-secret-api-key>")

    # Prepare headers with API key and content type
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Prepare request body - always send the parameters
    request_data = {
        "overwrite_visible_properties": overwrite_visible_properties,
        "overwrite_pois": overwrite_pois,
    }

    # Make the API call
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=request_data, headers=headers, timeout=timeout)

    if response.status_code == 202:
        print(f"✅ Success: {response.json()}")
        return response.json()  # type: ignore
    else:
        error_msg = f"❌ API request failed with status code {response.status_code}: {response.text}"
        print(error_msg)
        raise Exception(error_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call the prefill cache endpoint")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://0.0.0.0:8000/prefill_cache",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-key",
        help="API key for authentication",
    )
    # Modified to handle as flags with default False
    parser.add_argument(
        "--overwrite-visible-properties",
        action="store_true",
        help="Whether to overwrite existing cached visible property data (flag)",
    )
    parser.add_argument(
        "--overwrite-pois",
        action="store_true",
        help="Whether to overwrite existing POI data (flag)",
    )

    # Set defaults to False
    parser.set_defaults(overwrite_visible_properties=False, overwrite_pois=False)

    args = parser.parse_args()

    try:
        response = asyncio.run(
            call_prefill_endpoint(
                api_url=args.api_url,
                api_key=args.api_key,
                overwrite_visible_properties=args.overwrite_visible_properties,
                overwrite_pois=args.overwrite_pois,
            )
        )
        print("Cache prefill triggered successfully. The process will continue in the background on the server.")
    except Exception as e:
        print(f"Failed to trigger cache prefill: {str(e)}")
