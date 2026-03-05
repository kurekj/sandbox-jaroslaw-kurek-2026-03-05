import asyncio
import os
import time
from typing import Optional

import httpx
import pandas as pd

from src.v2.api.services.load_leads_df import load_leads_data_db
from src.v2.autoencoder.preprocess_data import load_current_properties_data


async def send_dataframe_for_scoring(
    df: pd.DataFrame,
    api_url: str = "http://0.0.0.0:8000/calculate_scores",
    api_key: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout: int = 300,  # 5 minutes timeout by default
) -> pd.DataFrame:
    """
    Send a pandas DataFrame to the API for scoring and receive scored DataFrame back.
    Supports both synchronous and asynchronous usage.

    Args:
        df (pandas.DataFrame): The input DataFrame to send for scoring
        api_url (str): The API endpoint URL
        api_key (Optional[str]): The API key for authentication. If None, will try to get from environment.
        client (Optional[httpx.AsyncClient]): An existing httpx client for async requests
        timeout (int): Request timeout in seconds (default: 300s = 5min)

    Returns:
        pandas.DataFrame: The scored DataFrame returned by the API
    """
    # Convert DataFrame to request format matching ScoresRequest model
    request_data = {"data": df.to_dict(orient="records")}

    # Get API key from argument or environment variable
    if api_key is None:
        api_key = os.environ.get("API_KEY", "<your-secret-api-key>")

    # Prepare headers with API key and content type
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Check if we're in async mode (client provided)
    if client is not None:
        # Async request
        response = await client.post(api_url, json=request_data, headers=headers, timeout=timeout)

        if response.status_code == 200:
            response_json = response.json()
            print(f"Metadata: {response_json.get('metadata', {})}")
            result_df = pd.DataFrame(response_json["scores"])
            return result_df
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    else:
        # Synchronous request using httpx
        with httpx.Client(timeout=timeout) as sync_client:
            response = sync_client.post(api_url, json=request_data, headers=headers)

            if response.status_code == 200:
                response_json = response.json()
                print(f"Metadata: {response_json.get('metadata', {})}")
                result_df = pd.DataFrame(response_json["scores"])
                return result_df
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


async def load_data_for_sampling() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load user and property data once to be used for multiple samplings.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (user_ids_df, property_ids_df)
    """
    print("Loading data for sampling...")

    # Load leads data
    leads_df = await load_leads_data_db()
    leads_df = leads_df[leads_df["property_id"].notna()]
    leads_df = leads_df[leads_df["algolytics_uuid"].notna()]

    # Load property data
    raw_visible_apartments_df = await load_current_properties_data()
    raw_visible_apartments_df = raw_visible_apartments_df[raw_visible_apartments_df["property_id"].notna()]

    # Extract unique user IDs and property IDs
    user_ids_df = pd.DataFrame({"user_id": leads_df["algolytics_uuid"].unique()})
    property_ids_df = pd.DataFrame({"property_id": raw_visible_apartments_df["property_id"].unique()})

    print(f"Loaded {len(user_ids_df)} unique users and {len(property_ids_df)} unique properties")

    return user_ids_df, property_ids_df


def sample_from_loaded_data(user_ids_df: pd.DataFrame, property_ids_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample n users and properties from already loaded data.

    Args:
        user_ids_df (pd.DataFrame): DataFrame containing user IDs
        property_ids_df (pd.DataFrame): DataFrame containing property IDs
        n (int): Number of users and properties to sample

    Returns:
        pd.DataFrame: DataFrame with user_id and property_id columns
    """
    # Sample users and properties
    sampled_users = user_ids_df.sample(min(n, len(user_ids_df)))
    sampled_properties = property_ids_df.sample(min(n, len(property_ids_df)))

    return pd.DataFrame(
        {"user_id": sampled_users["user_id"].values, "property_id": sampled_properties["property_id"].values}
    )


async def run_api_tests(
    user_ids_df: pd.DataFrame,
    property_ids_df: pd.DataFrame,
    num_tests: int,
    sample_size: int,
    api_url: str,
    api_key: str,
) -> None:
    """
    Run multiple tests against the API, sampling different users and properties each time.

    Args:
        user_ids_df (pd.DataFrame): DataFrame containing user IDs
        property_ids_df (pd.DataFrame): DataFrame containing property IDs
        num_tests (int): Number of test iterations to run
        sample_size (int): Number of users and properties to sample in each test
        api_url (str): The API endpoint URL
        api_key (str): The API key for authentication
    """
    print(f"Starting {num_tests} API tests with sample size of {sample_size}")

    for i in range(num_tests):
        print(f"\n--- Test {i + 1}/{num_tests} ---")

        # Sample from already loaded data
        test_df = sample_from_loaded_data(user_ids_df, property_ids_df, sample_size)

        print(f"Sample {i + 1}: {len(test_df)} rows")

        try:
            # Send to API and get results
            start_time = pd.Timestamp.now()
            result_df = await send_dataframe_for_scoring(test_df, api_url=api_url, api_key=api_key)
            end_time = pd.Timestamp.now()

            duration = (end_time - start_time).total_seconds()
            print(f"API call successful: {len(result_df)} results in {duration:.2f} seconds")

            # Print sample results
            if len(result_df) > 0:
                print("\nSample results:")
                print(result_df.head(3))

        except Exception as e:
            print(f"Error during API call: {str(e)}")


async def run_batch_api_test(
    user_ids_df: pd.DataFrame,
    property_ids_df: pd.DataFrame,
    batch_size: int,
    sample_size: int,
    api_url: str,
    api_key: str,
) -> list[pd.DataFrame]:
    """
    Run a single batch test with multiple parallel API calls.

    Args:
        user_ids_df (pd.DataFrame): DataFrame containing user IDs
        property_ids_df (pd.DataFrame): DataFrame containing property IDs
        batch_size (int): Number of parallel requests to send
        sample_size (int): Number of users and properties to sample in each request
        api_url (str): The API endpoint URL
        api_key (str): The API key for authentication

    Returns:
        list[pd.DataFrame]: List of result DataFrames
    """
    # Create multiple samples
    test_dfs = [sample_from_loaded_data(user_ids_df, property_ids_df, sample_size) for _ in range(batch_size)]

    # Create a shared client for all requests
    async with httpx.AsyncClient() as client:
        # Send all requests in parallel
        tasks = [send_dataframe_for_scoring(df, api_url=api_url, api_key=api_key, client=client) for df in test_dfs]

        return await asyncio.gather(*tasks, return_exceptions=True)  # type: ignore


async def run_batched_api_tests(
    user_ids_df: pd.DataFrame,
    property_ids_df: pd.DataFrame,
    num_tests: int = 5,
    batch_size: int = 5,
    sample_size: int = 20,
    api_url: str = "http://0.0.0.0:8000/calculate_scores",
    api_key: str = "test-key",
) -> None:
    """
    Run multiple batches of API tests, with each batch containing multiple parallel requests.

    Args:
        user_ids_df (pd.DataFrame): DataFrame containing user IDs
        property_ids_df (pd.DataFrame): DataFrame containing property IDs
        num_tests (int): Number of test batches to run
        batch_size (int): Number of parallel requests in each batch
        sample_size (int): Number of users and properties to sample in each request
        api_url (str): The API endpoint URL
        api_key (str): The API key for authentication
    """
    print(
        f"Starting {num_tests} API test batches, each with {batch_size} parallel requests (sample size: {sample_size})"
    )

    total_requests = 0
    total_successful = 0
    total_failed = 0
    total_duration = 0.

    for i in range(num_tests):
        print(f"\n--- Batch {i + 1}/{num_tests} ---")
        print(f"Sending {batch_size} parallel requests with {sample_size} samples each")

        # Time the batch
        start_time = time.time()
        results = await run_batch_api_test(user_ids_df, property_ids_df, batch_size, sample_size, api_url, api_key)
        end_time = time.time()

        batch_duration = end_time - start_time
        total_duration += batch_duration

        # Count successful and failed requests
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))

        total_requests += len(results)
        total_successful += successful
        total_failed += failed

        # Print batch summary
        print(f"Batch completed in {batch_duration:.2f} seconds")
        print(f"Successful requests: {successful}/{batch_size}")

        # Print sample results from first successful result
        for idx, result in enumerate(results):
            if not isinstance(result, Exception):
                print(f"\nSample result from request {idx + 1}:")
                print(result.head(2))
                break

        # Print any errors
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"\nError in request {idx + 1}: {str(result)}")

    # Print overall summary
    print("\n=== Summary ===")
    print(f"Total requests: {total_requests}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average batch duration: {total_duration / num_tests:.2f} seconds")
    if total_successful > 0:
        print(f"Average request duration: {total_duration / total_successful:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run API tests for scoring users and properties")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "loop"],
        default="single",
        help="Test mode: single test or loop through multiple tests",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=3,
        help="Number of test iterations to run when in loop mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel requests in each batch (default: 1 = no batching)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of users and properties to sample in each test",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://0.0.0.0:8000/calculate_scores",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-key",
        help="API key for authentication",
    )

    args = parser.parse_args()

    # Load data once for all test modes
    user_ids_df, property_ids_df = asyncio.run(load_data_for_sampling())

    # Determine if we're in batch mode
    is_batch = args.batch_size > 1

    if args.mode == "loop":
        if is_batch:
            # Run batched API tests with parallel requests in a loop
            asyncio.run(
                run_batched_api_tests(
                    user_ids_df,
                    property_ids_df,
                    num_tests=args.num_tests,
                    batch_size=args.batch_size,
                    sample_size=args.sample_size,
                    api_url=args.api_url,
                    api_key=args.api_key,
                )
            )
        else:
            # Run sequential API tests in a loop
            asyncio.run(
                run_api_tests(
                    user_ids_df,
                    property_ids_df,
                    num_tests=args.num_tests,
                    sample_size=args.sample_size,
                    api_url=args.api_url,
                    api_key=args.api_key
                )
            )
    else:
        # Single mode
        if is_batch:
            # Run a single batch of parallel requests
            print(f"Running a single batch of {args.batch_size} parallel requests with {args.sample_size} samples each")

            results = asyncio.run(
                run_batch_api_test(
                    user_ids_df, property_ids_df, args.batch_size, args.sample_size, args.api_url, args.api_key
                )
            )

            # Print results summary
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            print(f"Successfully completed {successful}/{args.batch_size} requests")

            # Print sample results
            for idx, result in enumerate(results):
                if not isinstance(result, Exception):
                    print(f"\nSample result from request {idx + 1}:")
                    print(result.head(5))
                    break

            # Print any errors
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"\nError in request {idx + 1}: {str(result)}")
        else:
            # Run a single test (original behavior)
            test_df = sample_from_loaded_data(user_ids_df, property_ids_df, n=args.sample_size)

            print("Sending DataFrame to API:")
            print(test_df.head(10))
            print(f"Total rows: {len(test_df)}")
            print("\n")

            # Send to API and get results
            try:
                result_df = asyncio.run(send_dataframe_for_scoring(test_df, api_url=args.api_url, api_key=args.api_key))

                print("Received DataFrame from API:")
                print(result_df.head(10))
                print(f"Total rows: {len(result_df)}")
            except Exception as e:
                print(f"Error: {str(e)}")
