"""Classiq platform integration client"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json
from datetime import datetime
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ClassiqClient:
    """Client for interacting with Classiq quantum platform"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://platform.classiq.io/api/v1"
        self.session = None
        self._backend_info = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()

    async def create_circuit(self, circuit_config: Dict[str, Any]) -> str:
        """Create a quantum circuit using Classiq synthesis"""
        endpoint = f"{self.base_url}/circuits/synthesize"

        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )

            async with self.session.post(endpoint, json=circuit_config) as response:
                if response.status == 200:
                    result = await response.json()
                    circuit_id = result.get("circuit_id")
                    logger.info(f"Created circuit: {circuit_id}")
                    return circuit_id
                else:
                    error = await response.text()
                    raise Exception(f"Circuit creation failed: {error}")

        except Exception as e:
            logger.error(f"Failed to create circuit: {e}")
            raise

    async def execute_circuit(
            self,
            circuit_id: str,
            backend: str = "simulator",
            shots: int = 1024,
            parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Execute a quantum circuit"""
        endpoint = f"{self.base_url}/circuits/{circuit_id}/execute"

        execution_config = {
            "backend": backend,
            "shots": shots,
            "parameters": parameters or {}
        }

        try:
            async with self.session.post(endpoint, json=execution_config) as response:
                if response.status == 200:
                    result = await response.json()
                    job_id = result.get("job_id")

                    # Wait for job completion
                    execution_result = await self._wait_for_job(job_id)
                    return execution_result
                else:
                    error = await response.text()
                    raise Exception(f"Circuit execution failed: {error}")

        except Exception as e:
            logger.error(f"Failed to execute circuit: {e}")
            raise

    async def _wait_for_job(
            self,
            job_id: str,
            timeout: int = 300,
            poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Wait for quantum job completion"""
        endpoint = f"{self.base_url}/jobs/{job_id}"

        start_time = asyncio.get_event_loop().time()

        while True:
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    job_data = await response.json()
                    status = job_data.get("status")

                    if status == "COMPLETED":
                        return job_data.get("result", {})
                    elif status == "FAILED":
                        raise Exception(f"Job failed: {job_data.get('error')}")
                    elif status in ["QUEUED", "RUNNING"]:
                        # Check timeout
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            raise TimeoutError(f"Job {job_id} timed out")

                        await asyncio.sleep(poll_interval)
                    else:
                        raise Exception(f"Unknown job status: {status}")
                else:
                    raise Exception(f"Failed to get job status: {response.status}")

    async def get_backend_status(self) -> Dict[str, Any]:
        """Get quantum backend status"""
        endpoint = f"{self.base_url}/backends/status"

        try:
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unknown", "error": await response.text()}

        except Exception as e:
            logger.error(f"Failed to get backend status: {e}")
            return {"status": "error", "error": str(e)}

    async def optimize_circuit(self, circuit_id: str) -> str:
        """Optimize a quantum circuit"""
        endpoint = f"{self.base_url}/circuits/{circuit_id}/optimize"

        optimization_config = {
            "optimization_level": 2,
            "target_backend": settings.quantum_backend
        }

        try:
            async with self.session.post(endpoint, json=optimization_config) as response:
                if response.status == 200:
                    result = await response.json()
                    optimized_id = result.get("optimized_circuit_id")
                    logger.info(f"Optimized circuit: {circuit_id} -> {optimized_id}")
                    return optimized_id
                else:
                    error = await response.text()
                    logger.warning(f"Circuit optimization failed: {error}")
                    return circuit_id  # Return original if optimization fails

        except Exception as e:
            logger.error(f"Failed to optimize circuit: {e}")
            return circuit_id

    async def estimate_resources(self, circuit_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate quantum resources needed"""
        endpoint = f"{self.base_url}/circuits/estimate"

        try:
            async with self.session.post(endpoint, json=circuit_config) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": await response.text()}

        except Exception as e:
            logger.error(f"Failed to estimate resources: {e}")
            return {"error": str(e)}