"""Quantum Natural Language Processing model"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio
from sklearn.preprocessing import normalize

from app.quantum.classiq_client import ClassiqClient
from app.quantum.circuit_builder import CircuitBuilder

logger = logging.getLogger(__name__)


@dataclass
class QuantumFeatures:
    """Container for quantum-encoded text features"""
    amplitude_encoding: np.ndarray
    phase_encoding: np.ndarray
    entanglement_pattern: str
    num_qubits: int


class QuantumNLPModel:
    """Quantum-enhanced NLP model for sentiment analysis"""

    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        self.circuit_builder = CircuitBuilder()

        # Model parameters
        self.num_qubits = 8  # Adjust based on available resources
        self.num_layers = 3  # Variational layers
        self.feature_dim = 2 ** self.num_qubits

        # Learned parameters (would be trained in real implementation)
        self._init_variational_parameters()

    def _init_variational_parameters(self):
        """Initialize variational circuit parameters"""
        # Random initialization (in practice, these would be learned)
        np.random.seed(42)
        self.theta = np.random.randn(self.num_layers, self.num_qubits, 3) * 0.1

    async def encode_text(self, text: str) -> QuantumFeatures:
        """Encode text into quantum features"""
        # Convert text to classical features first
        classical_features = self._text_to_classical_features(text)

        # Normalize to unit vector for amplitude encoding
        normalized_features = normalize(
            classical_features.reshape(1, -1), norm='l2'
        )[0]

        # Pad or truncate to match quantum dimension
        if len(normalized_features) < self.feature_dim:
            amplitude_encoding = np.pad(
                normalized_features,
                (0, self.feature_dim - len(normalized_features)),
                mode='constant'
            )
        else:
            amplitude_encoding = normalized_features[:self.feature_dim]

        # Create phase encoding from text structure
        phase_encoding = self._create_phase_encoding(text)

        # Determine entanglement pattern based on text complexity
        entanglement_pattern = self._select_entanglement_pattern(text)

        return QuantumFeatures(
            amplitude_encoding=amplitude_encoding,
            phase_encoding=phase_encoding,
            entanglement_pattern=entanglement_pattern,
            num_qubits=self.num_qubits
        )

    def _text_to_classical_features(self, text: str) -> np.ndarray:
        """Convert text to classical feature vector"""
        # Simple bag-of-words style encoding with enhancements
        # In practice, use pre-trained embeddings

        # Character-level features
        char_features = self._extract_char_features(text)

        # Word-level features
        word_features = self._extract_word_features(text)

        # Syntactic features
        syntax_features = self._extract_syntax_features(text)

        # Combine all features
        features = np.concatenate([
            char_features,
            word_features,
            syntax_features
        ])

        return features

    def _extract_char_features(self, text: str) -> np.ndarray:
        """Extract character-level features"""
        # Simple character frequency histogram
        char_counts = np.zeros(256)  # ASCII

        for char in text:
            if ord(char) < 256:
                char_counts[ord(char)] += 1

        # Normalize
        if char_counts.sum() > 0:
            char_counts /= char_counts.sum()

        return char_counts

    def _extract_word_features(self, text: str) -> np.ndarray:
        """Extract word-level features"""
        words = text.lower().split()

        # Simple features
        features = [
            len(words),  # Word count
            np.mean([len(w) for w in words]) if words else 0,  # Avg word length
            len(set(words)) / max(len(words), 1),  # Vocabulary richness
        ]

        # Sentiment lexicon features
        positive_words = {"good", "great", "excellent", "positive", "up", "gain"}
        negative_words = {"bad", "poor", "negative", "down", "loss", "fall"}

        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)

        features.extend([
            pos_count / max(len(words), 1),
            neg_count / max(len(words), 1)
        ])

        return np.array(features)

    def _extract_syntax_features(self, text: str) -> np.ndarray:
        """Extract syntactic features"""
        features = [
            text.count('!') / max(len(text), 1),  # Exclamation density
            text.count('?') / max(len(text), 1),  # Question density
            text.count('.') / max(len(text), 1),  # Period density
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
        ]

        return np.array(features)

    def _create_phase_encoding(self, text: str) -> np.ndarray:
        """Create phase encoding from text structure"""
        # Map text patterns to phases
        phases = np.zeros(self.num_qubits)

        # Use text hash for deterministic phase assignment
        text_hash = hash(text) % (2 ** 32)

        for i in range(self.num_qubits):
            # Extract bits from hash for phase
            bit_mask = 0xFF << (i * 8 % 32)
            phase_bits = (text_hash & bit_mask) >> (i * 8 % 32)
            phases[i] = (phase_bits / 255.0) * 2 * np.pi

        return phases

    def _select_entanglement_pattern(self, text: str) -> str:
        """Select entanglement pattern based on text complexity"""
        complexity = len(set(text.split())) / max(len(text.split()), 1)

        if complexity < 0.3:
            return "linear"
        elif complexity < 0.6:
            return "circular"
        else:
            return "all-to-all"

    async def classify_sentiment(
            self,
            quantum_features: QuantumFeatures
    ) -> Dict[str, Any]:
        """Classify sentiment using quantum circuit"""
        # Build variational quantum circuit
        circuit_config = self._build_classification_circuit(quantum_features)

        try:
            # Create circuit
            circuit_id = await self.classiq_client.create_circuit(circuit_config)

            # Optimize circuit
            optimized_id = await self.classiq_client.optimize_circuit(circuit_id)

            # Execute circuit
            result = await self.classiq_client.execute_circuit(
                optimized_id,
                backend=settings.quantum_backend,
                shots=2048
            )

            # Process results
            probabilities = self._process_measurement_results(result)

            return {
                "probabilities": probabilities,
                "circuit_depth": result.get("circuit_depth", 0),
                "execution_time": result.get("execution_time", 0),
                "backend": result.get("backend", "unknown")
            }

        except Exception as e:
            logger.error(f"Quantum classification failed: {e}")
            # Fallback to classical-like output
            return {
                "probabilities": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
                "circuit_depth": 0,
                "execution_time": 0,
                "backend": "classical_fallback"
            }

    def _build_classification_circuit(
            self,
            quantum_features: QuantumFeatures
    ) -> Dict[str, Any]:
        """Build quantum classification circuit configuration"""
        return {
            "name": "qnlp_sentiment_classifier",
            "quantum_circuit": {
                "num_qubits": self.num_qubits,
                "layers": [
                    # State preparation layer
                    {
                        "type": "state_preparation",
                        "amplitudes": quantum_features.amplitude_encoding.tolist(),
                        "phases": quantum_features.phase_encoding.tolist()
                    },
                    # Variational layers
                    *[
                        {
                            "type": "variational_layer",
                            "parameters": self.theta[i].tolist(),
                            "entanglement": quantum_features.entanglement_pattern
                        }
                        for i in range(self.num_layers)
                    ],
                    # Measurement layer
                    {
                        "type": "measurement",
                        "qubits": list(range(min(3, self.num_qubits))),  # Measure subset
                        "basis": "computational"
                    }
                ]
            },
            "optimization_level": 2,
            "constraints": {
                "max_depth": 100,
                "max_gates": 500
            }
        }

    def _process_measurement_results(
            self,
            result: Dict[str, Any]
    ) -> np.ndarray:
        """Process quantum measurement results into sentiment probabilities"""
        counts = result.get("counts", {})
        total_shots = sum(counts.values())

        if total_shots == 0:
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Map measurement outcomes to sentiment classes
        # 5 sentiment classes: very negative, negative, neutral, positive, very positive
        sentiment_probs = np.zeros(5)

        for bitstring, count in counts.items():
            # Convert bitstring to sentiment index
            # Simple mapping: use first 3 bits
            if len(bitstring) >= 3:
                index = int(bitstring[:3], 2)
                if index < 5:
                    sentiment_probs[index] += count
                else:
                    # Map higher values to neutral
                    sentiment_probs[2] += count

        # Normalize
        sentiment_probs /= total_shots

        # Apply smoothing to avoid zero probabilities
        sentiment_probs = (sentiment_probs + 0.01) / (1 + 0.05)

        return sentiment_probs