from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import numpy as np

from hardware import (
    build_sample_circuit,
    discover_ionq_devices,
    discover_lrz_backends,
    generic_job_counts,
    hardware_python_status,
    normalized_counts,
    prediction_from_counts,
    simulate_ideal_shots,
    submit_ionq_job,
)
from quantum_pgm import build_naimark_dilation, build_reduced_pgm_measurement


def unit_rows(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    values = rng.normal(size=(rows, columns))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class HardwareHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(77)
        self.X = unit_rows(rng, 12, 2)
        self.y = np.asarray(["A"] * 6 + ["B"] * 6)
        measurement = build_reduced_pgm_measurement(
            self.X,
            self.y,
            copies=1,
            prior_mode="uniform",
            relative_tolerance=1e-11,
        )
        self.dilation = build_naimark_dilation(measurement, exact_qubit_limit=8)

    def test_local_shot_simulation_is_reproducible(self) -> None:
        first = simulate_ideal_shots(
            self.dilation, self.X[0], shots=500, seed=9
        )
        second = simulate_ideal_shots(
            self.dilation, self.X[0], shots=500, seed=9
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 500)

    def test_count_normalization_and_prediction(self) -> None:
        counts = {"0": 75, "1": 25}
        frequencies = normalized_counts(counts, outcome_qubits=1)
        prediction, returned = prediction_from_counts(
            counts, np.asarray(["A", "B"]), outcome_qubits=1
        )
        self.assertEqual(prediction, "A")
        self.assertEqual(frequencies, returned)
        self.assertAlmostEqual(sum(frequencies.values()), 1.0)

    def test_python_compatibility_report_is_well_formed(self) -> None:
        report = hardware_python_status()
        self.assertIn("version", report)
        self.assertIsInstance(report["aws_compatible"], bool)
        self.assertIsInstance(report["lrz_compatible"], bool)

    def test_lrz_discovery_survives_backend_with_unavailable_target(self) -> None:
        class Status:
            operational = True
            status_msg = "ONLINE"
            pending_jobs = 2

        class BrokenBackend:
            name = "MUNIQC-Atoms20"

            @property
            def num_qubits(self):
                raise NotImplementedError(
                    "Target for MUNIQC-Atoms20 is not available."
                )

            def status(self):
                return Status()

        class GoodBackend:
            name = "EQE1"
            num_qubits = 20
            max_shots = 10_000

            def status(self):
                return Status()

        class Adapter:
            online_argument = None

            def __init__(self, *, token):
                self.token = token

            def backends(self, *, online=False):
                Adapter.online_argument = online
                return [BrokenBackend(), GoodBackend()]

        mqss_module = types.ModuleType("mqss")
        adapter_module = types.ModuleType("mqss.qiskit_adapter")
        adapter_module.MQSSQiskitAdapter = Adapter
        mqss_module.qiskit_adapter = adapter_module

        previous_mqss = sys.modules.get("mqss")
        previous_adapter = sys.modules.get("mqss.qiskit_adapter")
        sys.modules["mqss"] = mqss_module
        sys.modules["mqss.qiskit_adapter"] = adapter_module
        try:
            devices = discover_lrz_backends("test-token", online_only=True)
        finally:
            if previous_mqss is None:
                sys.modules.pop("mqss", None)
            else:
                sys.modules["mqss"] = previous_mqss
            if previous_adapter is None:
                sys.modules.pop("mqss.qiskit_adapter", None)
            else:
                sys.modules["mqss.qiskit_adapter"] = previous_adapter

        self.assertTrue(Adapter.online_argument)
        self.assertEqual(
            [device.name for device in devices], ["EQE1", "MUNIQC-Atoms20"]
        )
        eqe1 = devices[0]
        broken = devices[1]
        self.assertEqual(eqe1.qubits, 20)
        self.assertEqual(eqe1.max_shots, 10_000)
        self.assertIsNone(broken.qubits)
        self.assertIn("Target for MUNIQC-Atoms20", broken.notes)

    def test_generic_job_counts_reads_ibm_sampler_v2_register(self) -> None:
        class Register:
            def get_counts(self):
                return {"00": 7, "01": 3}

        class Data:
            m = Register()

        class PublicationResult:
            data = Data()

        class Job:
            def result(self):
                return [PublicationResult()]

        self.assertEqual(
            generic_job_counts(Job(), shots=10), {"00": 7, "01": 3}
        )

    def test_generic_job_counts_converts_cloud_probabilities_to_shots(self) -> None:
        class Job:
            def get_counts(self):
                return {"0": 0.75, "1": 0.25}

        counts = generic_job_counts(Job(), shots=100)
        self.assertEqual(counts, {"0": 75, "1": 25})
        self.assertEqual(sum(counts.values()), 100)

    def test_ionq_discovery_uses_concrete_cloud_catalog(self) -> None:
        class Response:
            def json(self):
                return [
                    {
                        "backend": "simulator",
                        "qubits": 29,
                        "status": "available",
                        "location": "IonQ Cloud",
                    },
                    {
                        "backend": "qpu.forte-1",
                        "qubits": 36,
                        "status": "available",
                        "location": "College Park",
                        "average_queue_time": 3_600,
                    },
                    {
                        "backend": "qpu.aria-1",
                        "qubits": 25,
                        "status": "retired",
                    },
                ]

        class Client:
            api_headers = {"Authorization": "redacted"}

            def make_path(self, *parts):
                return "/".join(parts)

            def get_with_retry(self, path, headers=None):
                self.path = path
                self.headers = headers
                return Response()

        class GenericBackend:
            client = Client()

        class Provider:
            def __init__(self, token=None):
                self.token = token

            def backends(self):
                return [GenericBackend()]

        ionq_module = types.ModuleType("qiskit_ionq")
        ionq_module.IonQProvider = Provider
        previous = sys.modules.get("qiskit_ionq")
        sys.modules["qiskit_ionq"] = ionq_module
        try:
            devices = discover_ionq_devices("test-token", simulators=False)
        finally:
            if previous is None:
                sys.modules.pop("qiskit_ionq", None)
            else:
                sys.modules["qiskit_ionq"] = previous

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].name, "qpu.forte-1")
        self.assertEqual(devices[0].qubits, 36)
        self.assertIn("60 min", devices[0].notes)

    def test_ionq_ideal_simulator_uses_one_provider_shot(self) -> None:
        class Job:
            def job_id(self):
                return "ionq-job"

        class Backend:
            def run(self, circuit, **options):
                self.circuit = circuit
                self.options = options
                return Job()

        backend = Backend()

        class Provider:
            def __init__(self, token=None):
                self.token = token

            def get_backend(self, name):
                self.name = name
                return backend

        ionq_module = types.ModuleType("qiskit_ionq")
        ionq_module.IonQProvider = Provider
        qiskit_module = types.ModuleType("qiskit")
        qiskit_module.transpile = lambda circuit, **kwargs: circuit

        previous_ionq = sys.modules.get("qiskit_ionq")
        previous_qiskit = sys.modules.get("qiskit")
        sys.modules["qiskit_ionq"] = ionq_module
        sys.modules["qiskit"] = qiskit_module
        try:
            _, job_id = submit_ionq_job(
                object(),
                token="test-token",
                backend_name="simulator",
                shots=1_000,
                noise_model="ideal",
            )
        finally:
            if previous_ionq is None:
                sys.modules.pop("qiskit_ionq", None)
            else:
                sys.modules["qiskit_ionq"] = previous_ionq
            if previous_qiskit is None:
                sys.modules.pop("qiskit", None)
            else:
                sys.modules["qiskit"] = previous_qiskit

        self.assertEqual(job_id, "ionq-job")
        self.assertEqual(backend.options["shots"], 1)

    @unittest.skipUnless(
        importlib.util.find_spec("qiskit") is not None,
        "Qiskit non installato nell'ambiente di test",
    )
    def test_hardware_circuit_includes_state_preparation_and_measurement(self) -> None:
        circuit = build_sample_circuit(self.dilation, self.X[0])
        operations = circuit.count_ops()
        self.assertIn("state_preparation", operations)
        self.assertIn("measure", operations)
        self.assertEqual(circuit.num_qubits, self.dilation.resources.total_qubits)


if __name__ == "__main__":
    unittest.main()
