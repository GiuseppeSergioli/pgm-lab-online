from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import numpy as np

import hardware
from hardware import (
    build_sample_circuit,
    discover_aqt_simulators,
    discover_ibm_fake_backends,
    discover_ionq_devices,
    discover_lrz_backends,
    generic_job_counts,
    hardware_python_status,
    normalized_counts,
    prediction_from_counts,
    simulate_basic_shots,
    simulate_braket_local_shots,
    simulate_ibm_fake_shots,
    simulate_ideal_shots,
    submit_aqt_job,
    submit_ionq_job,
    verify_aws_identity,
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

    def test_aws_identity_accepts_isolated_session_credentials(self) -> None:
        captured = {}

        class STSClient:
            def get_caller_identity(self):
                return {
                    "Account": "123456789012",
                    "Arn": "arn:aws:iam::123456789012:user/test",
                    "UserId": "test-user",
                }

        class BotoSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def client(self, name):
                self.client_name = name
                return STSClient()

        class AwsSession:
            def __init__(self, *, boto_session):
                self.boto_session = boto_session

        boto3_module = types.ModuleType("boto3")
        boto3_module.Session = BotoSession
        braket_module = types.ModuleType("braket")
        braket_aws_module = types.ModuleType("braket.aws")
        braket_aws_module.AwsSession = AwsSession
        braket_module.aws = braket_aws_module

        previous_boto3 = sys.modules.get("boto3")
        previous_braket = sys.modules.get("braket")
        previous_braket_aws = sys.modules.get("braket.aws")
        sys.modules["boto3"] = boto3_module
        sys.modules["braket"] = braket_module
        sys.modules["braket.aws"] = braket_aws_module
        try:
            identity = verify_aws_identity(
                None,
                "eu-west-2",
                access_key_id="temporary-access-key",
                secret_access_key="temporary-secret",
                session_token="temporary-session-token",
            )
        finally:
            for name, previous in (
                ("boto3", previous_boto3),
                ("braket", previous_braket),
                ("braket.aws", previous_braket_aws),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(identity["account"], "123456789012")
        self.assertEqual(identity["region"], "eu-west-2")
        self.assertEqual(captured["aws_access_key_id"], "temporary-access-key")
        self.assertEqual(captured["aws_secret_access_key"], "temporary-secret")
        self.assertEqual(
            captured["aws_session_token"], "temporary-session-token"
        )
        self.assertNotIn("profile_name", captured)

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

    def test_braket_local_simulator_never_uses_aws_credentials(self) -> None:
        captured = {}

        class Result:
            def get_counts(self):
                return {"0": 6, "1": 4}

        class Job:
            def result(self):
                return Result()

        class Backend:
            def __init__(self, *, name):
                captured["name"] = name

            def run(self, circuit, *, shots):
                captured["circuit"] = circuit
                captured["shots"] = shots
                return Job()

        qiskit_module = types.ModuleType("qiskit")
        qiskit_module.transpile = lambda circuit, **kwargs: circuit
        braket_provider = types.ModuleType("qiskit_braket_provider")
        braket_provider.BraketLocalBackend = Backend
        previous_qiskit = sys.modules.get("qiskit")
        previous_braket = sys.modules.get("qiskit_braket_provider")
        sys.modules["qiskit"] = qiskit_module
        sys.modules["qiskit_braket_provider"] = braket_provider
        try:
            counts = simulate_braket_local_shots(
                object(),
                shots=10,
                seed=12,
                backend_name="braket_sv",
            )
        finally:
            for name, previous in (
                ("qiskit", previous_qiskit),
                ("qiskit_braket_provider", previous_braket),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(counts, {"0": 6, "1": 4})
        self.assertEqual(captured["name"], "braket_sv")
        self.assertEqual(captured["shots"], 10)
        self.assertFalse(any(name.startswith("AWS_") for name in captured))

    def test_basic_simulator_preserves_shots_and_seed(self) -> None:
        captured = {}

        class Result:
            def get_counts(self):
                return {"0": 8, "1": 2}

        class Job:
            def result(self):
                return Result()

        class BasicSimulator:
            def run(self, circuit, *, shots, seed_simulator):
                captured.update(
                    circuit=circuit,
                    shots=shots,
                    seed_simulator=seed_simulator,
                )
                return Job()

        qiskit_module = types.ModuleType("qiskit")
        qiskit_module.transpile = lambda circuit, **kwargs: circuit
        providers_module = types.ModuleType("qiskit.providers")
        basic_module = types.ModuleType("qiskit.providers.basic_provider")
        basic_module.BasicSimulator = BasicSimulator
        previous = {
            name: sys.modules.get(name)
            for name in (
                "qiskit",
                "qiskit.providers",
                "qiskit.providers.basic_provider",
            )
        }
        sys.modules["qiskit"] = qiskit_module
        sys.modules["qiskit.providers"] = providers_module
        sys.modules["qiskit.providers.basic_provider"] = basic_module
        try:
            counts = simulate_basic_shots(
                object(), shots=10, seed=41, optimization_level=0
            )
        finally:
            for name, old_module in previous.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        self.assertEqual(counts, {"0": 8, "1": 2})
        self.assertEqual(captured["shots"], 10)
        self.assertEqual(captured["seed_simulator"], 41)

    def test_ibm_fake_simulator_uses_selected_snapshot_and_seed(self) -> None:
        captured = {}

        class FakeBackend:
            name = "fake_test"

        class Result:
            def get_counts(self):
                return {"00": 5, "01": 7}

        class RunResult:
            def result(self):
                return Result()

        class AerSimulator:
            @classmethod
            def from_backend(cls, backend):
                captured["snapshot"] = backend.name
                return cls()

            def run(self, circuit, *, shots, seed_simulator):
                captured.update(
                    circuit=circuit,
                    shots=shots,
                    seed_simulator=seed_simulator,
                )
                return RunResult()

        qiskit_module = types.ModuleType("qiskit")
        qiskit_module.transpile = lambda circuit, **kwargs: circuit
        aer_module = types.ModuleType("qiskit_aer")
        aer_module.AerSimulator = AerSimulator
        previous_qiskit = sys.modules.get("qiskit")
        previous_aer = sys.modules.get("qiskit_aer")
        previous_catalog = hardware._ibm_fake_backend_catalog
        sys.modules["qiskit"] = qiskit_module
        sys.modules["qiskit_aer"] = aer_module
        hardware._ibm_fake_backend_catalog = lambda: (FakeBackend(),)
        try:
            counts = simulate_ibm_fake_shots(
                object(),
                backend_name="fake_test",
                shots=12,
                seed=73,
                optimization_level=1,
            )
        finally:
            hardware._ibm_fake_backend_catalog = previous_catalog
            for name, old_module in (
                ("qiskit", previous_qiskit),
                ("qiskit_aer", previous_aer),
            ):
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

        self.assertEqual(counts, {"00": 5, "01": 7})
        self.assertEqual(captured["snapshot"], "fake_test")
        self.assertEqual(captured["shots"], 12)
        self.assertEqual(captured["seed_simulator"], 73)

    def test_aqt_discovery_separates_offline_and_cloud_simulators(self) -> None:
        class Configuration:
            simulator = True
            n_qubits = 20
            max_shots = 2_000

        class Backend:
            def __init__(self, name):
                self.name = name
                self.resource_id = name

            def configuration(self):
                return Configuration()

            def status(self):
                return True

        class Provider:
            def __init__(self, token=None):
                self.token = token

            def backends(self):
                return [
                    Backend("offline_simulator_no_noise"),
                    Backend("offline_simulator_noise"),
                    Backend("simulator_cloud"),
                ]

        root = types.ModuleType("qiskit_aqt_provider")
        provider_module = types.ModuleType("qiskit_aqt_provider.aqt_provider")
        provider_module.AQTProvider = Provider
        root.aqt_provider = provider_module
        previous_root = sys.modules.get("qiskit_aqt_provider")
        previous_provider = sys.modules.get("qiskit_aqt_provider.aqt_provider")
        sys.modules["qiskit_aqt_provider"] = root
        sys.modules["qiskit_aqt_provider.aqt_provider"] = provider_module
        try:
            offline = discover_aqt_simulators(None, offline=True)
            cloud = discover_aqt_simulators("token", offline=False)
        finally:
            for name, previous in (
                ("qiskit_aqt_provider", previous_root),
                ("qiskit_aqt_provider.aqt_provider", previous_provider),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(len(offline), 2)
        self.assertTrue(all(device.provider == "AQT Offline" for device in offline))
        self.assertEqual([device.name for device in cloud], ["simulator_cloud"])

    def test_aqt_submission_uses_selected_backend_and_shots(self) -> None:
        captured = {}

        class Job:
            def job_id(self):
                return "aqt-job"

        class Backend:
            def run(self, circuit, *, shots):
                captured["circuit"] = circuit
                captured["shots"] = shots
                return Job()

        backend = Backend()

        class Provider:
            def __init__(self, token=None):
                captured["token"] = token

            def get_backend(self, name=None, **kwargs):
                captured["backend"] = name
                captured["workspace"] = kwargs.get("workspace")
                return backend

        root = types.ModuleType("qiskit_aqt_provider")
        provider_module = types.ModuleType("qiskit_aqt_provider.aqt_provider")
        provider_module.AQTProvider = Provider
        root.aqt_provider = provider_module
        qiskit_module = types.ModuleType("qiskit")
        qiskit_module.transpile = lambda circuit, **kwargs: circuit
        previous_root = sys.modules.get("qiskit_aqt_provider")
        previous_provider = sys.modules.get("qiskit_aqt_provider.aqt_provider")
        previous_qiskit = sys.modules.get("qiskit")
        sys.modules["qiskit_aqt_provider"] = root
        sys.modules["qiskit_aqt_provider.aqt_provider"] = provider_module
        sys.modules["qiskit"] = qiskit_module
        try:
            _, job_id = submit_aqt_job(
                object(),
                token="aqt-token",
                backend_name="simulator_cloud",
                shots=200,
                workspace="workspace-a",
            )
        finally:
            for name, previous in (
                ("qiskit_aqt_provider", previous_root),
                ("qiskit_aqt_provider.aqt_provider", previous_provider),
                ("qiskit", previous_qiskit),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(job_id, "aqt-job")
        self.assertEqual(captured["token"], "aqt-token")
        self.assertEqual(captured["backend"], "simulator_cloud")
        self.assertEqual(captured["workspace"], "workspace-a")
        self.assertEqual(captured["shots"], 200)

    def test_aqt_cloud_discovery_uses_resource_type_not_name_heuristics(self) -> None:
        captured = []

        class Backend:
            name = "resource-42"
            resource_id = "resource-42"
            workspace_id = "research-team"
            num_qubits = 12

            def status(self):
                return True

        class Provider:
            def __init__(self, token=None):
                self.token = token

            def backends(self, *, backend_type=None):
                captured.append(backend_type)
                return [Backend()] if backend_type == "simulator" else []

        root = types.ModuleType("qiskit_aqt_provider")
        root.AQTProvider = Provider
        previous_root = sys.modules.get("qiskit_aqt_provider")
        sys.modules["qiskit_aqt_provider"] = root
        try:
            devices = discover_aqt_simulators("token", offline=False)
        finally:
            if previous_root is None:
                sys.modules.pop("qiskit_aqt_provider", None)
            else:
                sys.modules["qiskit_aqt_provider"] = previous_root

        self.assertEqual(captured, ["simulator"])
        self.assertEqual([device.name for device in devices], ["resource-42"])
        self.assertEqual(devices[0].region, "research-team")
        self.assertEqual(
            devices[0].identifier, "research-team:resource-42"
        )

    def test_ibm_fake_discovery_filters_oversized_snapshots(self) -> None:
        class Backend:
            def __init__(self, name, qubits):
                self.name = name
                self.num_qubits = qubits

        class Provider:
            def backends(self):
                return [Backend("fake_small", 7), Backend("fake_huge", 127)]

        root = types.ModuleType("qiskit_ibm_runtime")
        fake_module = types.ModuleType("qiskit_ibm_runtime.fake_provider")
        fake_module.FakeProviderForBackendV2 = Provider
        root.fake_provider = fake_module
        previous_root = sys.modules.get("qiskit_ibm_runtime")
        previous_fake = sys.modules.get("qiskit_ibm_runtime.fake_provider")
        sys.modules["qiskit_ibm_runtime"] = root
        sys.modules["qiskit_ibm_runtime.fake_provider"] = fake_module
        hardware._ibm_fake_backend_catalog.cache_clear()
        try:
            devices = discover_ibm_fake_backends(
                min_num_qubits=5,
                max_num_qubits=32,
            )
        finally:
            hardware._ibm_fake_backend_catalog.cache_clear()
            for name, previous in (
                ("qiskit_ibm_runtime", previous_root),
                ("qiskit_ibm_runtime.fake_provider", previous_fake),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual([device.name for device in devices], ["fake_small"])

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
