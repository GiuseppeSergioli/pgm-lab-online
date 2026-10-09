"""Optional local and cloud execution helpers for the trained PGM circuit.

Cloud SDKs are imported lazily. No credential is written to disk, logged, cached,
or embedded in a circuit/export by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import inspect
import os
import sys
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

from quantum_pgm import NaimarkDilation, outcome_probabilities


@dataclass(frozen=True)
class DeviceDescriptor:
    provider: str
    name: str
    identifier: str
    device_type: str
    status: str
    region: str
    qubits: int | None
    simulator: bool
    pending_jobs: int | None = None
    max_shots: int | None = None
    notes: str = ""


def hardware_python_status() -> dict[str, Any]:
    version = sys.version_info
    return {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "aws_compatible": (version.major, version.minor) >= (3, 11),
        "lrz_compatible": (3, 9) <= (version.major, version.minor) < (3, 14),
        "extended_providers_compatible": (version.major, version.minor) >= (3, 10),
    }


def build_sample_circuit(
    dilation: NaimarkDilation,
    reduced_state: NDArray,
    *,
    name: str = "PGM_classification",
    optimized_isometry: bool = True,
) -> Any:
    """Create an executable circuit including test-state preparation and PGM."""

    if dilation.unitary is None or dilation.isometry is None:
        raise ValueError("Per l'esecuzione hardware serve la dilatazione esatta.")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import Isometry, StatePreparation, UnitaryGate

    resources = dilation.resources
    state = np.asarray(reduced_state, dtype=np.complex128).reshape(-1)
    if state.size != resources.feature_dimension:
        raise ValueError("Lo stato test non ha la dimensione ridotta attesa.")
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("Lo stato test deve avere norma unitaria.")

    padded = np.zeros(resources.padded_system_dimension, dtype=np.complex128)
    padded[: state.size] = state
    system = QuantumRegister(resources.system_qubits, "sys")
    outcome = QuantumRegister(resources.outcome_qubits, "out")
    classical = ClassicalRegister(resources.outcome_qubits, "m")
    circuit = QuantumCircuit(system, outcome, classical, name=name)
    circuit.append(
        StatePreparation(padded, normalize=False, label="Prepare |phi_c(x)>"),
        list(system),
    )
    circuit.barrier()
    if optimized_isometry:
        circuit.append(
            Isometry(
                dilation.isometry,
                # Output ancillas are already implicit in the isometry dimensions.
                num_ancillas_zero=0,
                num_ancillas_dirty=0,
                epsilon=1e-12,
            ),
            list(system) + list(outcome),
        )
    else:
        circuit.append(
            UnitaryGate(dilation.unitary, label="U_PGM (Naimark)"),
            list(system) + list(outcome),
        )
    circuit.barrier()
    circuit.measure(outcome, classical)
    return circuit


def circuit_from_qpy(payload: bytes) -> Any:
    from qiskit import qpy

    circuits = qpy.load(BytesIO(payload))
    if len(circuits) != 1:
        raise ValueError("Il file QPY deve contenere esattamente un circuito.")
    return circuits[0]


def simulate_ideal_shots(
    dilation: NaimarkDilation,
    reduced_state: NDArray,
    *,
    shots: int,
    seed: int,
) -> dict[str, int]:
    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")
    probabilities = outcome_probabilities(dilation, reduced_state)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    samples = np.random.default_rng(seed).multinomial(shots, probabilities)
    width = dilation.resources.outcome_qubits
    return {
        format(index, f"0{width}b"): int(count)
        for index, count in enumerate(samples)
        if count > 0
    }


def simulate_aer_shots(
    circuit: Any,
    *,
    shots: int,
    seed: int,
    method: str = "automatic",
    optimization_level: int = 1,
    one_qubit_error: float = 0.0,
    two_qubit_error: float = 0.0,
    readout_error: float = 0.0,
) -> dict[str, int]:
    """Execute the actual circuit with Qiskit Aer, optionally with simple noise."""

    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")
    for label, probability in (
        ("errore a un qubit", one_qubit_error),
        ("errore a due qubit", two_qubit_error),
        ("errore di lettura", readout_error),
    ):
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"La probabilità di {label} deve essere in [0, 1).")

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    noise_model = None
    if any(value > 0.0 for value in (one_qubit_error, two_qubit_error, readout_error)):
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

        noise_model = NoiseModel()
        if one_qubit_error > 0.0:
            one_error = depolarizing_error(one_qubit_error, 1)
            for gate in ("rz", "sx", "x", "rx", "ry", "h"):
                noise_model.add_all_qubit_quantum_error(one_error, gate)
        if two_qubit_error > 0.0:
            two_error = depolarizing_error(two_qubit_error, 2)
            for gate in ("cx", "cz", "ecr", "rxx", "rzz"):
                noise_model.add_all_qubit_quantum_error(two_error, gate)
        if readout_error > 0.0:
            readout = ReadoutError(
                [
                    [1.0 - readout_error, readout_error],
                    [readout_error, 1.0 - readout_error],
                ]
            )
            noise_model.add_all_qubit_readout_error(readout)

    simulator = AerSimulator(method=method, noise_model=noise_model)
    transpiled = transpile(
        circuit,
        simulator,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )
    result = simulator.run(
        transpiled,
        shots=int(shots),
        seed_simulator=int(seed),
    ).result()
    counts = result.get_counts()
    return {str(key): int(value) for key, value in counts.items()}


def simulate_basic_shots(
    circuit: Any,
    *,
    shots: int,
    seed: int,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Execute a circuit with Qiskit's dependency-free reference simulator.

    ``BasicSimulator`` is intentionally kept as a conservative fallback.  It is
    slower than Aer, but it ships with Qiskit itself and therefore remains useful
    when an optional native simulator wheel cannot be loaded on a platform.
    """

    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")

    from qiskit import transpile
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    executable = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )
    job = backend.run(
        executable,
        shots=int(shots),
        seed_simulator=int(seed),
    )
    return generic_job_counts(job, shots=shots)


def simulate_braket_local_shots(
    circuit: Any,
    *,
    shots: int,
    seed: int,
    backend_name: str,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Execute a Qiskit circuit on a local Amazon Braket simulator.

    Supported names are ``braket_sv`` (state vector) and ``braket_dm`` (density
    matrix).  No AWS account, network request, S3 bucket, or credential is used.
    The seed is applied to transpilation; Braket's local sampling API does not
    currently expose a portable sampling-seed option through its Qiskit adapter.
    """

    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")
    if backend_name not in {"braket_sv", "braket_dm"}:
        raise ValueError("Simulatore Amazon Braket locale non riconosciuto.")

    from qiskit import transpile
    try:
        # Public import used by current qiskit-braket-provider releases.
        from qiskit_braket_provider import BraketLocalBackend
    except ImportError:  # pragma: no cover - compatibility with older layouts
        # Keep the pinned 0.4.x provider usable if the class is exposed only
        # from its historical submodule.
        from qiskit_braket_provider.providers import BraketLocalBackend

    backend = BraketLocalBackend(name=backend_name)
    executable = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )
    job = backend.run(executable, shots=int(shots))
    return generic_job_counts(job, shots=shots)


def normalized_counts(
    counts: Mapping[str, int | float],
    *,
    outcome_qubits: int,
) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for raw_key, raw_count in counts.items():
        key = str(raw_key).replace(" ", "")[-outcome_qubits:]
        cleaned[key] = cleaned.get(key, 0.0) + float(raw_count)
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Il risultato non contiene conteggi utilizzabili.")
    return {key: count / total for key, count in cleaned.items()}


def prediction_from_counts(
    counts: Mapping[str, int | float],
    classes: NDArray,
    *,
    outcome_qubits: int,
) -> tuple[Any, dict[str, float]]:
    frequencies = normalized_counts(counts, outcome_qubits=outcome_qubits)
    valid = {
        bitstring: probability
        for bitstring, probability in frequencies.items()
        if int(bitstring, 2) < len(classes)
    }
    if not valid:
        raise ValueError("Tutti gli shot sono finiti in esiti binari non assegnati.")
    winning_bits = max(valid, key=valid.get)
    return np.asarray(classes)[int(winning_bits, 2)], frequencies


def _safe_attribute(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _safe_call_or_value(obj: Any, name: str, default: Any = None) -> Any:
    value = _safe_attribute(obj, name, default)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _backend_configuration(backend: Any) -> Any | None:
    return _safe_call_or_value(backend, "configuration", None)


def _backend_name(backend: Any) -> str:
    value = _safe_call_or_value(backend, "name", "")
    if not value:
        value = _safe_call_or_value(backend, "backend_name", "")
    return str(value or backend.__class__.__name__)


def _backend_qubits(backend: Any) -> tuple[int | None, str]:
    """Read qubit metadata without letting a missing BackendV2 target abort discovery."""

    warning = ""
    try:
        value = getattr(backend, "num_qubits")
        if callable(value):
            value = value()
        if value is not None:
            return int(value), warning
    except Exception as error:
        warning = f"Metadati target non disponibili: {error}"

    configuration = _backend_configuration(backend)
    for name in ("num_qubits", "n_qubits"):
        value = _safe_attribute(configuration, name, None) if configuration else None
        if value is not None:
            try:
                return int(value), warning
            except (TypeError, ValueError):
                pass
    return None, warning


def _backend_status(backend: Any) -> tuple[str, bool | None, int | None]:
    try:
        status = backend.status()
    except Exception:
        return "UNKNOWN", None, None

    # The IonQ provider intentionally returns a boolean instead of BackendStatus.
    if isinstance(status, bool):
        return ("ONLINE" if status else "OFFLINE"), status, None

    status_message = _safe_attribute(status, "status_msg", None)
    operational = _safe_attribute(status, "operational", None)
    pending = _safe_attribute(status, "pending_jobs", None)
    if pending is None:
        pending = _safe_attribute(status, "pending_jobs_count", None)
    try:
        pending_value = int(pending) if pending is not None else None
    except (TypeError, ValueError):
        pending_value = None

    if status_message:
        text = str(status_message)
    elif operational is True:
        text = "ONLINE"
    elif operational is False:
        text = "OFFLINE"
    else:
        status_name = _safe_attribute(status, "name", None)
        text = str(status_name or status)
    return text, bool(operational) if operational is not None else None, pending_value


def _backend_pending_jobs(backend: Any, fallback: int | None = None) -> int | None:
    try:
        value = getattr(backend, "num_pending_jobs")
        if callable(value):
            value = value()
        return int(value)
    except Exception:
        return fallback


def _backend_simulator_flag(backend: Any, name: str) -> bool:
    configuration = _backend_configuration(backend)
    flag = _safe_attribute(configuration, "simulator", None) if configuration else None
    if flag is None:
        flag = _safe_attribute(backend, "simulator", None)
    if flag is not None:
        return bool(flag)
    return "simulator" in name.lower()


def _backend_max_shots(backend: Any) -> int | None:
    configuration = _backend_configuration(backend)
    candidates = [
        _safe_attribute(configuration, "max_shots", None) if configuration else None,
        _safe_attribute(backend, "max_shots", None),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _job_identifier(job: Any) -> str:
    for name in ("job_id", "id"):
        value = _safe_call_or_value(job, name, None)
        if value:
            return str(value)
    return "ID non disponibile"


def _probabilities_to_counts(
    values: Mapping[str, int | float],
    *,
    shots: int | None,
) -> dict[str, int]:
    numeric = {str(key): float(value) for key, value in values.items()}
    if not numeric:
        raise ValueError("Il provider ha restituito un risultato vuoto.")

    total = float(sum(numeric.values()))
    looks_like_probabilities = (
        shots is not None
        and shots > 0
        and all(value >= 0.0 for value in numeric.values())
        and np.isclose(total, 1.0, atol=1e-6, rtol=1e-6)
    )
    if not looks_like_probabilities:
        return {key: int(round(value)) for key, value in numeric.items()}

    target = int(shots)
    exact = {key: value * target for key, value in numeric.items()}
    rounded = {key: int(np.floor(value)) for key, value in exact.items()}
    missing = target - sum(rounded.values())
    if missing > 0:
        order = sorted(exact, key=lambda key: exact[key] - rounded[key], reverse=True)
        for key in order[:missing]:
            rounded[key] += 1
    return rounded


def aws_available_profiles() -> tuple[str, ...]:
    try:
        import boto3
    except ImportError:
        return ()
    return tuple(sorted(set(boto3.Session().available_profiles)))


def _aws_sessions(
    profile: str | None,
    region: str,
    *,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> tuple[Any, Any]:
    import boto3
    from braket.aws import AwsSession

    if bool(access_key_id) != bool(secret_access_key):
        raise ValueError(
            "AWS Access Key ID e Secret Access Key devono essere fornite insieme."
        )
    session_arguments: dict[str, Any] = {"region_name": region}
    if profile:
        session_arguments["profile_name"] = profile
    if access_key_id and secret_access_key:
        session_arguments.update(
            {
                "aws_access_key_id": access_key_id,
                "aws_secret_access_key": secret_access_key,
            }
        )
        if session_token:
            session_arguments["aws_session_token"] = session_token
    boto_session = boto3.Session(**session_arguments)
    return boto_session, AwsSession(boto_session=boto_session)


def verify_aws_identity(
    profile: str | None,
    region: str,
    *,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> dict[str, str]:
    boto_session, _ = _aws_sessions(
        profile,
        region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )
    identity = boto_session.client("sts").get_caller_identity()
    return {
        "account": str(identity.get("Account", "")),
        "arn": str(identity.get("Arn", "")),
        "user_id": str(identity.get("UserId", "")),
        "region": region,
    }


def discover_aws_devices(
    profile: str | None,
    region: str,
    *,
    simulators: bool,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> tuple[DeviceDescriptor, ...]:
    from braket.aws import AwsDevice
    from braket.aws.aws_device import AwsDeviceType

    _, aws_session = _aws_sessions(
        profile,
        region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )
    requested_type = AwsDeviceType.SIMULATOR if simulators else AwsDeviceType.QPU
    discovery_arguments: dict[str, Any] = {
        "types": [requested_type],
        "aws_session": aws_session,
    }
    if not simulators:
        discovery_arguments["statuses"] = ["ONLINE"]
    devices = AwsDevice.get_devices(**discovery_arguments)
    descriptors = []
    for device in devices:
        properties = getattr(device, "properties", None)
        paradigm = getattr(properties, "paradigm", None)
        qubits = getattr(paradigm, "qubitCount", None)
        arn = str(getattr(device, "arn", ""))
        device_region = arn.split(":")[3] if arn.count(":") >= 3 else region
        descriptors.append(
            DeviceDescriptor(
                provider=str(getattr(device, "provider_name", "Amazon Braket")),
                name=str(getattr(device, "name", arn)),
                identifier=arn,
                device_type="SIMULATOR" if simulators else "QPU",
                status=str(getattr(device, "status", "UNKNOWN")),
                region=device_region,
                qubits=int(qubits) if qubits is not None else None,
                simulator=simulators,
            )
        )
    return tuple(sorted(descriptors, key=lambda item: (item.provider, item.name)))


@contextmanager
def _temporary_aws_environment(profile: str, region: str) -> Iterator[None]:
    keys = ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["AWS_PROFILE"] = profile
    os.environ["AWS_REGION"] = region
    os.environ["AWS_DEFAULT_REGION"] = region
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _create_braket_provider(
    profile: str | None,
    region: str,
    *,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> Any:
    from qiskit_braket_provider import BraketProvider

    _, aws_session = _aws_sessions(
        profile,
        region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )
    try:
        parameters = inspect.signature(BraketProvider).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "aws_session" in parameters:
        return BraketProvider(aws_session=aws_session)
    if access_key_id or secret_access_key or session_token:
        raise RuntimeError(
            "Questa versione del provider Braket non accetta una sessione AWS "
            "isolata. Aggiornare qiskit-braket-provider."
        )
    if not profile:
        raise RuntimeError("Nessun profilo o credenziale AWS disponibile.")
    with _temporary_aws_environment(profile, region):
        return BraketProvider()


def submit_aws_job(
    circuit: Any,
    *,
    profile: str | None,
    region: str,
    backend_name: str,
    backend_identifier: str | None = None,
    shots: int,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> tuple[Any, str]:
    provider = _create_braket_provider(
        profile,
        region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )

    def submit() -> tuple[Any, str]:
        backend = None
        lookup_errors = []
        for candidate in (backend_identifier, backend_name):
            if not candidate:
                continue
            try:
                backend = provider.get_backend(candidate)
                break
            except Exception as error:
                lookup_errors.append(str(error))
        if backend is None:
            raise LookupError(
                "Backend Amazon Braket non trovato: " + " | ".join(lookup_errors)
            )
        job = backend.run(circuit, shots=shots)
        return job, _job_identifier(job)

    if profile:
        with _temporary_aws_environment(profile, region):
            return submit()
    return submit()


def discover_lrz_backends(
    token: str,
    *,
    online_only: bool = True,
) -> tuple[DeviceDescriptor, ...]:
    """Discover LRZ resources without failing on one backend's missing target.

    Some MQSS resources can be listed while their Qiskit ``Target`` metadata is not
    available. Accessing ``BackendV2.num_qubits`` then raises ``NotImplementedError``.
    The resource is retained with unknown qubit count instead of aborting discovery.
    """

    from mqss.qiskit_adapter import MQSSQiskitAdapter

    adapter = MQSSQiskitAdapter(token=token)
    try:
        backends = adapter.backends(online=online_only)
    except TypeError:
        # Compatibility with older adapter releases that did not expose online=.
        backends = adapter.backends()

    descriptors = []
    for backend in backends:
        name = _backend_name(backend)
        status_value, operational, pending_from_status = _backend_status(backend)
        if online_only and operational is False:
            continue
        qubits, metadata_warning = _backend_qubits(backend)
        pending_jobs = _backend_pending_jobs(backend, pending_from_status)
        simulator = _backend_simulator_flag(backend, name)
        descriptors.append(
            DeviceDescriptor(
                provider="LRZ / MQSS",
                name=name,
                identifier=name,
                device_type="SIMULATORE MQSS" if simulator else "QPU / RISORSA MQSS",
                status=status_value,
                region="LRZ Munich",
                qubits=qubits,
                simulator=simulator,
                pending_jobs=pending_jobs,
                max_shots=_backend_max_shots(backend),
                notes=metadata_warning,
            )
        )
    return tuple(sorted(descriptors, key=lambda item: item.name.lower()))


def submit_lrz_job(
    circuit: Any,
    *,
    token: str,
    backend_name: str,
    shots: int,
    queued: bool = True,
) -> tuple[Any, str]:
    from mqss.qiskit_adapter import MQSSQiskitAdapter

    adapter = MQSSQiskitAdapter(token=token)
    backend = adapter.get_backend(backend_name)
    job = backend.run(circuit, shots=shots, queued=queued)
    return job, _job_identifier(job)


def _ionq_provider(token: str | None) -> Any:
    from qiskit_ionq import IonQProvider

    return IonQProvider(token) if token else IonQProvider()


def _ionq_catalog(provider: Any) -> list[Mapping[str, Any]]:
    """Read IonQ's live backend catalog, with a graceful SDK-only fallback.

    ``IonQProvider.backends()`` exposes generic QPU/simulator handles.  The cloud
    API catalog contains the concrete systems (for example ``qpu.forte-1``),
    their current status, qubit count, location, and average queue time.
    """

    try:
        generic_backends = list(provider.backends())
        if not generic_backends:
            return []
        client = _safe_attribute(generic_backends[0], "client", None)
        if client is None:
            return []
        make_path = _safe_attribute(client, "make_path", None)
        get_with_retry = _safe_attribute(client, "get_with_retry", None)
        if not callable(make_path) or not callable(get_with_retry):
            return []
        response = get_with_retry(
            make_path("backends"),
            headers=_safe_attribute(client, "api_headers", None),
        )
        payload = response.json()
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            items = payload.get("backends", [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, Mapping)]
    except Exception:
        return []
    return []


def _ionq_queue_note(raw_seconds: Any, degraded: Any) -> str:
    notes: list[str] = []
    try:
        seconds = max(0, int(float(raw_seconds)))
    except (TypeError, ValueError):
        seconds = 0
    if seconds:
        if seconds < 120:
            queue_text = f"{seconds} s"
        elif seconds < 7_200:
            queue_text = f"{seconds / 60.0:.0f} min"
        else:
            queue_text = f"{seconds / 3_600.0:.1f} h"
        notes.append(f"Attesa media dichiarata: {queue_text}")
    if bool(degraded):
        notes.append("Prestazioni dichiarate come degradate")
    return " · ".join(notes)


def discover_ionq_devices(
    token: str | None,
    *,
    simulators: bool,
) -> tuple[DeviceDescriptor, ...]:
    provider = _ionq_provider(token)
    catalog = _ionq_catalog(provider)
    if catalog:
        descriptors = []
        for item in catalog:
            backend_id = str(item.get("backend", "")).strip()
            if not backend_id:
                continue
            is_simulator = backend_id == "simulator" or backend_id.startswith(
                "simulator."
            )
            if is_simulator != simulators:
                continue
            raw_status = str(item.get("status", "UNKNOWN")).strip()
            status_key = raw_status.lower()
            if status_key in {"unavailable", "offline", "retired"}:
                continue
            try:
                qubits = int(item["qubits"]) if item.get("qubits") is not None else None
            except (TypeError, ValueError):
                qubits = None
            descriptors.append(
                DeviceDescriptor(
                    provider="IonQ Quantum Cloud",
                    name=backend_id,
                    identifier=backend_id,
                    device_type=(
                        "SIMULATORE CLOUD"
                        if is_simulator
                        else "QPU IONI INTRAPPOLATI"
                    ),
                    status=raw_status.upper() or "UNKNOWN",
                    region=str(item.get("location") or "IonQ Cloud"),
                    qubits=qubits,
                    simulator=is_simulator,
                    notes=_ionq_queue_note(
                        item.get("average_queue_time"), item.get("degraded")
                    ),
                )
            )
        if descriptors:
            return tuple(sorted(descriptors, key=lambda item: item.name.lower()))

    # Fallback for provider/API versions that do not expose the catalog endpoint.
    descriptors = []
    for backend in provider.backends():
        name = _backend_name(backend)
        is_simulator = _backend_simulator_flag(backend, name)
        if is_simulator != simulators:
            continue
        status_value, operational, pending_from_status = _backend_status(backend)
        if operational is False:
            continue
        qubits, metadata_warning = _backend_qubits(backend)
        descriptors.append(
            DeviceDescriptor(
                provider="IonQ Quantum Cloud",
                name=name,
                identifier=name,
                device_type="SIMULATORE CLOUD" if is_simulator else "QPU IONI INTRAPPOLATI",
                status=status_value,
                region="IonQ Cloud",
                qubits=qubits,
                simulator=is_simulator,
                pending_jobs=_backend_pending_jobs(backend, pending_from_status),
                max_shots=_backend_max_shots(backend),
                notes=metadata_warning,
            )
        )
    return tuple(sorted(descriptors, key=lambda item: item.name.lower()))


def submit_ionq_job(
    circuit: Any,
    *,
    token: str | None,
    backend_name: str,
    shots: int,
    optimization_level: int = 1,
    noise_model: str | None = None,
) -> tuple[Any, str]:
    from qiskit import transpile

    provider = _ionq_provider(token)
    backend = provider.get_backend(backend_name)
    executable = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
    )
    run_options: dict[str, Any] = {"shots": int(shots)}
    if noise_model and noise_model != "ideal":
        run_options["noise_model"] = noise_model
    elif "simulator" in backend_name.lower():
        # IonQ's ideal state-vector simulator returns probabilities and accepts
        # one physical shot. The caller later expands those probabilities to the
        # requested virtual shot count for a uniform report.
        run_options["shots"] = 1
    job = backend.run(executable, **run_options)
    return job, _job_identifier(job)


def _aqt_provider(token: str | None) -> Any:
    try:
        # Public import documented by qiskit-aqt-provider 1.15.
        from qiskit_aqt_provider import AQTProvider
    except ImportError:  # pragma: no cover - compatibility with older layouts
        from qiskit_aqt_provider.aqt_provider import AQTProvider

    return AQTProvider(token) if token else AQTProvider()


def _aqt_backend(
    provider: Any,
    backend_name: str,
    *,
    workspace: str | None = None,
) -> Any:
    """Resolve an AQT resource across provider versions and workspaces."""

    errors: list[str] = []
    attempts: list[dict[str, str]] = []
    if workspace:
        attempts.append({"name": backend_name, "workspace": workspace})
    attempts.append({"name": backend_name})
    if workspace != "default":
        attempts.append({"name": backend_name, "workspace": "default"})
    for arguments in attempts:
        try:
            return provider.get_backend(**arguments)
        except TypeError:
            try:
                workspace = arguments.get("workspace")
                if workspace is None:
                    return provider.get_backend(backend_name)
                return provider.get_backend(backend_name, workspace=workspace)
            except Exception as error:
                errors.append(str(error))
        except Exception as error:
            errors.append(str(error))
    raise LookupError(
        "Backend AQT non trovato: " + " | ".join(error for error in errors if error)
    )


def discover_aqt_simulators(
    token: str | None,
    *,
    offline: bool,
) -> tuple[DeviceDescriptor, ...]:
    """Return either AQT's local offline simulators or authorized cloud ones."""

    provider = _aqt_provider(token)
    requested_type = "offline_simulator" if offline else "simulator"
    filtered_by_type = True
    try:
        backends = list(provider.backends(backend_type=requested_type))
    except TypeError:  # pragma: no cover - compatibility with older providers
        filtered_by_type = False
        backends = list(provider.backends())
    descriptors: list[DeviceDescriptor] = []
    for backend in backends:
        name = _backend_name(backend)
        name_key = name.lower()
        is_offline = name_key.startswith("offline_") or "offline" in name_key
        if not filtered_by_type and is_offline != bool(offline):
            continue
        if (
            not filtered_by_type
            and not _backend_simulator_flag(backend, name)
            and "sim" not in name_key
        ):
            continue
        status_value, operational, pending_from_status = _backend_status(backend)
        if operational is False:
            continue
        qubits, metadata_warning = _backend_qubits(backend)
        workspace = _safe_attribute(backend, "workspace_id", None)
        resource_id = _safe_attribute(backend, "resource_id", None)
        notes = metadata_warning
        if workspace:
            workspace_note = f"Workspace: {workspace}"
            notes = " · ".join(item for item in (notes, workspace_note) if item)
        descriptors.append(
            DeviceDescriptor(
                provider="AQT Offline" if offline else "AQT Cloud",
                name=name,
                identifier=(
                    f"{workspace}:{resource_id or name}"
                    if workspace
                    else str(resource_id or name)
                ),
                device_type=(
                    "SIMULATORE LOCALE AQT"
                    if offline
                    else "SIMULATORE CLOUD AQT"
                ),
                status="LOCAL" if offline else status_value,
                region="Locale" if offline else str(workspace or "AQT Cloud"),
                qubits=qubits,
                simulator=True,
                pending_jobs=_backend_pending_jobs(backend, pending_from_status),
                max_shots=_backend_max_shots(backend) or 2_000,
                notes=notes,
            )
        )
    return tuple(
        sorted(
            descriptors,
            key=lambda item: (
                item.qubits if item.qubits is not None else 10**9,
                item.name.lower(),
            ),
        )
    )


def submit_aqt_job(
    circuit: Any,
    *,
    token: str | None,
    backend_name: str,
    shots: int,
    optimization_level: int = 1,
    workspace: str | None = None,
) -> tuple[Any, str]:
    """Submit to an AQT offline or cloud simulator through its Qiskit provider."""

    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")
    from qiskit import transpile

    provider = _aqt_provider(token)
    backend = _aqt_backend(provider, backend_name, workspace=workspace)
    executable = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
    )
    job = backend.run(executable, shots=int(shots))
    return job, _job_identifier(job)


def simulate_aqt_offline_shots(
    circuit: Any,
    *,
    shots: int,
    backend_name: str,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Run one of AQT's bundled ideal/noisy offline simulators synchronously."""

    job, _ = submit_aqt_job(
        circuit,
        token=None,
        backend_name=backend_name,
        shots=shots,
        optimization_level=optimization_level,
    )
    return generic_job_counts(job, shots=shots)


@lru_cache(maxsize=1)
def _ibm_fake_backend_catalog() -> tuple[Any, ...]:
    """Instantiate IBM snapshot backends once per process."""

    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    provider = FakeProviderForBackendV2()
    return tuple(provider.backends())


def discover_ibm_fake_backends(
    *,
    min_num_qubits: int = 1,
    max_num_qubits: int = 32,
) -> tuple[DeviceDescriptor, ...]:
    """List practical IBM device snapshots for local noisy Aer simulation.

    Very large snapshots are deliberately hidden.  The PGM circuit synthesis is
    already limited to a small number of qubits, and loading 100+ qubit noise
    models would add latency without improving the scientific comparison.
    """

    descriptors: list[DeviceDescriptor] = []
    seen: set[str] = set()
    for backend in _ibm_fake_backend_catalog():
        name = _backend_name(backend)
        if name in seen:
            continue
        qubits, metadata_warning = _backend_qubits(backend)
        if qubits is None or not int(min_num_qubits) <= qubits <= int(max_num_qubits):
            continue
        seen.add(name)
        descriptors.append(
            DeviceDescriptor(
                provider="IBM Fake Backend",
                name=name,
                identifier=name,
                device_type="SIMULATORE LOCALE CON SNAPSHOT IBM",
                status="LOCAL",
                region="Locale",
                qubits=qubits,
                simulator=True,
                max_shots=_backend_max_shots(backend),
                notes=metadata_warning,
            )
        )
    return tuple(sorted(descriptors, key=lambda item: (item.qubits or 10**9, item.name)))


def simulate_ibm_fake_shots(
    circuit: Any,
    *,
    backend_name: str,
    shots: int,
    seed: int,
    optimization_level: int = 1,
) -> dict[str, int]:
    """Simulate with an IBM topology and calibration snapshot through Aer."""

    if shots < 1:
        raise ValueError("Il numero di shot deve essere positivo.")
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    fake_backend = next(
        (
            backend
            for backend in _ibm_fake_backend_catalog()
            if _backend_name(backend) == backend_name
        ),
        None,
    )
    if fake_backend is None:
        raise LookupError(f"Snapshot IBM non trovato: {backend_name}")
    simulator = AerSimulator.from_backend(fake_backend)
    executable = transpile(
        circuit,
        backend=simulator,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed),
    )
    result = simulator.run(
        executable,
        shots=int(shots),
        seed_simulator=int(seed),
    ).result()
    counts = result.get_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _ibm_service(token: str | None, instance: str | None) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    arguments: dict[str, Any] = {"channel": "ibm_quantum_platform"}
    if token:
        arguments["token"] = token
    if instance:
        arguments["instance"] = instance
    return QiskitRuntimeService(**arguments)


def discover_ibm_devices(
    token: str | None,
    instance: str | None,
    *,
    simulators: bool = False,
    min_num_qubits: int = 1,
) -> tuple[DeviceDescriptor, ...]:
    service = _ibm_service(token, instance)
    backends = service.backends(
        simulator=simulators,
        operational=True,
        min_num_qubits=int(min_num_qubits),
    )
    descriptors = []
    for backend in backends:
        name = _backend_name(backend)
        status_value, _, pending_from_status = _backend_status(backend)
        qubits, metadata_warning = _backend_qubits(backend)
        descriptors.append(
            DeviceDescriptor(
                provider="IBM Quantum",
                name=name,
                identifier=name,
                device_type="SIMULATORE" if simulators else "QPU SUPERCONDUTTIVA",
                status=status_value,
                region="IBM Quantum Platform",
                qubits=qubits,
                simulator=simulators,
                pending_jobs=_backend_pending_jobs(backend, pending_from_status),
                max_shots=_backend_max_shots(backend),
                notes=metadata_warning,
            )
        )
    return tuple(
        sorted(
            descriptors,
            key=lambda item: (
                item.pending_jobs if item.pending_jobs is not None else 10**12,
                item.name.lower(),
            ),
        )
    )


def submit_ibm_job(
    circuit: Any,
    *,
    token: str | None,
    instance: str | None,
    backend_name: str,
    shots: int,
    optimization_level: int = 1,
) -> tuple[Any, str]:
    from qiskit_ibm_runtime import SamplerV2

    service = _ibm_service(token, instance)
    backend = service.backend(backend_name)
    try:
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )

        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(optimization_level),
        )
        executable = pass_manager.run(circuit)
    except (ImportError, AttributeError):
        from qiskit import transpile

        executable = transpile(
            circuit,
            backend=backend,
            optimization_level=int(optimization_level),
        )

    try:
        sampler = SamplerV2(mode=backend)
    except TypeError:
        sampler = SamplerV2(backend=backend)
    job = sampler.run([executable], shots=int(shots))
    return job, _job_identifier(job)


def generic_job_status(job: Any) -> str:
    status = job.status()
    return str(getattr(status, "name", status))


def _counts_from_primitive_result(result: Any) -> Mapping[str, int | float] | None:
    try:
        publication_result = result[0]
    except (TypeError, IndexError, KeyError):
        return None
    data = _safe_attribute(publication_result, "data", None)
    if data is None:
        return None

    register_names = ("m", "meas", "c", "cr")
    for register_name in register_names:
        register = _safe_attribute(data, register_name, None)
        get_counts = _safe_attribute(register, "get_counts", None)
        if callable(get_counts):
            counts = get_counts()
            if counts:
                return counts

    keys_method = _safe_attribute(data, "keys", None)
    if callable(keys_method):
        try:
            keys = list(keys_method())
        except Exception:
            keys = []
        for key in keys:
            register = _safe_attribute(data, str(key), None)
            get_counts = _safe_attribute(register, "get_counts", None)
            if callable(get_counts):
                counts = get_counts()
                if counts:
                    return counts
    return None


def generic_job_counts(job: Any, *, shots: int | None = None) -> dict[str, int]:
    direct_get_counts = _safe_attribute(job, "get_counts", None)
    if callable(direct_get_counts):
        try:
            direct_counts = direct_get_counts()
        except Exception:
            direct_counts = None
        if direct_counts:
            if isinstance(direct_counts, list):
                if len(direct_counts) != 1:
                    raise ValueError("Il job contiene più di un circuito.")
                direct_counts = direct_counts[0]
            return _probabilities_to_counts(direct_counts, shots=shots)

    result = job.result()
    if hasattr(result, "get_counts"):
        counts = result.get_counts()
        if isinstance(counts, list):
            if len(counts) != 1:
                raise ValueError("Il job contiene più di un circuito.")
            counts = counts[0]
        return _probabilities_to_counts(counts, shots=shots)

    primitive_counts = _counts_from_primitive_result(result)
    if primitive_counts is not None:
        return _probabilities_to_counts(primitive_counts, shots=shots)

    measurement_counts = getattr(result, "measurement_counts", None)
    if measurement_counts is not None:
        return _probabilities_to_counts(measurement_counts, shots=shots)
    raise ValueError("Il provider non ha restituito conteggi riconoscibili.")
