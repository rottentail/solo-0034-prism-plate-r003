"""Negative controls: inject each bias into a throwaway copy of the tree and show
that the pixel-level judgements in ``run.py`` reject it.

Three independent biases are injected, each in its own copied tree, so the
deliverable under verification is never modified:

``delayed-selection``
    The exposure re-reads the plate when it starts instead of using the revision
    captured at submission (the classic "reads latest稿" bug).

``geometry-reuse``
    The compiled scene is cached by plate id, so later revisions render with the
    first revision's contours (the classic "错误复用几何" bug).

``overlay-opacity``
    Layer opacity is dropped during raster composition, so geometry is intact but
    the transparent screen blend is wrong.

Every control asserts two things: the injected behaviour is visible in the real
PNG (so the check is aimed at the intended bias), and the corresponding journey
fails.  A mutation that the journey still passes would mean the judgement is not
effective.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import harness
import oracle
import scenes


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------

MUTATIONS = {
    "delayed-selection": {
        "file": "internal/service/service.go",
        "anchor": "\treturn s.Pool.Submit(func(ctx context.Context) (string, error) {\n"
                  "\t\tctx, cancel := context.WithTimeout(ctx, 30*time.Second)\n"
                  "\t\tdefer cancel()\n"
                  "\t\ta, e := s.render(ctx, id, r, format)\n",
        "replacement": "\treturn s.Pool.Submit(func(ctx context.Context) (string, error) {\n"
                       "\t\tctx, cancel := context.WithTimeout(ctx, 30*time.Second)\n"
                       "\t\tdefer cancel()\n"
                       "\t\tif latest, captureErr := s.Plate(id); captureErr == nil && len(latest.Revisions) > 0 {\n"
                       "\t\t\tr = latest.Revisions[len(latest.Revisions)-1]\n"
                       "\t\t}\n"
                       "\t\ta, e := s.render(ctx, id, r, format)\n",
        "description": "exposure resolves the latest revision at execution time",
    },
    "geometry-reuse": {
        "file": "internal/service/service.go",
        "anchor": "\tc, err := validate.Compile(r.Document)\n"
                  "\tif err != nil {\n"
                  "\t\treturn a, err\n"
                  "\t}\n",
        "replacement": "\tc, err := compiledFor(id, r)\n"
                       "\tif err != nil {\n"
                       "\t\treturn a, err\n"
                       "\t}\n",
        "import_anchor": '\t"strconv"\n',
        "import_addition": '\t"strconv"\n\t"sync"\n',
        "helper": "\nvar (\n"
                  "\tgeometryCacheMu sync.Mutex\n"
                  "\tgeometryCache   = map[string]scene.Compiled{}\n"
                  ")\n\n"
                  "// compiledFor returns the first compiled scene for a plate, mimicking an\n"
                  "// unsafe geometry cache that ignores the revision digest.\n"
                  "func compiledFor(id string, r repository.Revision) (scene.Compiled, error) {\n"
                  "\tgeometryCacheMu.Lock()\n"
                  "\tdefer geometryCacheMu.Unlock()\n"
                  "\tif cached, ok := geometryCache[id]; ok {\n"
                  "\t\treturn cached, nil\n"
                  "\t}\n"
                  "\tcompiled, err := validate.Compile(r.Document)\n"
                  "\tif err == nil {\n"
                  "\t\tgeometryCache[id] = compiled\n"
                  "\t}\n"
                  "\treturn compiled, err\n"
                  "}\n",
        "description": "compiled geometry is cached per plate, ignoring the revision",
    },
    "overlay-opacity": {
        "file": "internal/raster/raster.go",
        "anchor": "dst = paint.Over(dst, src.Scale(l.Layer.Opacity), l.Layer.Blend)",
        "replacement": "dst = paint.Over(dst, src, l.Layer.Blend)",
        "description": "layer opacity dropped; geometry intact, transparent blend wrong",
    },
}


class MutationError(RuntimeError):
    pass


def build_mutant(scratch: Path, name: str) -> tuple[str, dict]:
    specification = MUTATIONS[name]
    tree = scratch / f"tree-{name}"
    shutil.copytree(
        harness.ROOT, tree,
        ignore=shutil.ignore_patterns(".git", "verify", "bin", "var", "__pycache__"))
    target = tree / specification["file"]
    text = target.read_text()
    if specification["anchor"] not in text:
        raise MutationError(f"{name}: anchor not found in {specification['file']}")
    text = text.replace(specification["anchor"], specification["replacement"], 1)
    if specification.get("import_addition"):
        if specification["import_anchor"] not in text:
            raise MutationError(f"{name}: import anchor not found")
        text = text.replace(specification["import_anchor"],
                            specification["import_addition"], 1)
    if specification.get("helper"):
        text = text.replace("type Service struct {",
                            specification["helper"] + "\ntype Service struct {", 1)
    target.write_text(text)
    completed = subprocess.run(
        [harness.go_toolchain(), "build", "-o", str(scratch / f"prismplate-{name}"),
         "./cmd/prismplate"],
        cwd=tree, capture_output=True, text=True,
        env={**harness.environment(), "GOCACHE": "/tmp/prism-go-cache"})
    if completed.returncode != 0:
        raise MutationError(f"{name}: mutant build failed: {completed.stderr[-2000:]}")
    return str(scratch / f"prismplate-{name}"), {
        "bias": name,
        "file": specification["file"],
        "description": specification["description"],
    }


# ---------------------------------------------------------------------------
# bias probes (do not reuse journey code, so both must independently agree)
# ---------------------------------------------------------------------------

def probe_bias(name: str, binary: str, workdir: Path) -> dict:
    server = harness.Server(binary, workdir / f"bias-{name}")
    try:
        if name == "delayed-selection":
            return _probe_delayed(server)
        if name == "geometry-reuse":
            return _probe_geometry(server)
        if name == "overlay-opacity":
            return _probe_overlay(server)
        raise ValueError(name)
    finally:
        server.stop()


def _probe_delayed(server) -> dict:
    server.create_plate("gate-a", "甲", scenes.gate_document(512, 8))
    server.create_plate("gate-b", "乙", scenes.gate_document(512, 8))
    gate_a = server.submit("gate-a", 1)
    _states_until(server, gate_a["id"], "running")
    gate_b = server.submit("gate-b", 1)
    _states_until(server, gate_b["id"], "running")
    server.create_plate("selection-plate", "选稿", scenes.selection_document(0))
    job = server.submit("selection-plate", 1)
    server.revise("selection-plate", 1, scenes.selection_document(1))
    server.revise("selection-plate", 2, scenes.selection_document(2))
    server.cancel(gate_a["id"])
    server.cancel(gate_b["id"])
    final, _ = server.wait_job(job["id"], timeout=60)
    raw, _ = server.artifact_bytes(final["artifact"])
    png = oracle.decode_png(raw)
    captured = oracle.CompiledDocument(scenes.selection_document(0)).probe_values()
    latest = oracle.CompiledDocument(scenes.selection_document(2)).probe_values()
    observed = {key: oracle.png_probe(png, *point) for key, point in scenes.PROBES.items()}
    differs_from_captured = sum(observed[key] != captured[key] for key in observed)
    differs_from_latest = sum(observed[key] != latest[key] for key in observed)
    return {
        "observed": observed,
        "captured_revision_mismatches": differs_from_captured,
        "latest_revision_mismatches": differs_from_latest,
        "bias_confirmed": differs_from_captured > 0 and differs_from_latest == 0,
        "artifact": final["artifact"],
    }


def _probe_geometry(server) -> dict:
    server.create_plate("selection-plate", "选稿", scenes.selection_document(0))
    # Render revision 1 first so an unsafe per-plate geometry cache has something
    # stale to reuse, then render revision 2 and inspect the real PNG.
    first_artifact = server.render_sync("selection-plate", 1, "png")
    server.revise("selection-plate", 1, scenes.selection_document(1))
    server.revise("selection-plate", 2, scenes.selection_document(2))
    second = server.render_sync("selection-plate", 2, "png")
    first_raw, _ = server.artifact_bytes(first_artifact["id"])
    raw, _ = server.artifact_bytes(second["id"])
    first_png = oracle.decode_png(first_raw)
    png = oracle.decode_png(raw)
    observed = {key: oracle.png_probe(png, *point) for key, point in scenes.PROBES.items()}
    first_observed = {key: oracle.png_probe(first_png, *point)
                      for key, point in scenes.PROBES.items()}
    expected_second = oracle.CompiledDocument(scenes.selection_document(1)).probe_values()
    bias_confirmed = (
        observed["hole_r2"] != expected_second["hole_r2"]
        and observed == first_observed
    )
    return {
        "observed_r2": observed,
        "observed_r1": first_observed,
        "expected_r2": expected_second,
        "reuses_first_revision_image": observed == first_observed,
        "bias_confirmed": bias_confirmed,
        "artifact": second["id"],
    }


def _probe_overlay(server) -> dict:
    server.create_plate("selection-plate", "选稿", scenes.selection_document(0))
    artifact = server.render_sync("selection-plate", 1, "png")
    raw, _ = server.artifact_bytes(artifact["id"])
    png = oracle.decode_png(raw)
    observed = {key: oracle.png_probe(png, *point) for key, point in scenes.PROBES.items()}
    expected = oracle.CompiledDocument(scenes.selection_document(0)).probe_values()
    # Probes outside the transparent glaze must be unaffected: the bias is the
    # lost layer opacity, not a geometry error.
    non_glaze = ["background", "ink_outside_glaze", "hole_r2"]
    geometry_mismatches = sum(observed[key] != expected[key] for key in non_glaze)
    return {
        "observed_overlay": observed["overlay"],
        "expected_overlay": expected["overlay"],
        "non_glaze_mismatches": geometry_mismatches,
        "bias_confirmed": observed["overlay"] != expected["overlay"] and geometry_mismatches == 0,
        "artifact": artifact["id"],
    }


def _states_until(server, job_id: str, wanted: str, timeout: float = 30.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.job(job_id)["state"] == wanted:
            return
        time.sleep(0.001)
    raise AssertionError(f"job {job_id} never reached {wanted}")


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

# Which verification journey must reject each bias.  ``overlay-opacity`` is
# caught by the pixel-exact geometry journey because that journey compares every
# pixel, not just the hole probes.
JOURNEY_FOR_BIAS = {
    "delayed-selection": "delayed-selection",
    "geometry-reuse": "geometry-reuse",
    "overlay-opacity": "geometry-reuse",
}


def run_controls(workdir: Path) -> dict:
    """Build every mutant, confirm the bias, then confirm the journey rejects it."""
    from run import JOURNEYS  # local import avoids a circular import at module load

    scratch = Path(tempfile.mkdtemp(prefix="prism-biases-"))
    controls = {}
    try:
        for name in ("delayed-selection", "geometry-reuse", "overlay-opacity"):
            binary, metadata = build_mutant(scratch, name)
            probe = probe_bias(name, binary, workdir)
            if not probe["bias_confirmed"]:
                raise AssertionError(f"{name}: mutant did not exhibit the intended bias: {probe}")
            # The journey must now fail; a pass would mean the judgement missed the bias.
            journey = JOURNEY_FOR_BIAS[name]
            server = harness.Server(binary, workdir / f"mutant-{name}")
            journey_failure = None
            try:
                JOURNEYS[journey](server, _SilentRecorder())
            except AssertionError as error:
                journey_failure = str(error)[:400]
            finally:
                server.stop()
            if journey_failure is None:
                raise AssertionError(
                    f"{journey} passed a {name} mutant; judgement ineffective")
            controls[name] = {**metadata, "journey": journey, "probe": probe,
                              "journey_rejected": True,
                              "journey_error": journey_failure}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return controls


class _SilentRecorder:
    def note(self, *_args, **_kwargs):
        return None
