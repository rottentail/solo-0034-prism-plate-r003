"""Independent pixel oracle and PNG reader for prism-plate verification.

Deliberately shares no code with the Go server.  It re-implements the published
scene contract (inverse affine mapping, fixed Q/C subdivision, even-odd/nonzero
containment, linear-light premultiplied compositing, 2x2 coverage sampling) and
a strict PNG reader, so an image can be judged without trusting artifact
metadata, job state, ETags or the server's own digests.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib

from scenes import PROBES  # noqa: F401 - journeys import PROBES from here


# ---------------------------------------------------------------------------
# geometry / paint
# ---------------------------------------------------------------------------

def _linear(v: float) -> float:
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def _srgb(v: float) -> float:
    v = max(0.0, min(1.0, v))
    if v <= 0.0031308:
        return v * 12.92
    return 1.055 * (v ** (1 / 2.4)) - 0.055


def _go_round(v: float) -> int:
    # Go's math.Round: half away from zero.  All inputs here are >= 0.
    return int(math.floor(v + 0.5))


def _from_color(color: dict):
    alpha = color["a"]
    return (_linear(color["r"]) * alpha, _linear(color["g"]) * alpha,
            _linear(color["b"]) * alpha, alpha)


def _scale(pixel, factor):
    return (pixel[0] * factor, pixel[1] * factor, pixel[2] * factor, pixel[3] * factor)


def _over(dst, src, mode: str):
    if mode == "normal" or src[3] == 0 or dst[3] == 0:
        def blend(d, s):
            return s + d * (1 - src[3])
    else:
        def blend(d, s):
            a = d / dst[3]
            b = s / src[3]
            v = a * b
            if mode == "screen":
                v = a + b - a * b
            return (1 - src[3]) * d + (1 - dst[3]) * s + src[3] * dst[3] * v
    return (blend(dst[0], src[0]), blend(dst[1], src[1]), blend(dst[2], src[2]),
            src[3] + dst[3] * (1 - src[3]))


def _rgba8(pixel):
    if pixel[3] <= 0:
        return (0, 0, 0, 0)
    return (
        _go_round(_srgb(pixel[0] / pixel[3]) * 255),
        _go_round(_srgb(pixel[1] / pixel[3]) * 255),
        _go_round(_srgb(pixel[2] / pixel[3]) * 255),
        _go_round(min(1.0, pixel[3]) * 255),
    )


def flatten(ops: list[dict]) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    ring: list[tuple[float, float]] = []
    at = None

    def flush():
        nonlocal ring
        if len(ring) >= 3:
            out.append(ring)
        ring = []

    for op in ops:
        kind = op["kind"]
        points = [(point["x"], point["y"]) for point in op.get("points", [])]
        if kind == "M":
            flush()
            at = points[0]
            ring = [at]
        elif kind == "L":
            at = points[0]
            ring.append(at)
        elif kind == "Z":
            flush()
        elif kind == "Q":
            start, control, end = at, points[0], points[1]
            for i in range(1, 17):
                t = i / 16
                u = 1 - t
                ring.append((u * u * start[0] + 2 * u * t * control[0] + t * t * end[0],
                             u * u * start[1] + 2 * u * t * control[1] + t * t * end[1]))
            at = end
        elif kind == "C":
            start, a, b, end = at, points[0], points[1], points[2]
            for i in range(1, 25):
                t = i / 24
                u = 1 - t
                ring.append((
                    u * u * u * start[0] + 3 * u * u * t * a[0] + 3 * u * t * t * b[0] + t * t * t * end[0],
                    u * u * u * start[1] + 3 * u * u * t * a[1] + 3 * u * t * t * b[1] + t * t * t * end[1]))
            at = end
        else:
            raise ValueError(f"unknown contour verb {kind}")
    flush()
    return out


def contains(path, point, rule: str) -> bool:
    winding = 0
    crossings = 0
    for ring in path:
        count = len(ring)
        for i in range(count):
            a = ring[i]
            b = ring[(i + 1) % count]
            if (a[1] <= point[1] and b[1] > point[1]) or (b[1] <= point[1] and a[1] > point[1]):
                x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x > point[0]:
                    crossings += 1
                    winding += 1 if b[1] > a[1] else -1
    return crossings % 2 == 1 if rule == "evenodd" else winding != 0


def _inverse(matrix):
    det = matrix[0] * matrix[3] - matrix[1] * matrix[2]
    if abs(det) < 1e-10:
        raise ValueError("singular transform")
    return (matrix[3] / det, -matrix[1] / det, -matrix[2] / det, matrix[0] / det,
            (matrix[2] * matrix[5] - matrix[3] * matrix[4]) / det,
            (matrix[1] * matrix[4] - matrix[0] * matrix[5]) / det)


def _apply(matrix, point):
    return (matrix[0] * point[0] + matrix[2] * point[1] + matrix[4],
            matrix[1] * point[0] + matrix[3] * point[1] + matrix[5])


def _sample_brush(brush: dict, point):
    if brush["kind"] == "solid":
        return _from_color(brush["color"])
    dx = brush["to"]["x"] - brush["from"]["x"]
    dy = brush["to"]["y"] - brush["from"]["y"]
    denominator = dx * dx + dy * dy
    t = ((point[0] - brush["from"]["x"]) * dx +
         (point[1] - brush["from"]["y"]) * dy) / denominator
    t = max(0.0, min(1.0, t))
    stops = brush["stops"]
    for i in range(1, len(stops)):
        low, high = stops[i - 1], stops[i]
        if t <= high["at"]:
            u = (t - low["at"]) / (high["at"] - low["at"])
            a, b = _from_color(low["color"]), _from_color(high["color"])
            return tuple(a[j] + (b[j] - a[j]) * u for j in range(4))
    return _from_color(stops[-1]["color"])


class CompiledDocument:
    """Structural compilation mirroring the published scene contract."""

    def __init__(self, document: dict):
        self.document = document
        self.width = document["width"]
        self.height = document["height"]
        self.background = document["background"]
        self.layers = []
        for layer in document["layers"]:
            compiled = {
                "opacity": layer.get("opacity", 0.0),
                "blend": layer.get("blend", ""),
                "clip": flatten(layer["clip"]) if layer.get("clip") else None,
                "mask": flatten(layer["mask"]) if layer.get("mask") else None,
                "shapes": [],
            }
            for shape in layer["shapes"]:
                compiled["shapes"].append({
                    "path": flatten(shape["path"]),
                    "rule": shape["fill_rule"],
                    "brush": shape["brush"],
                    "inverse": _inverse(shape["transform"]),
                })
            self.layers.append(compiled)

    def pixel(self, x: int, y: int):
        total = [0.0, 0.0, 0.0, 0.0]
        for sample_y in range(2):
            for sample_x in range(2):
                point = (x + (sample_x + 0.5) / 2.0, y + (sample_y + 0.5) / 2.0)
                dst = _from_color(self.background)
                for layer in self.layers:
                    if layer["clip"] is not None and not contains(layer["clip"], point, "nonzero"):
                        continue
                    if layer["mask"] is not None and not contains(layer["mask"], point, "nonzero"):
                        continue
                    src = (0.0, 0.0, 0.0, 0.0)
                    for shape in layer["shapes"]:
                        local = _apply(shape["inverse"], point)
                        if contains(shape["path"], local, shape["rule"]):
                            src = _over(src, _sample_brush(shape["brush"], local), "normal")
                    dst = _over(dst, _scale(src, layer["opacity"]), layer["blend"])
                for channel in range(4):
                    total[channel] += dst[channel] / 4.0
        return _rgba8(total)

    def rgb_at(self, x: int, y: int):
        red, green, blue, alpha = self.pixel(x, y)
        if alpha != 255:
            raise ValueError(f"opaque journey expected; alpha {alpha} at {x},{y}")
        return (red, green, blue)

    def render_rgb(self) -> bytes:
        buffer = bytearray(self.width * self.height * 3)
        offset = 0
        for y in range(self.height):
            for x in range(self.width):
                buffer[offset:offset + 3] = bytes(self.rgb_at(x, y))
                offset += 3
        return bytes(buffer)

    def probe_values(self) -> dict:
        return {name: self.rgb_at(x, y) for name, (x, y) in PROBES.items()}


# ---------------------------------------------------------------------------
# PNG reader
# ---------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(ValueError):
    pass


def decode_png(raw: bytes) -> dict:
    if raw[:8] != PNG_SIGNATURE:
        raise PngError("bad PNG signature")
    if len(raw) < 8:
        raise PngError("truncated PNG")
    position = 8
    compressed = bytearray()
    header = None
    saw_iend = False
    chunk_count = 0
    while position < len(raw):
        if position + 12 > len(raw):
            raise PngError("truncated chunk header")
        length = struct.unpack(">I", raw[position:position + 4])[0]
        chunk_type = raw[position + 4:position + 8]
        if position + 12 + length > len(raw):
            raise PngError("truncated chunk body")
        data = raw[position + 8:position + 8 + length]
        stored_crc = struct.unpack(">I", raw[position + 8 + length:position + 12 + length])[0]
        if stored_crc != zlib.crc32(chunk_type + data) & 0xFFFFFFFF:
            raise PngError(f"bad CRC in {chunk_type!r}")
        chunk_count += 1
        if chunk_type == b"IHDR":
            if header is not None:
                raise PngError("duplicate IHDR")
            header = struct.unpack(">IIBBBBB", data)
        elif chunk_type == b"IDAT":
            compressed += data
        elif chunk_type == b"IEND":
            saw_iend = True
            position += 12 + length
            break
        position += 12 + length
    if header is None or not saw_iend:
        raise PngError("missing IHDR or IEND")
    if position != len(raw):
        raise PngError("trailing bytes after IEND")
    width, height, depth, color_type, compression, filter_method, interlace = header
    if depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise PngError(f"unsupported PNG encoding {header}")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise PngError(f"unsupported colour type {color_type}")
    try:
        raw_rows = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise PngError(f"IDAT is not valid zlib: {error}") from error
    stride = width * channels
    if len(raw_rows) != height * (stride + 1):
        raise PngError("decompressed size does not match the header dimensions")
    output = bytearray()
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw_rows[cursor]
        cursor += 1
        line = bytearray(raw_rows[cursor:cursor + stride])
        cursor += stride
        if filter_type == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = previous[i]
                c = previous[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + predictor) & 0xFF
        elif filter_type != 0:
            raise PngError(f"unknown filter {filter_type}")
        output += line
        previous = line
    return {"width": width, "height": height, "channels": channels,
            "color_type": color_type, "chunks": chunk_count, "pixels": bytes(output)}


def png_rgb(png: dict) -> bytes:
    if png["channels"] == 3:
        return png["pixels"]
    if png["channels"] != 4:
        raise PngError(f"unsupported channel count {png['channels']}")
    count = png["width"] * png["height"]
    output = bytearray(count * 3)
    pixels = png["pixels"]
    for i in range(count):
        red, green, blue, alpha = pixels[i * 4:i * 4 + 4]
        if alpha != 255:
            raise PngError(f"non-opaque sample at pixel {i}: alpha={alpha}")
        output[i * 3:i * 3 + 3] = bytes((red, green, blue))
    return bytes(output)


def png_probe(png: dict, x: int, y: int):
    if not (0 <= x < png["width"] and 0 <= y < png["height"]):
        raise PngError("probe outside image")
    offset = (y * png["width"] + x) * png["channels"]
    return tuple(png["pixels"][offset:offset + 3])


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def image_difference(actual: bytes, expected: bytes) -> dict:
    if len(actual) != len(expected):
        return {"pixels": -1, "samples": [], "reason": "length mismatch"}
    count = 0
    samples = []
    for offset in range(0, len(actual), 3):
        if actual[offset:offset + 3] != expected[offset:offset + 3]:
            count += 1
            if len(samples) < 8:
                samples.append({"offset": offset // 3,
                                "actual": list(actual[offset:offset + 3]),
                                "expected": list(expected[offset:offset + 3])})
    return {"pixels": count, "samples": samples}
