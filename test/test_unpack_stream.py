"""Tests for streaming ``msgpack.unpack()`` from file-like objects."""

import io

import pytest

from msgpack import ExtraData, ExtType, Timestamp, Unpacker, packb, unpack


class ShortReader:
    """File-like object that never returns a full read() request.

    It deliberately has no ``seek``, ``tell`` or ``fileno`` method so a test
    fails if ``unpack()`` relies on them.
    """

    def __init__(self, data, chunk):
        self._data = memoryview(data)
        self._pos = 0
        self.chunk = chunk
        self.read_calls = []

    def read(self, n=-1):
        assert n is not None and n >= 0, "unbounded read is not allowed"
        self.read_calls.append(n)
        n = min(n, self.chunk)
        ret = self._data[self._pos : self._pos + n].tobytes()
        self._pos += len(ret)
        return ret

    def __getattr__(self, name):
        raise AssertionError("unexpected attribute: %s" % name)

    @property
    def consumed(self):
        return self._pos


SHORT_READ_CHUNKS = [1, 2, 3, 5, 7, 16, 4096]


def _large_payload():
    field = bytes(range(256)) * 64  # 16 KiB
    return [field + i.to_bytes(2, "big") for i in range(32)]


@pytest.mark.parametrize("chunk", SHORT_READ_CHUNKS)
def test_unpack_short_reading_stream(chunk):
    value = {
        "ints": list(range(64)),
        "bin": bytes(range(256)) * 3,
        "str": "héllo wörld",
        "ext": ExtType(42, b"payload"),
        "nested": [{"a": [1, 2]}, {"b": None}],
    }
    stream = ShortReader(packb(value), chunk)
    assert unpack(stream) == value
    # The object must be read completely, no more and no less.
    assert stream.consumed == len(packb(value))


class FileShortReader:
    """Wrap a real file, returning at most ``chunk`` bytes per read."""

    def __init__(self, f, chunk):
        self._f = f
        self.chunk = chunk

    def read(self, n=-1):
        assert n is not None and n >= 0
        return self._f.read(min(n, self.chunk))

    def __getattr__(self, name):
        raise AssertionError("unexpected attribute: %s" % name)


@pytest.mark.parametrize("chunk", SHORT_READ_CHUNKS)
def test_unpack_short_reading_file(chunk, tmp_path):
    value = _large_payload()
    path = tmp_path / "data.msgpack"
    path.write_bytes(packb(value))

    with path.open("rb") as f:
        assert unpack(FileShortReader(f, chunk)) == value


def test_unpack_from_real_file_without_seek_or_size(tmp_path):
    # Only read(n) is available: no seek, tell, fileno, or __len__.
    value = _large_payload()
    path = tmp_path / "single.msgpack"
    path.write_bytes(packb(value))

    class ReadOnlyStream:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            assert n is not None and n >= 0
            return self._f.read(n)

        def __getattr__(self, name):
            raise AssertionError("unexpected attribute: %s" % name)

    with path.open("rb") as f:
        assert unpack(ReadOnlyStream(f)) == value


def test_unpack_trailing_bytes_stay_in_stream_for_short_reads(tmp_path):
    # unpack() reports trailing data via ExtraData, including bytes the stream
    # returned beyond the object.
    first = {"key": b"x" * 1000}
    trailing = packb({"other": b"y" * 1000})
    path = tmp_path / "many.msgpack"
    path.write_bytes(packb(first) + trailing)

    with path.open("rb") as f:
        with pytest.raises(ExtraData) as exc_info:
            unpack(FileShortReader(f, 7))
    assert exc_info.value.unpacked == first
    assert exc_info.value.extra == trailing


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"raw": True},
        {"use_list": False},
        {"object_hook": lambda d: {"hooked": d}},
        {"object_pairs_hook": lambda pairs: dict(pairs)},
        {"ext_hook": lambda code, data: (code, data)},
        {"timestamp": 3},
    ],
)
def test_unpack_options_via_stream(kwargs):
    from msgpack import unpackb

    value = {"a": [1, 2], "b": Timestamp(1, 2)}
    packed = packb(value)
    # The stream path must apply options exactly like the bytes path.
    assert unpack(ShortReader(packed, 3), **kwargs) == unpackb(packed, **kwargs)


def test_unpack_extra_data_contains_all_trailing_bytes():
    first = packb(100)
    trailing = b"\xc0" * (300 * 1024)  # more than one read_size
    for chunk in (1, 13, 64 * 1024):
        stream = ShortReader(first + trailing, chunk)
        with pytest.raises(ExtraData) as exc_info:
            unpack(stream)
        assert exc_info.value.unpacked == 100
        assert exc_info.value.extra == trailing
        assert stream.consumed == len(first) + len(trailing)


def test_unpack_no_extra_data_at_clean_eof():
    assert unpack(ShortReader(packb({"a": 1}), 1)) == {"a": 1}


def test_unpack_incomplete_input_raises_value_error():
    for chunk in (1, 2, 5):
        with pytest.raises(ValueError):
            unpack(ShortReader(packb([1, 2, 3])[:-1], chunk))


def test_unpack_empty_stream_raises_value_error():
    for chunk in (1, 16):
        with pytest.raises(ValueError):
            unpack(ShortReader(b"", chunk))


def test_unpack_from_actual_file_and_short_reader_agree(tmp_path):
    value = _large_payload()
    packed = packb(value)
    path = tmp_path / "big.msgpack"
    path.write_bytes(packed)

    with path.open("rb") as f:
        from_file = unpack(f)
    from_short = unpack(ShortReader(packed, 7))
    assert from_file == value
    assert from_short == value


def _traced_peak(func):
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = func()
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    return result, peak


def test_unpack_does_not_keep_whole_input_with_short_reads():
    """While decoding a large object, resident input should stay bounded.

    The result itself is kept (as expected), but neither the whole serialized
    input nor the part already decoded may be retained alongside it. A stream
    that returns only a few bytes per read() exercises the read-ahead path.
    """
    field = bytes(range(256)) * 256  # 64 KiB per field
    payload = [field + i.to_bytes(4, "big") for i in range(128)]  # ~8 MiB
    packed = packb(payload)
    result_size = sum(len(item) for item in payload)

    for chunk in (4096, 64 * 1024):
        result, peak = _traced_peak(lambda: unpack(ShortReader(packed, chunk)))
        assert result == payload
        # The previous implementation copied the whole input up front, so its
        # peak was about result_size + len(packed) (~19 MiB here). Streaming
        # keeps only a small buffer on top of the result.
        assert peak < result_size + 2 * 1024 * 1024, (chunk, peak)


def test_unpacker_class_matches_unpack(tmp_path):
    value = _large_payload()
    packed = packb(value)
    assert Unpacker(ShortReader(packed, 5)).unpack() == value
    assert Unpacker(io.BytesIO(packed), read_size=5).unpack() == value
