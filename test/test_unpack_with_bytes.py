"""Tests for Unpacker.unpack_with_bytes()."""

from io import BytesIO

from pytest import mark, raises

from msgpack import OutOfData, Packer, Unpacker, packb, unpackb
from msgpack.fallback import Unpacker as FallbackUnpacker

try:
    from msgpack._cmsgpack import Unpacker as CUnpacker
except ImportError:
    CUnpacker = None

# The same integer stored with different widths: re-packing would normalize
# these, so they are good probes that the exact input bytes are returned.
INT_WIDTH_RECORDS = [
    (b"\x01", 1),  # positive fixint
    (b"\xcc\x01", 1),  # uint 8
    (b"\xcd\x00\x01", 1),  # uint 16
    (b"\xce\x00\x00\x00\x01", 1),  # uint 32
    (b"\xcf\x00\x00\x00\x00\x00\x00\x00\x01", 1),  # uint 64
    (b"\xd0\x01", 1),  # int 8
    (b"\xd1\x00\x01", 1),  # int 16
    (b"\xd2\x00\x00\x00\x01", 1),  # int 32
    (b"\xd3\x00\x00\x00\x00\x00\x00\x00\x01", 1),  # int 64
]

# Map with a duplicate key: {"a": 1, "b": 2, "a": 3}.
DUP_MAP = b"\x83\xa1a\x01\xa1b\x02\xa1a\x03"
DUP_MAP_OBJ = {"a": 3, "b": 2}
DUP_MAP_PAIRS = [("a", 1), ("b", 2), ("a", 3)]

# Duplicate keys built through the Packer API, nested inside other values.
NESTED_DUP_RECORDS = [
    Packer().pack_map_pairs([("x", 1), ("x", 2)]),  # {"x": 1, "x": 2}
    b"\x91" + Packer().pack_map_pairs([(1, b"y"), (1, b"z")]),  # [{1: b"y", 1: b"z"}]
]
NESTED_DUP = b"".join(NESTED_DUP_RECORDS)


def collect(unpacker):
    """Read all remaining records with unpack_with_bytes()."""
    out = []
    while True:
        try:
            out.append(unpacker.unpack_with_bytes())
        except OutOfData:
            return out


def test_int_widths_keep_their_exact_bytes():
    stream = b"".join(raw for raw, _ in INT_WIDTH_RECORDS)
    unpacker = Unpacker()
    unpacker.feed(stream)
    for expected_raw, expected_obj in INT_WIDTH_RECORDS:
        obj, raw = unpacker.unpack_with_bytes()
        assert obj == expected_obj
        assert raw == expected_raw
        assert type(raw) is bytes
    with raises(OutOfData):
        unpacker.unpack_with_bytes()


def test_duplicate_keys_preserved_in_raw():
    unpacker = Unpacker()
    unpacker.feed(DUP_MAP)
    obj, raw = unpacker.unpack_with_bytes()
    assert obj == DUP_MAP_OBJ  # last value wins in a dict
    assert raw == DUP_MAP  # the bytes keep key order and the duplicate


def test_duplicate_keys_with_pairs_hook():
    unpacker = Unpacker(object_pairs_hook=list)
    unpacker.feed(DUP_MAP)
    obj, raw = unpacker.unpack_with_bytes()
    assert obj == DUP_MAP_PAIRS
    assert raw == DUP_MAP


def test_nested_duplicate_keys():
    unpacker = Unpacker(strict_map_key=False)
    unpacker.feed(NESTED_DUP)
    assert unpacker.unpack_with_bytes() == ({"x": 2}, NESTED_DUP_RECORDS[0])
    assert unpacker.unpack_with_bytes() == ([{1: b"z"}], NESTED_DUP_RECORDS[1])


def test_raw_bytes_concatenate_to_the_stream():
    records = [
        packb(None),
        packb(True),
        packb(0.5),
        packb("hello"),
        packb(b"\x00\x01"),
        packb([1, 2, 3]),
        packb({"k": "v"}),
        DUP_MAP,
        Packer().pack_ext_type(42, b"data"),
    ] + [raw for raw, _ in INT_WIDTH_RECORDS]
    unpacker = Unpacker()
    unpacker.feed(b"".join(records))
    got = collect(unpacker)
    assert [raw for _, raw in got] == records
    assert b"".join(raw for _, raw in got) == b"".join(records)
    for obj, raw in got:
        assert obj == unpackb(raw)


@mark.parametrize(
    "record",
    [raw for raw, _ in INT_WIDTH_RECORDS]
    + [DUP_MAP, NESTED_DUP_RECORDS[0], packb({"a": [1, 2, {"b": b"x"}]}), packb("hello")],
)
def test_split_record_at_every_position(record):
    # Any proper prefix of a record is incomplete; the record must be
    # delivered, with its exact bytes, once the rest has been fed.
    for cut in range(len(record)):
        unpacker = Unpacker()
        unpacker.feed(record[:cut])
        with raises(OutOfData):
            unpacker.unpack_with_bytes()
        unpacker.feed(record[cut:])
        obj, raw = unpacker.unpack_with_bytes()
        assert raw == record
        assert obj == unpackb(raw)
        with raises(OutOfData):
            unpacker.unpack_with_bytes()


def test_byte_by_byte_feed_leaves_next_record():
    records = [packb({"key": [1, 2, 3]}), DUP_MAP, packb("tail")]
    unpacker = Unpacker()
    got = []
    for i in range(len(b"".join(records))):
        unpacker.feed(b"".join(records)[i : i + 1])
        try:
            got.append(unpacker.unpack_with_bytes())
        except OutOfData:
            pass
    assert [raw for _, raw in got] == records
    assert got[1][0] == DUP_MAP_OBJ


def test_callback_called_once_when_record_is_delivered():
    delivered = []

    def callback(obj, raw):
        delivered.append((obj, raw))

    unpacker = Unpacker()
    cut = len(DUP_MAP) // 2
    unpacker.feed(DUP_MAP[:cut])
    with raises(OutOfData):
        unpacker.unpack_with_bytes(callback)
    assert delivered == []  # not delivered yet: the callback must not run

    unpacker.feed(DUP_MAP[cut:])
    ret = unpacker.unpack_with_bytes(callback)
    assert delivered == [(DUP_MAP_OBJ, DUP_MAP)]  # exactly once
    assert ret == delivered[0]

    with raises(OutOfData):
        unpacker.unpack_with_bytes(callback)
    assert delivered == [(DUP_MAP_OBJ, DUP_MAP)]  # and not again


def test_callback_called_once_per_record():
    delivered = []
    records = [packb(1), DUP_MAP, packb([2])]
    unpacker = Unpacker()
    unpacker.feed(b"".join(records))
    for _ in records:
        unpacker.unpack_with_bytes(lambda obj, raw: delivered.append((obj, raw)))
    assert [raw for _, raw in delivered] == records


def test_file_like_with_small_reads():
    records = [packb([1, 2]), packb("hello"), packb(None), packb(2**63 - 1), DUP_MAP]
    stream = b"".join(records)
    unpacker = Unpacker(BytesIO(stream), read_size=3)
    got = collect(unpacker)
    assert [raw for _, raw in got] == records
    assert unpacker.tell() == len(stream)


def test_file_like_partial_record_waits_for_more_data():
    record = DUP_MAP
    unpacker = Unpacker(BytesIO(record), read_size=2)
    obj, raw = unpacker.unpack_with_bytes()
    assert (obj, raw) == (DUP_MAP_OBJ, DUP_MAP)
    with raises(OutOfData):
        unpacker.unpack_with_bytes()


def test_mix_with_unpack_skip_tell_and_read_bytes():
    r1, r2, r3, r4 = packb(1), packb("two"), packb([3]), DUP_MAP
    stream = r1 + r2 + r3 + r4
    unpacker = Unpacker()
    unpacker.feed(stream)

    assert unpacker.unpack() == 1
    assert unpacker.tell() == len(r1)

    assert unpacker.unpack_with_bytes() == ("two", r2)
    assert unpacker.tell() == len(r1) + len(r2)

    unpacker.skip()
    assert unpacker.tell() == len(r1) + len(r2) + len(r3)

    assert unpacker.unpack_with_bytes() == (DUP_MAP_OBJ, r4)
    assert unpacker.tell() == len(stream)

    with raises(OutOfData):
        unpacker.unpack()


def test_mix_with_raw_reads_and_iteration():
    stream = packb(1) + b"RAW" + packb(2) + packb(3)
    unpacker = Unpacker()
    unpacker.feed(stream)
    assert unpacker.unpack_with_bytes() == (1, packb(1))
    assert unpacker.read_bytes(3) == b"RAW"
    assert unpacker.unpack_with_bytes() == (2, packb(2))
    assert next(unpacker) == 3
    assert unpacker.tell() == len(stream)


def test_resume_after_partial_plain_unpack():
    # A record started by unpack() can be completed by unpack_with_bytes()
    # and still yields the whole record's bytes.
    cut = 4
    unpacker = Unpacker()
    unpacker.feed(DUP_MAP[:cut])
    with raises(OutOfData):
        unpacker.unpack()
    unpacker.feed(DUP_MAP[cut:])
    assert unpacker.unpack_with_bytes() == (DUP_MAP_OBJ, DUP_MAP)


def test_read_array_header_then_elements_with_bytes():
    unpacker = Unpacker()
    unpacker.feed(b"\x93\x0a\x14\x1e")
    assert unpacker.read_array_header() == 3
    assert unpacker.unpack_with_bytes() == (10, b"\x0a")
    assert unpacker.unpack_with_bytes() == (20, b"\x14")
    assert unpacker.unpack_with_bytes() == (30, b"\x1e")


def test_decode_options_still_apply():
    unpacker = Unpacker(raw=True, use_list=False)
    packed = packb({b"k": [1, 2]})
    unpacker.feed(packed)
    obj, raw = unpacker.unpack_with_bytes()
    assert obj == {b"k": (1, 2)}
    assert raw == packed


def test_non_str_duplicate_map_keys():
    record = b"\x82\x01\x02\x01\x03"  # {1: 2, 1: 3}
    unpacker = Unpacker(strict_map_key=False)
    unpacker.feed(record)
    obj, raw = unpacker.unpack_with_bytes()
    assert obj == {1: 3}
    assert raw == record


def test_out_of_data_on_empty_input():
    unpacker = Unpacker()
    with raises(OutOfData):
        unpacker.unpack_with_bytes()
    unpacker.feed(b"")
    with raises(OutOfData):
        unpacker.unpack_with_bytes()


@mark.skipif(CUnpacker is None, reason="C extension not available")
def test_c_and_fallback_paths_agree():
    records = [
        packb(None),
        packb(0.5),
        packb("hello"),
        packb(b"\x00\x01"),
        packb([1, 2, 3]),
        DUP_MAP,
        Packer().pack_ext_type(42, b"data"),
    ] + [raw for raw, _ in INT_WIDTH_RECORDS]
    stream = b"".join(records)
    chunk_sizes = [1, 7, 2, 100, 3, 1, 50]

    results = {}
    for cls in (FallbackUnpacker, CUnpacker):
        unpacker = cls()
        got = []
        pos = 0
        for size in chunk_sizes:
            unpacker.feed(stream[pos : pos + size])
            pos += size
            got.extend(collect(unpacker))
        unpacker.feed(stream[pos:])
        got.extend(collect(unpacker))
        results[cls] = got

    assert results[FallbackUnpacker] == results[CUnpacker]
    assert [raw for _, raw in results[CUnpacker]] == records
