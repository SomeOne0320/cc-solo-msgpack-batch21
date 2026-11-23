#!/usr/bin/env python

import struct
import sys
from collections import OrderedDict
from io import BytesIO

import pytest

import msgpack
from msgpack import Packer, Unpacker, packb, unpackb


def check(data, use_list=False):
    re = unpackb(packb(data), use_list=use_list, strict_map_key=False)
    assert re == data


def testPack():
    test_data = [
        0,
        1,
        127,
        128,
        255,
        256,
        65535,
        65536,
        4294967295,
        4294967296,
        -1,
        -32,
        -33,
        -128,
        -129,
        -32768,
        -32769,
        -4294967296,
        -4294967297,
        1.0,
        b"",
        b"a",
        b"a" * 31,
        b"a" * 32,
        None,
        True,
        False,
        (),
        ((),),
        ((), None),
        {None: 0},
        (1 << 23),
    ]
    for td in test_data:
        check(td)


def testPackUnicode():
    test_data = ["", "abcd", ["defgh"], "Русский текст"]
    for td in test_data:
        re = unpackb(packb(td), use_list=1, raw=False)
        assert re == td
        packer = Packer()
        data = packer.pack(td)
        re = Unpacker(BytesIO(data), raw=False, use_list=1).unpack()
        assert re == td


def testPackBytes():
    test_data = [b"", b"abcd", (b"defgh",)]
    for td in test_data:
        check(td)


def testPackByteArrays():
    test_data = [bytearray(b""), bytearray(b"abcd"), (bytearray(b"defgh"),)]
    for td in test_data:
        check(td)


def testIgnoreUnicodeErrors():
    re = unpackb(packb(b"abc\xeddef", use_bin_type=False), raw=False, unicode_errors="ignore")
    assert re == "abcdef"


def testStrictUnicodeUnpack():
    packed = packb(b"abc\xeddef", use_bin_type=False)
    with pytest.raises(UnicodeDecodeError):
        unpackb(packed, raw=False, use_list=1)


def testIgnoreErrorsPack():
    re = unpackb(
        packb("abc\udc80\udcffdef", use_bin_type=True, unicode_errors="ignore"),
        raw=False,
        use_list=1,
    )
    assert re == "abcdef"


def testDecodeBinary():
    re = unpackb(packb(b"abc"), use_list=1)
    assert re == b"abc"


def testPackFloat():
    assert packb(1.0, use_single_float=True) == b"\xca" + struct.pack(">f", 1.0)
    assert packb(1.0, use_single_float=False) == b"\xcb" + struct.pack(">d", 1.0)


def testArraySize(sizes=[0, 5, 50, 1000]):
    bio = BytesIO()
    packer = Packer()
    for size in sizes:
        bio.write(packer.pack_array_header(size))
        for i in range(size):
            bio.write(packer.pack(i))

    bio.seek(0)
    unpacker = Unpacker(bio, use_list=1)
    for size in sizes:
        assert unpacker.unpack() == list(range(size))


def test_manualreset(sizes=[0, 5, 50, 1000]):
    packer = Packer(autoreset=False)
    for size in sizes:
        packer.pack_array_header(size)
        for i in range(size):
            packer.pack(i)

    bio = BytesIO(packer.bytes())
    unpacker = Unpacker(bio, use_list=1)
    for size in sizes:
        assert unpacker.unpack() == list(range(size))

    packer.reset()
    assert packer.bytes() == b""


def testMapSize(sizes=[0, 5, 50, 1000]):
    bio = BytesIO()
    packer = Packer()
    for size in sizes:
        bio.write(packer.pack_map_header(size))
        for i in range(size):
            bio.write(packer.pack(i))  # key
            bio.write(packer.pack(i * 2))  # value

    bio.seek(0)
    unpacker = Unpacker(bio, strict_map_key=False)
    for size in sizes:
        assert unpacker.unpack() == {i: i * 2 for i in range(size)}


def test_odict():
    seq = [(b"one", 1), (b"two", 2), (b"three", 3), (b"four", 4)]
    od = OrderedDict(seq)
    assert unpackb(packb(od), use_list=1) == dict(seq)

    def pair_hook(seq):
        return list(seq)

    assert unpackb(packb(od), object_pairs_hook=pair_hook, use_list=1) == seq


@pytest.mark.skipif(sys.version_info < (3, 15), reason="frozendict requires Python 3.15+")
def test_frozendict():
    import builtins

    fd = builtins.frozendict(a=1, b=[1, 2, 3])
    unpacked = unpackb(packb(fd), use_list=1)
    assert isinstance(unpacked, dict)
    assert unpacked == fd

    unpacked = unpackb(packb(fd), use_list=1, object_pairs_hook=builtins.frozendict)
    assert isinstance(unpacked, builtins.frozendict)
    assert unpacked == fd

    packer = Packer(strict_types=True)
    unpacked = unpackb(packer.pack(fd), use_list=1)
    assert isinstance(unpacked, dict)
    assert unpacked == fd


def test_pairlist():
    pairlist = [(b"a", 1), (2, b"b"), (b"foo", b"bar")]
    packer = Packer()
    packed = packer.pack_map_pairs(pairlist)
    unpacked = unpackb(packed, object_pairs_hook=list, strict_map_key=False)
    assert pairlist == unpacked


def test_get_buffer():
    packer = Packer(autoreset=0, use_bin_type=True)
    packer.pack([1, 2])
    strm = BytesIO()
    strm.write(packer.getbuffer())
    written = strm.getvalue()

    expected = packb([1, 2], use_bin_type=True)
    assert written == expected


@pytest.mark.skipif(
    Packer.__module__ == "msgpack.fallback",
    reason="buf_size only allocates in the C extension",
)
def test_buf_size_is_converted_once():
    # Asking twice let the allocation and the recorded capacity disagree,
    # so the packer overflowed a buffer smaller than the size it recorded.
    class Counting:
        count = 0

        def __int__(self):
            self.count += 1
            return 600

        __index__ = __int__

    buf_size = Counting()
    Packer(buf_size=buf_size)
    assert buf_size.count == 1


class Unpackable:
    """An object that can not be serialized and is not handled by default."""


def unpack_all(packed):
    unpacker = Unpacker(raw=False, strict_map_key=False)
    unpacker.feed(packed)
    return list(unpacker)


def test_failed_pack_keeps_earlier_messages():
    packer = Packer(autoreset=False)
    packer.pack(1)
    with pytest.raises(TypeError):
        packer.pack([1, Unpackable()])
    # Only this call's partial object is undone; the earlier message survives.
    assert packer.bytes() == packb(1)
    packer.pack(2)
    assert unpack_all(packer.bytes()) == [1, 2]


def test_failed_pack_autoreset_returns_no_partial_message():
    packer = Packer()
    assert packer.pack(1) == packb(1)
    with pytest.raises(TypeError):
        packer.pack([1, Unpackable()])
    assert packer.bytes() == b""
    assert packer.pack(2) == packb(2)


def test_failed_pack_pairs_rolls_back_whole_map():
    packer = Packer(autoreset=False)
    packer.pack("before")
    with pytest.raises(TypeError):
        packer.pack_map_pairs([("a", 1), ("b", Unpackable())])
    # The map header and the first pair must not be left behind.
    assert packer.bytes() == packb("before")
    packer.pack_map_pairs([("a", 1)])
    assert unpack_all(packer.bytes()) == ["before", {"a": 1}]


def test_failed_pack_map_pairs_autoreset():
    packer = Packer()
    with pytest.raises(TypeError):
        packer.pack_map_pairs([("a", Unpackable())])
    assert packer.bytes() == b""
    assert unpackb(packer.pack_map_pairs([("a", 1)])) == {"a": 1}


def test_failed_header_and_ext_calls_roll_back():
    packer = Packer(autoreset=False)
    packer.pack("keep")
    with pytest.raises(ValueError):
        packer.pack_array_header(-1)
    with pytest.raises(ValueError):
        packer.pack_map_header(-1)
    if Packer.__module__ == "msgpack.fallback":
        with pytest.raises(TypeError):
            packer.pack_ext_type(1, "not bytes")
    assert packer.bytes() == packb("keep")
    packer.pack_ext_type(0x42, b"A")
    assert unpack_all(packer.bytes()) == ["keep", msgpack.ExtType(0x42, b"A")]


def test_failed_pack_preserves_default_error():
    class CustomError(Exception):
        pass

    def default(obj):
        raise CustomError("from default")

    packer = Packer(autoreset=False, default=default)
    packer.pack(1)
    with pytest.raises(CustomError, match="from default"):
        packer.pack(Unpackable())
    assert packer.bytes() == packb(1)
    packer.pack(2)
    assert unpack_all(packer.bytes()) == [1, 2]
