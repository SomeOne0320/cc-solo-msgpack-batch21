# ruff: noqa: F401
import os

from .exceptions import *  # noqa: F403
from .exceptions import ExtraData, OutOfData
from .ext import ExtType, Timestamp

version = (1, 2, 4)
__version__ = "1.2.4"


if os.environ.get("MSGPACK_PUREPYTHON"):
    from .fallback import Packer, Unpacker, unpackb
else:
    try:
        from ._cmsgpack import Packer, Unpacker, unpackb
    except ImportError:
        from .fallback import Packer, Unpacker, unpackb


def pack(o, stream, **kwargs):
    """
    Pack object `o` and write it to `stream`

    See :class:`Packer` for options.
    """
    packer = Packer(**kwargs)
    stream.write(packer.pack(o))


def packb(o, **kwargs):
    """
    Pack object `o` and return packed bytes

    See :class:`Packer` for options.
    """
    return Packer(**kwargs).pack(o)


_UNPACK_READ_SIZE = 64 * 1024


def unpack(stream, **kwargs):
    """
    Unpack an object from `stream`.

    The stream is read in chunks as the object is decoded, so the whole
    serialized message does not have to be kept in memory. Only ``read(n)``
    is used; the stream is not required to support ``seek()`` or to report
    its size, and ``read(n)`` may return fewer than *n* bytes.

    Raises `ExtraData` when `stream` contains extra bytes. ``ExtraData.extra``
    contains all bytes that follow the unpacked object.
    See :class:`Unpacker` for options.
    """
    # unpackb() derives the length limits from the size of the packed data.
    # The size of a stream is unknown, so do not impose the default buffer
    # limit either; a single object can be as large as unpackb() allowed.
    max_buffer_size = kwargs.setdefault("max_buffer_size", 0)
    # Read in modest chunks so reading ahead does not hold a large part of the
    # input. A small max_buffer_size still gets a valid (smaller) read_size.
    kwargs.setdefault("read_size", min(_UNPACK_READ_SIZE, max_buffer_size or 2**31 - 1))
    unpacker = Unpacker(stream, **kwargs)
    try:
        ret = unpacker.unpack()
    except OutOfData:
        raise ValueError("Unpack failed: incomplete input")
    # Bytes after the object that were read ahead are still buffered. Collect
    # them first, then drain the rest of the stream so ExtraData.extra holds
    # every trailing byte.
    chunks = []
    first = unpacker._get_extradata()
    if first:
        chunks.append(first)
    while True:
        data = stream.read(_UNPACK_READ_SIZE)
        if not data:
            break
        chunks.append(data)
    if chunks:
        raise ExtraData(ret, b"".join(chunks))
    return ret


# alias for compatibility to simplejson/marshal/pickle.
load = unpack
loads = unpackb

dump = pack
dumps = packb
