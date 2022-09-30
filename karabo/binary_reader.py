from __future__ import annotations
from ctypes import c_char, c_char_p, c_uint8, c_uint32, c_uint64, LittleEndianStructure
import struct
import sys
import typing as tp
import mmap
import crc32c

class PackedBinaryStructure(LittleEndianStructure):

    @classmethod
    def create_from_bytes(cls, byte_str: bytes) -> PackedBinaryStructure:
        """Use format string representation (struct) to unpack binary data.

        Mapping of ctypes.Structures (CField) to struct
        https://docs.python.org/3/library/struct.html#format-strings
        """
        frmt = ['<']
        for f_name, f_ctype in cls._fields_:
            if hasattr(f_ctype, "_length_"): # array type
                frmt_type = f"{f_ctype._length_}s"
            else:
                frmt_type = f_ctype._type_
            if frmt_type == 'L' and getattr(cls, f_name).size == 8:
                frmt_type = 'Q'
            elif frmt_type == 'l' and getattr(cls, f_name).size == 8:
                frmt_type = 'q'
            frmt.append(frmt_type)
        frmt_str = ''.join(frmt)
        print(f"Unpack {cls.__name__} with format {frmt_str}")
        data = struct.unpack(frmt_str, byte_str)
        return cls(*data)

    @staticmethod
    def popCount(byteInput: int) -> int:
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 10:
            return PackedBinaryStructure.fastPopCount(byteInput)
        else:
            return PackedBinaryStructure.legacyPopCount(byteInput)

    @staticmethod
    def fastPopCount(byteInput: int) -> int:
        return byteInput.bit_count()

    @staticmethod
    def legacyPopCount(byteInput: int) -> int:
        count: int = 0
        while(byteInput > 0):
            count += byteInput & 0b1
            byteInput >>= 1
        return count

    def __str__(self):
        as_dict = {f_name: self.__getattribute__(f_name) for f_name, _ in self._fields_}
        return f"{type(self).__name__}({as_dict})"

class FileHeader(PackedBinaryStructure):
    _fields_ = [
        ("desc", c_char * 9),
        ("ver", c_uint8),
        ("littleEndinan", c_uint8),
        ("voidptrSize", c_uint8),
        ("intSize", c_uint8),
        ("longSize", c_uint8),
        ("floatSize", c_uint8),
        ("doubleSize", c_uint8),
        ("oskarVersion", c_uint32),
        ("reserved0", c_uint64), # python struct unpack() can NOT handle
        ("reserved1", c_uint64), # a 44byte c_uint8 array. Structures
        ("reserved2", c_uint64), # expects a bytearray, but unpack()
        ("reserved3", c_uint64), # delivers single ints (b) or a byteString (c).
        ("reserved4", c_uint64), # Hence we split to 5xQuad and 1xInt.
        ("reserved5", c_uint32)] # All these value must be 0 anyway.

class Tag(PackedBinaryStructure):
    _fields_ = [
        ("t", c_char),
        ("AorB", c_char),
        ("g", c_char),
        ("payloadElementSize", c_uint8),
        ("chunkFlags", c_uint8),
        ("payloadDataType", c_uint8),
        ("groupID", c_uint8),
        ("tagID", c_uint8),
        ("userIdx", c_uint32),
        ("payloadSize", c_uint64)]

    def isPayloadBigEndian(self) -> bool:
        return self.chunkFlags & 0b0001_0000

    def usesCRC32CEncoding(self) -> bool:
        return self.chunkFlags & 0b0100_0000
        
    def isTagExtended(self) -> bool:
        return self.chunkFlags & 0b1000_0000

    def isCharType(self) -> bool:
        return (self.payloadDataType & 0b0000_0001) != 0

    def isIntType(self) -> bool:
        return (self.payloadDataType & 0b0000_0010) != 0

    def isFloatType(self) -> bool:
        return (self.payloadDataType & 0b0000_0100) != 0

    def isDoubleType(self) -> bool:
        return (self.payloadDataType & 0b0000_1000) != 0

    def isSane(self) -> bool:
        return (self.payloadDataType & 0b1001_0000) == 0 and \
            PackedBinaryStructure.popCount(self.payloadDataType & 0b0000_1111) == 1
    
    def isMatrix(self) -> bool:
        return self.payloadDataType & 0b0100_0000

    def isComplex(self) -> bool:
        return self.payloadDataType & 0b0010_0000

    def printPayloadDataType(self) -> str:
        if not self.isSane():
            assert False, "Typedescription bits that should be 0 are not"
        
        txtDescription: str = ""
        
        if self.isCharType():
            txtDescription += "1 byte char type "
        elif self.isIntType():
            txtDescription += "4 byte integer type "
        elif self.isFloatType():
            txtDescription += "4 byte float type "
        elif self.isDoubleType():
            txtDescription += "8 byte double type "

        if self.isComplex():
            txtDescription += "complex data, first real, than imaginary "
        if self.isMatrix():
            txtDescription += "2x2 matrix data, [ab, cd] "

        return txtDescription

class OskarBinaryReader:        

    HEADER_SIZE = 64
    TAG_SIZE    = 20
    
    @staticmethod
    def interpreteString(payload: bytes) -> str:
        return str(payload, "utf-8")
    
    @staticmethod
    def interpreteNumbers(payload: bytes, isMatrix: bool, isComplex: bool) -> tp.List[any]:
        pass

    @staticmethod
    def _readCRCFromData(payload: bytes) -> int:
        crcBytes = payload[-4:]
        return struct.unpack("i", crcBytes)[0]

    @staticmethod
    def checkCRC(hasCRC: bool, rawTag: bytes, rawPayload: bytes) -> bool:
        if not hasCRC:
            print("old version - no CRC to check")
            return True

        print("check CRC...")

        data = rawTag + rawPayload
        rawPayload = rawPayload[:-4] # remove CRC from payload

        # calculate crc from data
        calculatedCRC: int = crc32c.crc32c(data[:-4])
        # read written crc
        readCRC: int = OskarBinaryReader._readCRCFromData(data)
        # check if equals
        return readCRC == calculatedCRC, rawPayload
        
    def interpretePayloadData(self, payload: bytes, tag: Tag):
        
        if tag.isCharType():
            return OskarBinaryReader.interpreteString(payload)
        else:
            return OskarBinaryReader.interpreteNumbers(payload, isMatrix=tag.isMatrix(), isComplex=tag.isComplex())
        
    def readBinaryFile(self, path : str): # -> TBD:
        print(f"open {path} as oskar binary file")  

        buff: mmap

        with open(path, mode="r+b") as f:
            buff = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

        # data = f.read(OskarBinaryReader.HEADER_SIZE)
        # header = FileHeader.create_from_bytes(data)
        
        header1 = FileHeader.from_buffer(buff[0:OskarBinaryReader.HEADER_SIZE])

        # rawTag = f.read(OskarBinaryReader.TAG_SIZE)
            # tag = Tag.create_from_bytes(rawTag)
            # rawPayload = f.read(tag.payloadSize)

            # crcIO, rawPayload = OskarBinaryReader.checkCRC(header.ver >= 2, rawTag, rawPayload)
            # assert crcIO, "crc wrong"

            # print(header.desc)
            # print(tag.printPayloadDataType())
           
            # payload = self.interpretePayloadData(rawPayload, tag)

            # print(payload)



r = OskarBinaryReader()
r.readBinaryFile("/home/filip/element_pattern_fit_x_0_100.bin")

