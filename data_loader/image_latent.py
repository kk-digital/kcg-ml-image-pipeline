import msgpack
import numpy as np

class LatentData:
    job_uuid: str
    file_hash: str
    latent_vector: list

    def __init__(self, job_uuid, file_hash, latent_vector):
        self.job_uuid = job_uuid
        self.file_hash = file_hash
        self.latent_vector = latent_vector

    def serialize(self):
        # Convert object to a dictionary
        return {
            "job_uuid": self.job_uuid,
            "file_hash": self.file_hash,
            "latent_vector": self.latent_vector,
        }

    @classmethod
    def deserialize(cls, data):
        # Convert dictionary back to object
        return cls(data["job_uuid"],
                   data["file_hash"],
                   data["latent_vector"])

    def get_msgpack_string(self):
        serialized = self.serialize()
        return msgpack.packb(serialized, default=encode_ndarray, use_bin_type=True, use_single_float=True)

    @classmethod
    def from_msgpack_string(cls, msgpack_string):
        data = msgpack.unpackb(msgpack_string.encode('latin1'), object_hook=decode_ndarray, raw=False)
        return cls.deserialize(data)

    @classmethod
    def from_msgpack_bytes(cls, msgpack_string):
        data = msgpack.unpackb(msgpack_string, object_hook=decode_ndarray, raw=False)
        return cls.deserialize(data)

def encode_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return {'__ndarray__': obj.tolist()}
    return obj

def decode_ndarray(packed_obj):
    if '__ndarray__' in packed_obj:
        return np.array(packed_obj['__ndarray__'])
    return packed_obj
