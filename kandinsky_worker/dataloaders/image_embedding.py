import msgpack
import numpy as np
import torch


class ImageEmbedding:
    job_uuid: str
    dataset: str
    image_embedding: list
    negative_image_embedding: list

    def __init__(self, job_uuid, dataset, image_embedding, negative_image_embedding= None):
        self.job_uuid = job_uuid
        self.dataset = dataset
        self.image_embedding = image_embedding
        self.negative_image_embedding = negative_image_embedding


    def serialize(self):
        # Convert object to a dictionary
        return {
            "job_uuid": self.job_uuid,
            "dataset": self.dataset,
            "image_embedding": self.image_embedding,
            "negative_image_embedding": self.negative_image_embedding
        }

    @classmethod
    def deserialize(cls, data):
        # Convert dictionary back to object
        negative_image_embedding = None
        if "negative_image_embedding" in data:
            negative_image_embedding = data["negative_image_embedding"]

        return cls(data["job_uuid"],
                   data["dataset"],
                   torch.from_numpy(data["image_embedding"]),
                   torch.from_numpy(negative_image_embedding))

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