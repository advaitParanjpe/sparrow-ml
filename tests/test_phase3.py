import numpy as np
import pytest

from sparrowml.sparsity.integer_reference import infer_sparse_compressed, infer_sparse_dense
from sparrowml.sparsity.metadata import decode_metadata, encode_lanes
from sparrowml.sparsity.packing import pack_metadata, unpack_metadata
from sparrowml.sparsity.pruning import compress_weights, decompress_weights, prune_2of4


def test_pruning_is_2of4_deterministic_and_tie_stable():
    weights = np.asarray([[5, -9, 9, 2, 4, -4, 4, -4], [-128, 127, 0, -1, 0, 0, 0, 0]], dtype=np.int8)
    sparse, mask = prune_2of4(weights); again, again_mask = prune_2of4(weights)
    assert np.array_equal(sparse, again) and mask == again_mask
    assert mask[0]["selected_lanes"] == [1, 2]
    assert mask[1]["selected_lanes"] == [0, 1]
    assert all(sum(item["binary_mask"]) == 2 for item in mask)
    assert all(np.count_nonzero(sparse[row, group : group + 4]) <= 2 for row in range(2) for group in (0, 4))


def test_metadata_compression_decompression_and_packing_contracts():
    assert [decode_metadata(value) for value in range(6)] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert [encode_lanes(decode_metadata(value)) for value in range(6)] == list(range(6))
    with pytest.raises(ValueError): decode_metadata(6)
    values = [0, 1, 2, 3, 4, 5] * 2 + [0, 1, 2, 3]
    payload = pack_metadata(values)
    assert len(payload) == 6 and unpack_metadata(payload, 16) == values
    with pytest.raises(ValueError): unpack_metadata(bytes([0xFF]), 1)
    dense = np.asarray([[5, -9, 9, 2]], dtype=np.int8); sparse, mask = prune_2of4(dense); compressed, metadata = compress_weights(sparse, mask)
    assert compressed.tolist() == [[-9, 9]] and np.array_equal(decompress_weights(compressed, metadata, dense.shape), sparse)


def test_compressed_reference_executes_selected_lanes_without_decompression():
    sparse = np.asarray([[2, 0, -3, 0]], dtype=np.int8); compressed = np.asarray([[2, -3]], dtype=np.int8)
    dense = infer_sparse_dense(np.asarray([4, 9, -5, 7], dtype=np.int8), sparse, np.asarray([1], dtype=np.int32), 0.5, np.asarray([0.25]), ("x",))
    packed = infer_sparse_compressed(np.asarray([4, 9, -5, 7], dtype=np.int8), compressed, [1], np.asarray([1], dtype=np.int32), 0.5, np.asarray([0.25]), ("x",), 4)
    assert dense.accumulators.tolist() == [24] and np.array_equal(dense.accumulators, packed.accumulators)
