"""完整 GGUF 来源的失败边界；不使用大权重或 GPU。"""
import struct
import tempfile
from pathlib import Path
import unittest
from freetoken.models.qwen4_exp.gguf_source import GGUFShard, GGUFModelSource, map_tensor


def string(value):
    data=value.encode()
    return struct.pack("<Q",len(data))+data


def write_shard(path,metadata,tensors,body):
    data=struct.pack("<IIQQ",0x46554747,3,len(tensors),len(metadata))
    for key,value in metadata.items():
        data+=string(key)
        data+=struct.pack("<I",8)+string(value) if isinstance(value,str) else struct.pack("<II",4,value)
    for name,shape,kind,offset in tensors:
        data+=string(name)+struct.pack("<I",len(shape))+struct.pack("<"+"Q"*len(shape),*shape)+struct.pack("<IQ",kind,offset)
    data+=bytes((-len(data))%32)+body
    path.write_bytes(data)


class CompleteGGUFSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def pair(self):
        primary=self.root/"a.gguf";ple=self.root/"b.gguf"
        meta={"general.architecture":"qwen4exp","split.count":2,"split.no":0,
              "split.tensors.count":2,"qwen4exp.embedding_length":64}
        write_shard(primary,meta,[("token_embd.weight",(64,2),42,0)],bytes(range(36)))
        write_shard(ple,{"split.count":2,"split.no":1,"split.tensors.count":2},
                    [("per_layer_token_embd.weight",(32,3),20,0)],bytes(54))
        return primary,ple

    def test_secondary_split_can_omit_architecture_and_order_is_irrelevant(self):
        a,b=self.pair()
        source=GGUFModelSource([b,a])
        self.assertEqual(source.metadata["qwen4exp.embedding_length"],64)
        self.assertEqual(source.tensors["token_embd.weight"].read_rows(1,1),bytes(range(18,36)))
        self.assertEqual(source.tensors["per_layer_token_embd.weight"].row_bytes,18)

    def test_missing_or_duplicate_split_is_rejected(self):
        a,b=self.pair()
        for paths in ([a],[b],[a,a]):
            with self.subTest(paths=paths),self.assertRaises(ValueError):
                GGUFModelSource(paths)

    def test_conflicting_split_geometry_is_rejected(self):
        a,b=self.pair()
        write_shard(b,{"split.count":2,"split.no":1,"split.tensors.count":2,"qwen4exp.embedding_length":128},
                    [("per_layer_token_embd.weight",(32,3),20,0)],bytes(54))
        with self.assertRaisesRegex(ValueError,"geometry mismatch"):
            GGUFModelSource([a,b])

    def test_actual_tensor_count_must_match_declaration(self):
        a,b=self.pair()
        write_shard(b,{"split.count":2,"split.no":1,"split.tensors.count":2},[],b"")
        with self.assertRaisesRegex(ValueError,"tensor count mismatch"):
            GGUFModelSource([a,b])

    def test_overlapping_nonexpert_tensors_are_rejected(self):
        path=self.root/"overlap.gguf"
        write_shard(path,{},[("x",(32,2),20,0),("y",(32,2),20,32)],bytes(68))
        with self.assertRaisesRegex(ValueError,"overlapping"):
            GGUFShard(path)

    def test_truncation_misalignment_and_block_width_are_rejected(self):
        cases=[((64,2),42,0,bytes(35)),((64,1),42,1,bytes(32)),((63,1),42,0,bytes(32))]
        for index,(shape,kind,offset,body) in enumerate(cases):
            path=self.root/f"bad-{index}.gguf"
            write_shard(path,{},[("x",shape,kind,offset)],body)
            with self.subTest(index=index),self.assertRaises(ValueError):
                GGUFShard(path)

    def test_row_read_cannot_cross_tensor_boundary(self):
        a,b=self.pair();tensor=GGUFModelSource([a,b]).tensors["token_embd.weight"]
        for first,count in ((-1,1),(1,2),(0,-1)):
            with self.subTest(first=first,count=count),self.assertRaises(ValueError):
                tensor.read_rows(first,count)

    def test_unknown_tensor_fails_and_norm_and_gdn_semantics_are_explicit(self):
        with self.assertRaisesRegex(ValueError,"unmapped"):
            map_tensor("blk.0.surprise.weight")
        self.assertEqual(map_tensor("blk.0.attn_q_norm.weight").semantics,"folded_plus_one_norm")
        mapping=map_tensor("blk.0.ssm_a")
        self.assertEqual(mapping.semantics,"negative_exp_A")
        self.assertEqual(mapping.head_layout,"ggml_cyclic_gdn")


if __name__ == "__main__":
    unittest.main()
