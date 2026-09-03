# Codec source snapshot

These files are synchronized with UCM's `compress_lib` implementations. The
bridge hashes them again at build time, so runtime statistics identify the
exact local snapshot even when the source is refreshed in a later commit.

| File | SHA256 |
|---|---|
| `r160_base_bf16.cc` | `87e25416f2cd4acf05bd3f82c17d3d120c6f658c5b325914c82eda25bc74a30e` |
| `r160_base_bf16.h` | `08c997ef37f4196a4369ade0595d69b80263dcfdf5bf12e638b20f5276b46f09` |
| `tunstall_bf16_r160.cc` | `9dfc5c642f5893c4efac1b50927022c3fea1a17fbcbf21c14ed87dcd38b424e4` |
| `tunstall_bf16_r160.h` | `501a5aa0a7328948b012e1521b873c34e80729b747b7b5c909e19770db39ab2b` |
| `tunstall_bf16_r200.cc` | `f5fa2b48641b879f2ea99e1a38d02d2a455c710099e5e7b738d49aabe6d672ec` |
| `tunstall_bf16_r200.h` | `e5629417a670ab0207a20c26fb6890d6c59810f5c28bdd2051a0eca2309ce693` |
| `tunstall.cc` | `d52577d51c8789ef98b85758f3dbd439419433526d799fb5463b797f0e147a35` |
| `tunstall.h` | `76798dd468e56db7a5a8ab552a69879360e973b8273d9cc819af1252f76e6f34` |

To refresh the snapshot, replace all affected files together, update this
manifest, rebuild on the target machine, and rerun all registered codec smoke tests.
