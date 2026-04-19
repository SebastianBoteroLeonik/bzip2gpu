# List of stuff needed for project completion

**Preamble**:

This file is for keeping track of what is done and what is to be done.
Please mark any complete stages with an `x`. This stages _should_ be mostly independent so we _should_ be able to work on them in any order, as long as we write proper tests.
If anything is partially implemented with more things to be done please include it as a nested subtask. Example:

- [ ] Big thing
  - [x] Easy part
  - [ ] Hard part

## Main algorithmic steps

### Compression

- [x] RLE1
- [x] fBWT
- [x] fMTF
  - [x] Symbol table generation
  - [x] Partial MTF
  - [x] Full MTF
- [x] RLE2
- [ ] Huffman encoding
  - [x] Huffman tree building
  - [ ] Stream huffman encoding

### Decompression

- [ ] RLE1
- [ ] iBWT
- [ ] iMTF
- [ ] RLE2
- [ ] Huffman decoding

## Others

### Stream splitting

- [ ] Creating structures and logic for splitting the data into n \* 100k blocks in compression
- [ ] Creating structures and logic for finding block boundaries in decompression and splitting accordingly

### IO

- [ ] Raw file ingestion
- [ ] Raw file dump
- [ ] bzip2 file decoding
- [ ] bzip2 file encoding

### CLI

- [ ] proper command-line argument parsing compatible with bzip2
