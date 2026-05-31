#include "cli.h"
#include "compression.h"
#include "io.h"
#include <fstream>
#include <iostream>
#include <vector>

bool writeBinaryFile(const std::string &filename,
                     const std::vector<uint8_t> &data) {
  std::ofstream outFile(filename, std::ios::out | std::ios::binary);
  if (!outFile) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return false;
  }
  outFile.write(reinterpret_cast<const char *>(data.data()), data.size());
  return outFile.good();
}

int main(int argc, char **argv) {
  CliOptions options = parse_args(argc, argv);

  if (options.input_files.empty()) {
    // std::istreambuf_iterator<char> begin(std::cin), end;
    // std::vector<uint8_t> input(begin, end);
    BZFileInputStream input(10, 900000);

    std::vector<uint8_t> output;
    bzip2_gpu_compress(input, options.block_size, output);

    std::cout.write(reinterpret_cast<const char *>(output.data()),
                    output.size());
  } else {
    for (const auto &file : options.input_files) {
      // std::ifstream inFile(file, std::ios::in | std::ios::binary);
      // if (!inFile) {
      //   std::cerr << "bzip2gpu: Can't open input file " << file << std::endl;
      //   continue;
      // }
      //
      // std::vector<uint8_t> input((std::istreambuf_iterator<char>(inFile)),
      //                            std::istreambuf_iterator<char>());
      // inFile.close();
      BZFileInputStream input(10, 900000, file);

      std::vector<uint8_t> output;
      bzip2_gpu_compress(input, options.block_size, output);

      if (options.stdout_output) {
        std::cout.write(reinterpret_cast<const char *>(output.data()),
                        output.size());
      } else {
        std::string out_filename = file + ".bz2";
        writeBinaryFile(out_filename, output);
      }
    }
  }

  return 0;
}
