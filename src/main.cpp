#include "cli.h"
#include "compression.h"
#include "io.h"
#include "stopwatch.h"
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
  Stopwatch stopwatch{};

  stopwatch.start("All");
  if (options.input_files.empty()) {
    BZFileInputStream input(10, 900000);

    std::vector<uint8_t> output;
    bzip2_gpu_compress(input, options.block_size, output);

    std::cout.write(reinterpret_cast<const char *>(output.data()),
                    output.size());
  } else {
    for (const auto &file : options.input_files) {
      stopwatch.start("input creation");
      BZFileInputStream input(10, 900000, file);
      stopwatch.end();

      std::vector<uint8_t> output;
      stopwatch.start("compression");
      bzip2_gpu_compress(input, options.block_size, output);
      stopwatch.end();

      if (options.stdout_output) {
        stopwatch.start("compressed file write to stdout");
        std::cout.write(reinterpret_cast<const char *>(output.data()),
                        output.size());
        stopwatch.end();
      } else {
        stopwatch.start("compressed file write");
        std::string out_filename = file + ".bz2";
        writeBinaryFile(out_filename, output);
        stopwatch.end();
      }
    }
  }
  stopwatch.end();

  return 0;
}
