#ifndef CLI_H
#define CLI_H

#include <string>
#include <vector>

struct CliOptions {
    bool compress = true;
    bool stdout_output = false;
    int block_size = 9;
    std::vector<std::string> input_files;
};

CliOptions parse_args(int argc, char **argv);

#endif // !CLI_H
