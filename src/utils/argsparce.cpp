#include "cli.h"
#include <getopt.h>
#include <stdio.h>  /* for printf */
#include <stdlib.h> /* for exit */

void show_help() {
  printf("CUDA implementation of bzip2\n"
         "   usage: bzip2gpu [flags and input files in any order]\n"
         "\n"
         "   -h --help           print this message\n"
         // "   -d --decompress     force decompression\n"
         "   -z --compress       force compression\n"
         // "   -k --keep           keep (don't delete) input files\n"
         // "   -f --force          overwrite existing output files\n"
         // "   -t --test           test compressed file integrity\n"
         "   -c --stdout         output to standard out\n"
         // "   -q --quiet          suppress noncritical error messages\n"
         // "   -v --verbose        be verbose (a 2nd -v gives more)\n"
         // "   -L --license        display software version & license\n"
         // "   -V --version        display software version & license\n"
         // "   -s --small          use less memory (at most 2500k)\n"
         "   -1 .. -9            set block size to 100k .. 900k\n"
         "   --fast              alias for -1\n"
         "   --best              alias for -9\n"
         // "\n"
         // "   If invoked as `bzip2', default action is to compress.\n"
         // "              as `bunzip2',  default action is to decompress.\n"
         // "              as `bzcat', default action is to decompress to
         // stdout.\n"
         "\n"
         "   If no file names are given, bzip2gpu compresses or decompresses\n"
         "   from standard input to standard output.  You can combine\n"
         "   short flags, so `-z -4' means the same as -z4 or -zv, &c.\n");
}

void parse_args(int argc, char **argv) {
  while (1) {
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},   {"compress", no_argument, 0, 'z'},
        {"stdout", no_argument, 0, 'c'}, {"fast", no_argument, 0, '1'},
        {"best", no_argument, 0, '9'},   {0, 0, 0, 0}};

    int c = getopt_long(argc, argv, ":hzc123456789", long_options, NULL);
    if (c == -1)
      break;
    switch (c) {
    case 'z':
      printf("option z [TODO]\n");
      break;
    case 'c':
      printf("option c [TODO]\n");
      break;
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      printf("option %c [TODO]\n", c);
      break;

    case 'h':
      show_help();
      exit(EXIT_SUCCESS);
      break;

    case ':':
      printf("Missing argument [TODO]\n");
      break;

    case '?':
      printf("\n");
      if (optopt) {
        printf("UNKNOWN OPTION: -%c\n", optopt);
      } else {
        printf("UNKNOWN OPTION: %s\n", argv[optind - 1]);
      }
      printf("\n");
      show_help();
      exit(EXIT_FAILURE);
      break;

    default:
      fprintf(stderr, "?? getopt returned character code 0%o ??\n", c);
      fprintf(stderr, "This error is unexpected. Aboting\n");
      exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }
}
