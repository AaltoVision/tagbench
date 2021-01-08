#pragma once

extern int     opterr,             /* if error message should be printed */
  optind,             /* index into parent argv vector */
  optopt,                 /* character checked for validity */
  optreset;               /* reset getopt */
extern char    *optarg;                /* argument associated with option */

/*
* getopt --
*      Parse argc/argv argument vector.
*/
int getopt(int nargc, char * const nargv[], const char *ostr);
