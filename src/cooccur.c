//  Tool to calculate word-word cooccurrence statistics
//
//  Copyright (c) 2014, 2018 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241

#define HASHFN bitwisehash

typedef double real;

/* 词-词共现矩阵元素 */
typedef struct cooccur_rec {
  int word1;
  int word2;
  real val;
} CREC;

/* merge 阶段使用 */
typedef struct cooccur_rec_id {
  int word1;
  int word2;
  real val;
  int id; /* file index */
} CRECID;

typedef struct hashrec {
  char *word;
  long long id;
  struct hashrec *next;
} HASHREC;

int verbose = 2;  // 0, 1, or 2

// GloVe 计算共现矩阵时采用的是稠密矩阵+稀疏矩阵这样的混合存储结构。对于词频
// 大的词，采用稠密矩阵，稀疏词采用稀疏矩阵, max_product 变量控制何时采用稠密
// 矩阵存储，overflow_length 变量控制何时将稀疏矩阵写入到磁盘(用于后续合并)。

// word id 是按照词频排序的，词频越高的，word id 越小，GloVe 认为词频越高的
// 单词越容易共现，可以常驻内存中提高效率，max_product 这个变量就是控制常驻
// 内存单词的阈值，当两个单词 id 的乘积大于 max_product 时，则其共现元素常驻
// 内存，否则可能会写入到磁盘中。
// Cutoff for product of word frequency ranks below which cooccurrence counts
// will be stored in a compressed full array
long long max_product;

// 这个变量控制写入到磁盘的阈值
// Number of cooccurrence records whose product exceeds max_product to store in
// memory before writing to disk
long long overflow_length;
int window_size = 15;  // default context window size
int symmetric = 1;     // 0: asymmetric, 1: symmetric

// soft limit, in gigabytes, used to estimate optimal array sizes
real memory_limit = 3;

// Flag to control the distance weighting of cooccurrence counts
int distance_weighting = 1;
char *vocab_file, *file_head;

/* Efficient string comparison */
int scmp(char *s1, char *s2) {
  while (*s1 != '\0' && *s1 == *s2) {
    s1++;
    s2++;
  }
  return (*s1 - *s2);
}

/* Move-to-front hashing and hash function from Hugh Williams,
 * http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
  char c;
  unsigned int h;
  h = seed;
  for (; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
  return ((unsigned int)((h & 0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
HASHREC **inithashtable() {
  int i;
  HASHREC **ht;
  ht = (HASHREC **)malloc(sizeof(HASHREC *) * TSIZE);
  for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *)NULL;
  return (ht);
}

/* Search hash table for given string, return record if found, else NULL */
HASHREC *hashsearch(HASHREC **ht, char *w) {
  HASHREC *htmp, *hprv;
  unsigned int hval = HASHFN(w, TSIZE, SEED);
  for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0;
       hprv = htmp, htmp = htmp->next)
    ;
  if (htmp != NULL && hprv != NULL) {  // move to front on access
    hprv->next = htmp->next;
    htmp->next = ht[hval];
    ht[hval] = htmp;
  }
  return (htmp);
}

/* Insert string in hash table, check for duplicates which should be absent */
void hashinsert(HASHREC **ht, char *w, long long id) {
  HASHREC *htmp, *hprv;
  unsigned int hval = HASHFN(w, TSIZE, SEED);
  for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0;
       hprv = htmp, htmp = htmp->next)
    ;
  if (htmp == NULL) {
    htmp = (HASHREC *)malloc(sizeof(HASHREC));
    htmp->word = (char *)malloc(strlen(w) + 1);
    strcpy(htmp->word, w);
    htmp->id = id;
    htmp->next = NULL;
    if (hprv == NULL)
      ht[hval] = htmp;
    else
      hprv->next = htmp;
  } else
    fprintf(stderr, "Error, duplicate entry located: %s.\n", htmp->word);
  return;
}

/* Read word from input stream. Return 1 when encounter '\n' or EOF (but
   separate from word), 0 otherwise.
   Words can be separated by space(s), tab(s), or newline(s). Carriage return
   characters are just ignored.
   (Okay for Windows, but not for Mac OS 9-. Ignored even if by themselves or in
   words.)
   A newline is taken as indicating a new document (contexts won't cross
   newline).
   Argument word array is assumed to be of size MAX_STRING_LENGTH.
   words will be truncated if too long. They are truncated with some care so
   that they cannot truncate in the middle of a utf-8 character, but
   still little to no harm will be done for other encodings like iso-8859-1.
   (This function appears identically copied in vocab_count.c and cooccur.c.)
 */
int get_word(char *word, FILE *fin) {
  int i = 0, ch;
  for (;;) {
    ch = fgetc(fin);
    if (ch == '\r') continue;
    if (i == 0 && ((ch == '\n') || (ch == EOF))) {
      word[i] = 0;
      return 1;
    }
    if (i == 0 && ((ch == ' ') || (ch == '\t')))
      continue;  // skip leading space
    if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (ch == '\n')
        ungetc(ch, fin);  // return the newline next time as document ender
      break;
    }
    if (i < MAX_STRING_LENGTH - 1)
      word[i++] = ch;  // don't allow words to exceed MAX_STRING_LENGTH
  }
  word[i] = 0;  // null terminate
  // avoid truncation destroying a multibyte UTF-8 char except if only thing on
  // line (so the i > x tests won't overwrite word[0])
  // see https://en.wikipedia.org/wiki/UTF-8#Description
  if (i == MAX_STRING_LENGTH - 1 && (word[i - 1] & 0x80) == 0x80) {
    if ((word[i - 1] & 0xC0) == 0xC0) {
      word[i - 1] = '\0';
    } else if (i > 2 && (word[i - 2] & 0xE0) == 0xE0) {
      word[i - 2] = '\0';
    } else if (i > 3 && (word[i - 3] & 0xF8) == 0xF0) {
      word[i - 3] = '\0';
    }
  }
  return 0;
}

/* Write sorted chunk of cooccurrence records to file, accumulating duplicate
 * entries */
int write_chunk(CREC *cr, long long length, FILE *fout) {
  if (length == 0) return 0;

  long long a = 0;
  CREC old = cr[a];

  for (a = 1; a < length; a++) {
    if (cr[a].word1 == old.word1 && cr[a].word2 == old.word2) {
      old.val += cr[a].val;
      continue;
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    old = cr[a];
  }
  fwrite(&old, sizeof(CREC), 1, fout);
  return 0;
}

/* Check if two cooccurrence records are for the same two words, used for qsort
 */
int compare_crec(const void *a, const void *b) {
  int c;
  if ((c = ((CREC *)a)->word1 - ((CREC *)b)->word1) != 0)
    return c;
  else
    return (((CREC *)a)->word2 - ((CREC *)b)->word2);
}

/* Check if two cooccurrence records are for the same two words */
int compare_crecid(CRECID a, CRECID b) {
  int c;
  if ((c = a.word1 - b.word1) != 0)
    return c;
  else
    return a.word2 - b.word2;
}

/* Swap two entries of priority queue */
void swap_entry(CRECID *pq, int i, int j) {
  CRECID temp = pq[i];
  pq[i] = pq[j];
  pq[j] = temp;
}

/* Insert entry into priority queue */
void insert(CRECID *pq, CRECID new, int size) {
  int j = size - 1, p;
  pq[j] = new;
  while ((p = (j - 1) / 2) >= 0) {
    if (compare_crecid(pq[p], pq[j]) > 0) {
      swap_entry(pq, p, j);
      j = p;
    } else
      break;
  }
}

/* Delete entry from priority queue */
void delete (CRECID *pq, int size) {
  int j, p = 0;
  pq[p] = pq[size - 1];
  while ((j = 2 * p + 1) < size - 1) {
    if (j == size - 2) {
      if (compare_crecid(pq[p], pq[j]) > 0) swap_entry(pq, p, j);
      return;
    } else {
      if (compare_crecid(pq[j], pq[j + 1]) < 0) {
        if (compare_crecid(pq[p], pq[j]) > 0) {
          swap_entry(pq, p, j);
          p = j;
        } else
          return;
      } else {
        if (compare_crecid(pq[p], pq[j + 1]) > 0) {
          swap_entry(pq, p, j + 1);
          p = j + 1;
        } else
          return;
      }
    }
  }
}

/* Write top node of priority queue to file, accumulating duplicate entries */
int merge_write(CRECID new, CRECID *old, FILE *fout) {
  if (new.word1 == old->word1 &&new.word2 == old->word2) {
    old->val += new.val;
    return 0;  // Indicates duplicate entry
  }
  fwrite(old, sizeof(CREC), 1, fout);
  *old = new;
  return 1;  // Actually wrote to file
}

/* 每个 cooccurrence records 文件是根据 word id 排好序的，
 * 采用优先队列的方式合并所有文件 */
/* Merge [num] sorted files of cooccurrence records */
int merge_files(int num) {
  int i, size;
  long long counter = 0;
  CRECID *pq, new, old;
  char filename[200];
  FILE **fid, *fout;
  fid = malloc(sizeof(FILE) * num);
  pq = malloc(sizeof(CRECID) * num);
  fout = stdout;
  if (verbose > 1)
    fprintf(stderr, "Merging cooccurrence files: processed 0 lines.");

  /* Open all files and add first entry of each to priority queue */
  for (i = 0; i < num; i++) {
    sprintf(filename, "%s_%04d.bin", file_head, i);
    fid[i] = fopen(filename, "rb");
    if (fid[i] == NULL) {
      fprintf(stderr, "Unable to open file %s.\n", filename);
      return 1;
    }
    fread(&new, sizeof(CREC), 1, fid[i]);
    new.id = i;
    insert(pq, new, i + 1);
  }

  /* Pop top node, save it in old to see if the next entry is a duplicate */
  size = num;
  old = pq[0];
  i = pq[0].id;
  delete (pq, size);
  fread(&new, sizeof(CREC), 1, fid[i]);
  if (feof(fid[i]))
    size--;
  else {
    new.id = i;
    insert(pq, new, size);
  }

  /* Repeatedly pop top node and fill priority queue until files have reached
   * EOF */
  while (size > 0) {
    // Only count the lines written to file, not duplicates
    counter += merge_write(pq[0], &old, fout);
    if ((counter % 100000) == 0)
      if (verbose > 1) fprintf(stderr, "\033[39G%lld lines.", counter);
    i = pq[0].id;
    delete (pq, size);
    fread(&new, sizeof(CREC), 1, fid[i]);
    if (feof(fid[i]))
      size--;
    else {
      new.id = i;
      insert(pq, new, size);
    }
  }
  fwrite(&old, sizeof(CREC), 1, fout);
  fprintf(stderr, "\033[0GMerging cooccurrence files: processed %lld lines.\n",
          ++counter);
  for (i = 0; i < num; i++) {
    sprintf(filename, "%s_%04d.bin", file_head, i);
    remove(filename);
  }
  fprintf(stderr, "\n");
  return 0;
}

/* Collect word-word cooccurrence counts from input stream */
int get_cooccurrence() {
  int flag, x, y, fidcounter = 1;
  long long a, j = 0, k, id, counter = 0, ind = 0, vocab_size, w1, w2;
  long long *lookup, *history;
  char format[20], filename[200], str[MAX_STRING_LENGTH + 1];
  FILE *fid, *foverflow;
  real *bigram_table, r;
  HASHREC *htmp, **vocab_hash = inithashtable();
  CREC *cr = malloc(sizeof(CREC) * (overflow_length + 1));
  history = malloc(sizeof(long long) * window_size);

  fprintf(stderr, "COUNTING COOCCURRENCES\n");
  if (verbose > 0) {
    fprintf(stderr, "window size: %d\n", window_size);
    if (symmetric == 0)
      fprintf(stderr, "context: asymmetric\n");
    else
      fprintf(stderr, "context: symmetric\n");
  }
  if (verbose > 1) fprintf(stderr, "max product: %lld\n", max_product);
  if (verbose > 1) fprintf(stderr, "overflow length: %lld\n", overflow_length);

  // Format to read from vocab file, which has (irrelevant) frequency data
  sprintf(format, "%%%ds %%lld", MAX_STRING_LENGTH);
  if (verbose > 1)
    fprintf(stderr, "Reading vocab from file \"%s\"...", vocab_file);
  fid = fopen(vocab_file, "r");
  if (fid == NULL) {
    fprintf(stderr, "Unable to open vocab file %s.\n", vocab_file);
    return 1;
  }

  // Here id is not used: inserting vocab words into hash table with their
  // frequency rank, j
  while (fscanf(fid, format, str, &id) != EOF) hashinsert(vocab_hash, str, ++j);
  fclose(fid);
  vocab_size = j;
  j = 0;
  if (verbose > 1)
    fprintf(stderr, "loaded %lld words.\nBuilding lookup table...", vocab_size);

  /* Build auxiliary lookup table used to index into bigram_table */
  lookup = (long long *)calloc(vocab_size + 1, sizeof(long long));
  if (lookup == NULL) {
    fprintf(stderr, "Couldn't allocate memory!");
    return 1;
  }

  // 前面提到共现矩阵的存储方式是稠密矩阵+稀疏矩阵，依据是两个词 id 的乘积是否
  // 小于某个阈值；所以原始共现矩阵的每一行都有若干个元素可以用稠密矩阵存储;
  // lookup 数组存储的是每行稠密元素个数的累积和
  lookup[0] = 1;
  for (a = 1; a <= vocab_size; a++) {
    if ((lookup[a] = max_product / a) < vocab_size)
      lookup[a] += lookup[a - 1];
    else
      lookup[a] = lookup[a - 1] + vocab_size;
  }
  if (verbose > 1)
    fprintf(stderr, "table contains %lld elements.\n", lookup[a - 1]);

  // bigram_table 是稠密矩阵，存储高频词-词共现, 即原始共现矩阵的左上角
  /* Allocate memory for full array which will store all cooccurrence counts for
   * words whose product of frequency ranks is less than max_product */
  bigram_table = (real *)calloc(lookup[a - 1], sizeof(real));
  if (bigram_table == NULL) {
    fprintf(stderr, "Couldn't allocate memory!");
    return 1;
  }

  fid = stdin;
  sprintf(filename, "%s_%04d.bin", file_head, fidcounter);
  foverflow = fopen(filename, "wb");
  if (verbose > 1) fprintf(stderr, "Processing token: 0");

  /* For each token in input stream, calculate a weighted cooccurrence sum
   * within window_size */
  while (1) {
    if (ind >= overflow_length - window_size) {
      // If overflow buffer is (almost) full, sort it and write it to temporary
      // file.
      qsort(cr, ind, sizeof(CREC), compare_crec);
      write_chunk(cr, ind, foverflow);
      fclose(foverflow);
      fidcounter++;
      sprintf(filename, "%s_%04d.bin", file_head, fidcounter);
      foverflow = fopen(filename, "wb");
      ind = 0;
    }
    flag = get_word(str, fid);
    if (verbose > 2) fprintf(stderr, "Maybe processing token: %s\n", str);
    if (flag == 1) {
      // Newline, reset line index (j); maybe eof.
      if (feof(fid)) {
        if (verbose > 2) fprintf(stderr, "Not getting coocurs as at eof\n");
        break;
      }
      j = 0;
      if (verbose > 2) fprintf(stderr, "Not getting coocurs as at newline\n");
      continue;
    }
    counter++;
    if ((counter % 100000) == 0)
      if (verbose > 1) fprintf(stderr, "\033[19G%lld", counter);
    htmp = hashsearch(vocab_hash, str);
    if (htmp == NULL) {
      if (verbose > 2)
        fprintf(stderr, "Not getting coocurs as word not in vocab\n");
      continue;  // Skip out-of-vocabulary words
    }
    w2 = htmp->id;  // Target word (frequency rank)
    for (k = j - 1; k >= ((j > window_size) ? j - window_size : 0); k--) {
      // Iterate over all words to the left of target word, but not past
      // beginning of line
      w1 = history[k % window_size];  // Context word (frequency rank)
      if (verbose > 2)
        fprintf(stderr, "Adding cooccur between words %lld and %lld.\n", w1,
                w2);
      if (w1 < max_product / w2) {
        // Product is small enough to store in a full array
        // Weight by inverse of distance between words if needed
        bigram_table[lookup[w1 - 1] + w2 - 2] +=
            distance_weighting ? 1.0 / ((real)(j - k)) : 1.0;
        if (symmetric > 0)
          // If symmetric context is used, exchange roles of w2 and w1 (ie look
          // at right context too)
          bigram_table[lookup[w2 - 1] + w1 - 2] +=
              distance_weighting ? 1.0 / ((real)(j - k)) : 1.0;
      } else {
        // Product is too big, data is likely to be sparse. Store these
        // entries in a temporary buffer to be sorted, merged (accumulated),
        // and written to file when it gets full.
        cr[ind].word1 = w1;
        cr[ind].word2 = w2;
        cr[ind].val = distance_weighting ? 1.0 / ((real)(j - k)) : 1.0;
        ind++;                // Keep track of how full temporary buffer is
        if (symmetric > 0) {
          // Symmetric context
          cr[ind].word1 = w2;
          cr[ind].word2 = w1;
          cr[ind].val = distance_weighting ? 1.0 / ((real)(j - k)) : 1.0;
          ind++;
        }
      }
    }
    // Target word is stored in circular buffer to become context word in the
    // future
    history[j % window_size] = w2;
    j++;
  }

  /* Write out temp buffer for the final time (it may not be full) */
  if (verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", counter);
  qsort(cr, ind, sizeof(CREC), compare_crec);
  write_chunk(cr, ind, foverflow);
  sprintf(filename, "%s_0000.bin", file_head);

  /* Write out full bigram_table, skipping zeros */
  if (verbose > 1) fprintf(stderr, "Writing cooccurrences to disk");
  fid = fopen(filename, "wb");
  j = 1e6;
  for (x = 1; x <= vocab_size; x++) {
    if ((long long)(0.75 * log(vocab_size / x)) < j) {
      j = (long long)(0.75 * log(vocab_size / x));
      if (verbose > 1) fprintf(stderr, ".");
    }  // log's to make it look (sort of) pretty
    for (y = 1; y <= (lookup[x] - lookup[x - 1]); y++) {
      if ((r = bigram_table[lookup[x - 1] - 2 + y]) != 0) {
        fwrite(&x, sizeof(int), 1, fid);
        fwrite(&y, sizeof(int), 1, fid);
        fwrite(&r, sizeof(real), 1, fid);
      }
    }
  }

  if (verbose > 1) fprintf(stderr, "%d files in total.\n", fidcounter + 1);
  fclose(fid);
  fclose(foverflow);
  free(cr);
  free(lookup);
  free(bigram_table);
  free(vocab_hash);
  return merge_files(fidcounter + 1);  // Merge the sorted temporary files
}

int find_arg(char *str, int argc, char **argv) {
  int i;
  for (i = 1; i < argc; i++) {
    if (!scmp(str, argv[i])) {
      if (i == argc - 1) {
        printf("No argument given for %s\n", str);
        exit(1);
      }
      return i;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  real rlimit, n = 1e5;
  vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
  file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);

  if (argc == 1) {
    printf("Tool to calculate word-word cooccurrence statistics\n");
    printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
    printf("Usage options:\n");
    printf("\t-verbose <int>\n");
    printf("\t\tSet verbosity: 0, 1, 2 (default), or 3\n");
    printf("\t-symmetric <int>\n");
    printf(
        "\t\tIf <int> = 0, only use left context; if <int> = 1 (default), use "
        "left and right\n");
    printf("\t-window-size <int>\n");
    printf(
        "\t\tNumber of context words to the left (and to the right, if "
        "symmetric = 1); default 15\n");
    printf("\t-vocab-file <file>\n");
    printf(
        "\t\tFile containing vocabulary (truncated unigram counts, produced by "
        "'vocab_count'); default vocab.txt\n");
    printf("\t-memory <float>\n");
    printf(
        "\t\tSoft limit for memory consumption, in GB -- based on simple "
        "heuristic, so not extremely accurate; default 4.0\n");
    printf("\t-max-product <int>\n");
    printf(
        "\t\tLimit the size of dense cooccurrence array by specifying the max "
        "product <int> of the frequency counts of the two cooccurring "
        "words.\n\t\tThis value overrides that which is automatically produced "
        "by '-memory'. Typically only needs adjustment for use with very large "
        "corpora.\n");
    printf("\t-overflow-length <int>\n");
    printf(
        "\t\tLimit to length <int> the sparse overflow array, which buffers "
        "cooccurrence data that does not fit in the dense array, before "
        "writing to disk. \n\t\tThis value overrides that which is "
        "automatically produced by '-memory'. Typically only needs adjustment "
        "for use with very large corpora.\n");
    printf("\t-overflow-file <file>\n");
    printf(
        "\t\tFilename, excluding extension, for temporary files; default "
        "overflow\n");
    printf("\t-distance-weighting <int>\n");
    printf(
        "\t\tIf <int> = 0, do not weight cooccurrence count by distance "
        "between words; if <int> = 1 (default), weight the cooccurrence count "
        "by inverse of distance between words\n");

    printf("\nExample usage:\n");
    printf(
        "./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file "
        "vocab.txt -memory 8.0 -overflow-file tempoverflow < corpus.txt > "
        "cooccurrences.bin\n\n");
    return 0;
  }

  if ((i = find_arg((char *)"-verbose", argc, argv)) > 0)
    verbose = atoi(argv[i + 1]);
  if ((i = find_arg((char *)"-symmetric", argc, argv)) > 0)
    symmetric = atoi(argv[i + 1]);
  if ((i = find_arg((char *)"-window-size", argc, argv)) > 0)
    window_size = atoi(argv[i + 1]);
  if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0)
    strcpy(vocab_file, argv[i + 1]);
  else
    strcpy(vocab_file, (char *)"vocab.txt");
  if ((i = find_arg((char *)"-overflow-file", argc, argv)) > 0)
    strcpy(file_head, argv[i + 1]);
  else
    strcpy(file_head, (char *)"overflow");
  if ((i = find_arg((char *)"-memory", argc, argv)) > 0)
    memory_limit = atof(argv[i + 1]);
  if ((i = find_arg((char *)"-distance-weighting", argc, argv)) > 0)
    distance_weighting = atoi(argv[i + 1]);

  /* The memory_limit determines a limit on the number of elements in
   * bigram_table and the overflow buffer estimate the maximum value that
   * max_product can take so that this limit is still satisfied */
  rlimit = 0.85 * (real)memory_limit * 1073741824 / (sizeof(CREC));
  while (fabs(rlimit - n * (log(n) + 0.1544313298)) > 1e-3)
    n = rlimit / (log(n) + 0.1544313298);
  max_product = (long long)n;
  overflow_length = (long long)rlimit / 6;  // 0.85 + 1/6 ~= 1

  /* Override estimates by specifying limits explicitly on the command line */
  if ((i = find_arg((char *)"-max-product", argc, argv)) > 0)
    max_product = atoll(argv[i + 1]);
  if ((i = find_arg((char *)"-overflow-length", argc, argv)) > 0)
    overflow_length = atoll(argv[i + 1]);

  return get_cooccurrence();
}
