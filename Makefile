CC = gcc
CXX = g++
#For older gcc, use -O3 or -O2 instead of -Ofast
# CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
CFLAGS = -std=gnu99 -lm -pthread -Ofast -march=native -funroll-loops -Wall -Wextra -Wpedantic
CXXFLAGS = -std=c++11 -lm -pthread -Ofast -march=native -funroll-loops -Wall -Wextra -Wpedantic
BUILDDIR := build
SRCDIR := src

all: dir glove shuffle cooccur vocab_count query_nn

dir :
	mkdir -p $(BUILDDIR)
glove : $(SRCDIR)/glove.c
	$(CC) $(SRCDIR)/glove.c -o $(BUILDDIR)/glove $(CFLAGS)
shuffle : $(SRCDIR)/shuffle.c
	$(CC) $(SRCDIR)/shuffle.c -o $(BUILDDIR)/shuffle $(CFLAGS)
cooccur : $(SRCDIR)/cooccur.c
	$(CC) $(SRCDIR)/cooccur.c -o $(BUILDDIR)/cooccur $(CFLAGS)
vocab_count : $(SRCDIR)/vocab_count.c
	$(CC) $(SRCDIR)/vocab_count.c -o $(BUILDDIR)/vocab_count $(CFLAGS)
query_nn : $(SRCDIR)/query_nn.cc
	${CXX} ${SRCDIR}/query_nn.cc -o ${BUILDDIR}/query_nn $(CXXFLAGS)

clean:
	rm -rf glove shuffle cooccur vocab_count query_nn build
