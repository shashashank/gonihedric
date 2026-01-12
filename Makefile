CXX := g++ # This is the main compiler
# CXX := clang++ --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
SRCEXT := cpp
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%.$(SRCEXT),$(BUILDDIR)/%.o,$(SOURCES))
EXECUTABLES := $(patsubst $(SRCDIR)/%.$(SRCEXT),$(BINDIR)/%,$(SOURCES))
CFLAGS := -g -O3 -std=c++20 -Wall
LIB := -fopenmp #-pthread -lmongoclient -L lib -lboost_thread-mt -lboost_filesystem-mt -lboost_system-mt
INC := -I include

all: $(EXECUTABLES)

$(BINDIR)/%: $(BUILDDIR)/%.o
	@mkdir -p $(BINDIR)
	@echo " Linking $@..."
	@echo " $(CXX) $^ -o $@ $(LIB)"; $(CXX) $^ -o $@ $(LIB)

# Compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CXX) $(CFLAGS) $(INC) $(LIB) -c -o $@ $<"; $(CXX) $(CFLAGS) $(INC) $(LIB) -c -o $@ $<


.PHONY: clean
clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) bin/*"; $(RM) -r $(BUILDDIR) bin/*

# Tests
tester:
	$(CXX) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

# Spikes
ticket:
	$(CXX) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

