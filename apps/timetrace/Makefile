CXX := g++ --std=c++20 -W -Wall -O3

EXES := ttviz

all: $(EXES)

clean:
	rm -f $(EXES) *.o *.a

ttviz: ttviz.cpp
	$(CXX) $^ -lgd -o $@
