# only compiles on mac osx so far 

run: all
	./convnetmain

all:
	clang++  -std=c++1z -I/Library/Frameworks/SDL2.framework/Headers -I/Library/Frameworks/SDL2_image.framework/Headers -F/Library/Frameworks -framework SDL2 -framework SDL2_image -framework OpenCL convnetmain.cpp -o convnetmain

clean:
	rm ./convnetmain
	rm *.o
	