# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sfy/study/BREWESS/lib/pg/nsg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sfy/study/BREWESS/lib/pg/nsg/build

# Include any dependencies generated for this target.
include src/CMakeFiles/efanna2e_s.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/efanna2e_s.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/efanna2e_s.dir/flags.make

src/CMakeFiles/efanna2e_s.dir/index.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index.cpp.o: ../src/index.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sfy/study/BREWESS/lib/pg/nsg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index.cpp.o"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/efanna2e_s.dir/index.cpp.o -c /home/sfy/study/BREWESS/lib/pg/nsg/src/index.cpp

src/CMakeFiles/efanna2e_s.dir/index.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index.cpp.i"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sfy/study/BREWESS/lib/pg/nsg/src/index.cpp > CMakeFiles/efanna2e_s.dir/index.cpp.i

src/CMakeFiles/efanna2e_s.dir/index.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index.cpp.s"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sfy/study/BREWESS/lib/pg/nsg/src/index.cpp -o CMakeFiles/efanna2e_s.dir/index.cpp.s

src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o: src/CMakeFiles/efanna2e_s.dir/flags.make
src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o: ../src/index_nsg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sfy/study/BREWESS/lib/pg/nsg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o -c /home/sfy/study/BREWESS/lib/pg/nsg/src/index_nsg.cpp

src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/efanna2e_s.dir/index_nsg.cpp.i"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sfy/study/BREWESS/lib/pg/nsg/src/index_nsg.cpp > CMakeFiles/efanna2e_s.dir/index_nsg.cpp.i

src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/efanna2e_s.dir/index_nsg.cpp.s"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sfy/study/BREWESS/lib/pg/nsg/src/index_nsg.cpp -o CMakeFiles/efanna2e_s.dir/index_nsg.cpp.s

# Object files for target efanna2e_s
efanna2e_s_OBJECTS = \
"CMakeFiles/efanna2e_s.dir/index.cpp.o" \
"CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o"

# External object files for target efanna2e_s
efanna2e_s_EXTERNAL_OBJECTS =

src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/build.make
src/libefanna2e_s.a: src/CMakeFiles/efanna2e_s.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sfy/study/BREWESS/lib/pg/nsg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libefanna2e_s.a"
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && $(CMAKE_COMMAND) -P CMakeFiles/efanna2e_s.dir/cmake_clean_target.cmake
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/efanna2e_s.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/efanna2e_s.dir/build: src/libefanna2e_s.a

.PHONY : src/CMakeFiles/efanna2e_s.dir/build

src/CMakeFiles/efanna2e_s.dir/clean:
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build/src && $(CMAKE_COMMAND) -P CMakeFiles/efanna2e_s.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/efanna2e_s.dir/clean

src/CMakeFiles/efanna2e_s.dir/depend:
	cd /home/sfy/study/BREWESS/lib/pg/nsg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sfy/study/BREWESS/lib/pg/nsg /home/sfy/study/BREWESS/lib/pg/nsg/src /home/sfy/study/BREWESS/lib/pg/nsg/build /home/sfy/study/BREWESS/lib/pg/nsg/build/src /home/sfy/study/BREWESS/lib/pg/nsg/build/src/CMakeFiles/efanna2e_s.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/efanna2e_s.dir/depend

