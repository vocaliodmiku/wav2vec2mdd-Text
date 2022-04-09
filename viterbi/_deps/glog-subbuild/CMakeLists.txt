# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.20.2)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(glog-populate NONE)



include(ExternalProject)
ExternalProject_Add(glog-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/google/glog/archive/v0.4.0.zip" "URL_HASH" "SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc"
                    SOURCE_DIR          "/home/sait/Workplace/viterbi/_deps/glog-src"
                    BINARY_DIR          "/home/sait/Workplace/viterbi/_deps/glog-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


