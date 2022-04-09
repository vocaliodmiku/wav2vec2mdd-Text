#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "glog::glog" for configuration ""
set_property(TARGET glog::glog APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(glog::glog PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libglog.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glog::glog )
list(APPEND _IMPORT_CHECK_FILES_FOR_glog::glog "${_IMPORT_PREFIX}/lib/libglog.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
