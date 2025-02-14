# Disable loop not vectorized warnings on Clang. This generates a lot of
# diagnostic messages when compiling AMReX that we can't do anything about
if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  target_compile_options(
    amrex PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-pass-failed>)
  if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND PELEC_ENABLE_FPE_TRAP_FOR_TESTS)
    target_compile_options(
      amrex PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-ffp-exception-behavior=maytrap>)
  endif()
endif()
