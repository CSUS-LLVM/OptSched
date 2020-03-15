# Test runner infrastructure for OptSched. This configures the OptSched test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

list(APPEND OPTSCHED_TEST_DEPS
  OptSched
  )

set(OPTSCHED_TEST_PARAMS
  optsched_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_custom_target(optsched-test-depends DEPENDS ${OPTSCHED_TEST_DEPS})
set_target_properties(optsched-test-depends PROPERTIES FOLDER "OptSched tests")

add_lit_testsuite(check-optsched "Running the OptSched regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${OPTSCHED_TEST_PARAMS}
  DEPENDS ${OPTSCHED_TEST_DEPS}
  ARGS ${OPTSCHED_TEST_EXTRA_ARGS}
  )
set_target_properties(check-optsched PROPERTIES FOLDER "OptSched tests")

add_lit_testsuites(OPTSCHED ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${OPTSCHED_TEST_PARAMS}
  DEPENDS ${OPTSCHED_TEST_DEPS}
)
