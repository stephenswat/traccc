traccc_add_test(
    cuda

    # Define the sources for the test.
    test_cca.cpp

    LINK_LIBRARIES
    traccc_tests_common traccc::core traccc::io traccc::cuda GTest::gtest_main
)
