def pytest_addoption(parser):
    parser.addoption("--vis", action="store_true", help='Visualize pybullet simulation')