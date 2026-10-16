"""Explicit process naming and thread caps for single-process CPU checks."""
import os


def pytest_addoption(parser):
    parser.addoption("--process-name", default=os.environ.get("PROCESS_NAME"),
                     help="research process name visible in ps")


def pytest_configure(config):
    name = config.getoption("--process-name")
    if name:
        os.environ["PROCESS_NAME"] = name
        import setproctitle
        setproctitle.setproctitle(name)


def pytest_runtest_setup(item):
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(2)
