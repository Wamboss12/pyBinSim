import pybinsim
import logging

pybinsim.logger.setLevel(logging.INFO)    # defaults to INFO
# Use logging.WARNING for printing warnings only
with pybinsim.BinSim('example/demo1.cfg') as binsim:
    binsim.stream_start()
