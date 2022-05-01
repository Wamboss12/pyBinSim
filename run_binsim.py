import logging
import pybinsim

pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO

with pybinsim.BinSim('example/AAR_Audiolab.txt') as binsim:
    binsim.stream_start()