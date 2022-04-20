import pybinsim
import logging

pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO

with pybinsim.BinSim('AAR_Audiolab.txt') as binsim:
    binsim.stream_start()