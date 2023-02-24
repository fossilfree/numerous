import logging

logger = logging.getLogger('numerous_logger')

formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# TODO: FIX
#handler = logging.StreamHandler()
#handler.setFormatter(formatter)

#logger.addHandler(handler)
#logger.setLevel(logging.INFO)
