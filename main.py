"""
Main execution file to run diffusion code. This structure is necessary to cythonize the diffusion module
"""
from diffusion_core import Diffusion
import logging


def main():
    # # Creating main logger
    # logger = logging.getLogger('main')
    # logger.setLevel(logging.DEBUG)
    # # Create file handler which logs messages
    # fh = logging.FileHandler('core_polar.log')
    # fh.setLevel(logging.DEBUG)
    # # Create formater and add it to handlers
    # formatter = logging.Formatter('%(name)s -  %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # # add the handlers to the logger
    # logger.addHandler(fh)

    # logger.info('Creating instance of Diffusion class')
    diffusion = Diffusion()
    # logger.info('Calling solve_diffusion() function to solve Diffusion problem')
    diffusion.solve_diffusion()

    # logging.shutdown()


if __name__ == '__main__':
    main()
