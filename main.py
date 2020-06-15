"""
Main execution file to run diffusion code. This structure is necessary to cythonize the diffusion module
"""
from diffusion_core import Diffusion


def main():
    diffusion = Diffusion()
    diffusion.solve_diffusion()


if __name__ == '__main__':
    main()
