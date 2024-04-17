"""
Parsesr for the mdi-ani driver.
"""

import argparse

def create_parser():

    parser = argparse.ArgumentParser(description="MDI ANI Driver")

    parser.add_argument(
        "-mdi",
        help="flags for mdi.",
        default=None,
        type=str, 
    )

    parser.add_argument(
        "-nsteps",
        help="number of steps to run.",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--minimization",
        help="perform geometry optimization.",
        action="store_true",
    )

    parser.add_argument(
        "-out",
        help="The filepath for the output file of ANI energies.",
        default="ani_energies.out",
        type=str,
    )

    return parser
