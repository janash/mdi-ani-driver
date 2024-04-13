import sys

import numpy as np
import torch
import torchani


# Import the MDI Library
try:
    import mdi
except:
    raise Exception("Unable to import the MDI Library")

# Import MPI Library
try:
    from mpi4py import MPI
    use_mpi4py = True
    mpi_comm_world = MPI.COMM_WORLD
except ImportError:
    use_mpi4py = False
    mpi_comm_world = None


if __name__ == "__main__":

    # Read the command-line options
    iarg = 1
    mdi_options = None
    while iarg < len(sys.argv):
        arg = sys.argv[iarg]

        if arg == "-mdi":
            mdi_options = sys.argv[iarg + 1]
            iarg += 1
        else:
            raise Exception("Unrecognized command-line option")

        iarg += 1

    # Confirm that the MDI options were provided
    if mdi_options is None:
        raise Exception("-mdi command-line option was not provided")

    # Initialize the MDI Library
    mdi.MDI_Init(mdi_options)

    # Get the correct MPI intra-communicator for this code
    mpi_comm_world = mdi.MDI_MPI_get_world_comm()

    # Connect to the engines
    comm = mdi.MDI_Accept_Communicator()

    # Get the name of the engine
    mdi.MDI_Send_Command("<NAME", comm)
    engine_name = mdi.MDI_Recv(mdi.MDI_NAME_LENGTH, mdi.MDI_CHAR, comm)
    print("Engine name: " + engine_name)
    
   
    # Send the "EXIT" command to each of the engines
    mdi.MDI_Send_Command("EXIT", comm)