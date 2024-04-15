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

    # Get the number of atoms
    mdi.MDI_Send_Command("<NATOMS", comm)
    natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)
    print("Number of atoms: " + str(natoms))

    # Try to get the starting coordinates
    coords = np.empty(natoms*3, dtype=float) # make a buffer
    mdi.MDI_Send_Command("<COORDS", comm)
    mdi.MDI_Recv(natoms*3, mdi.MDI_DOUBLE, comm, coords)
    coords = coords.reshape(natoms, 3)
    print(coords.shape)

    # We will figure out the atoms based on mass.
    # LAMMPS doesn't store elements.
    masses = np.empty(natoms, dtype=float)
    mdi.MDI_Send_Command("<MASSES", comm)
    mdi.MDI_Recv(natoms, mdi.MDI_DOUBLE, comm, masses)
    print(masses)

    # If we want to be more sophisticated, we would
    # use a mapping of all of the elements to masses.
    # We know we only have hydrogen and oxygen, so
    # we'll do a simple mapping for now.
    element_map = {15.9994: 8, 1.008: 1}
    elements = np.array([element_map[mass] for mass in masses])
    

    # Get the box vectors
    box_vects = np.empty(9, dtype=float)
    mdi.MDI_Send_Command("<CELL", comm)
    mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm, buf=box_vects)
    print(box_vects)

    # Start an MD simulation
    mdi.MDI_Send_Command("@INIT_MD", comm) # Send MDI to INIT_MD node

    # Set up Torch ANI
    device = "cpu"
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    # set up tensors for ani boundary conditions
    cell = torch.tensor(box_vects.reshape(3, 3), device=device).float()
    pbc = torch.tensor([True, True, True], device=device)
    
    # Set the number of timesteps for our simulation
    nsteps = 10

    # iterate
    for i in range(nsteps):
        # Send MDI to forces node
        mdi.MDI_Send_Command("@FORCES", comm)

        # Get the current coordinates
        coords = np.empty(3*natoms, dtype=float)
        mdi.MDI_Send_Command("<COORDS", comm)
        mdi.MDI_Recv(3*natoms, mdi.MDI_DOUBLE, comm, buf=coords)

        # Get the energy from LAMMPS
        mdi.MDI_Send_Command("<ENERGY", comm)
        lammps_energy = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)

        # ani wants a torch tensor with shape (natoms, 3)
        coords_reshape = coords.reshape(natoms, 3)

        # create a torch tensor for coords
        coords_torch = torch.tensor([coords_reshape], requires_grad=True, device=device).float()
        
        # create a torch tensor for elements
        elements_torch = torch.tensor([elements], device=device)

        # calculate the forces using ANI
        energy = model((elements_torch, coords_torch), cell=cell, pbc=pbc).energies
        derivative = torch.autograd.grad(energy, coords_torch)[0]
        forces = -derivative.squeeze()

        print(f"timestep: {i} {lammps_energy} {energy.item()}")
        
        # get the forces as a numpy array
        forces_np = forces.cpu().detach().numpy()

        # reshape to be 1-dimensional like MDI wants
        forces_np = forces_np.reshape(natoms*3)

        # Send the forces to the engine
        mdi.MDI_Send_Command(">FORCES", comm)
        mdi.MDI_Send(forces_np, natoms*3, mdi.MDI_DOUBLE, comm)

    # Send the "EXIT" command to each of the engines
    mdi.MDI_Send_Command("EXIT", comm)