import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchani


# Import the MDI Library
import mdi

# Import MPI Library
try:
    from mpi4py import MPI
    use_mpi4py = True
    mpi_comm_world = MPI.COMM_WORLD
except ImportError:
    use_mpi4py = False
    mpi_comm_world = None

# Import the parser
from parser import create_parser

if __name__ == "__main__":

    # Read the command-line options
    args = create_parser().parse_args()

    mdi_options = args.mdi
    nsteps = args.nsteps
    node = "@INIT_MD" if not args.minimization else "@INIT_OPTG"
    outputfile = args.out

    if mdi_options is None:
        mdi_options = "-role DRIVER -name driver -method TCP -port 8021 -hostname localhost"
        warnings.warn(f"Warning: -mdi not provided. Using default value: {mdi_options}")

    # Initialize the MDI Library
    mdi.MDI_Init(mdi_options)

    # Get conversion factors
    bohr_to_angstrom = mdi.MDI_Conversion_Factor("bohr","angstrom")

    # Get the correct MPI intra-communicator for this code
    mpi_comm_world = mdi.MDI_MPI_get_world_comm()

    # Connect to the engines
    comm = mdi.MDI_Accept_Communicator()

    # Print the simulation type:
    if node == "@INIT_MD":
        print("Performing MD simulation")
    elif node == "@INIT_OPTG":
        print("Performing geometry optimization (minimization)")

    # Get the name of the engine
    mdi.MDI_Send_Command("<NAME", comm)
    engine_name = mdi.MDI_Recv(mdi.MDI_NAME_LENGTH, mdi.MDI_CHAR, comm)

    # Get the number of atoms
    mdi.MDI_Send_Command("<NATOMS", comm)
    natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)

    # We will figure out the atoms based on mass.
    # LAMMPS doesn't store elements.
    masses = np.empty(natoms, dtype=float)
    mdi.MDI_Send_Command("<MASSES", comm)
    mdi.MDI_Recv(natoms, mdi.MDI_DOUBLE, comm, masses)

    # If we want to be more sophisticated, we would
    # use a mapping of all of the elements to masses.
    # We know we only have hydrogen and oxygen, so
    # we'll do a simple mapping for now.
    element_map = {15.9994: 8, 1.008: 1}
    elements = np.array([[element_map[mass] for mass in masses]])
    
    # Get the box vectors
    box_vects = np.empty(9, dtype=float)
    mdi.MDI_Send_Command("<CELL", comm)
    mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm, buf=box_vects)
    box_vects = bohr_to_angstrom * box_vects

    # Start appropriate type of simulaton
    mdi.MDI_Send_Command(node, comm) 

    # Set up Torch ANI
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    # Set up tensors for ANI
    cell = torch.tensor(box_vects.reshape(3, 3), device=device).float()
    pbc = torch.tensor([True, True, True], device=device)
    elements_torch = torch.tensor(elements, device=device)

    # Run the simulation - iterate for nsteps
    energies = []
    for i in range(nsteps):
        # Send MDI to forces node
        mdi.MDI_Send_Command("@FORCES", comm)

        # Get the current coordinates
        coords = np.empty(3*natoms, dtype=float)
        mdi.MDI_Send_Command("<COORDS", comm)
        mdi.MDI_Recv(3*natoms, mdi.MDI_DOUBLE, comm, buf=coords)
        coords = bohr_to_angstrom * coords

        # Get the energy from LAMMPS
        mdi.MDI_Send_Command("<ENERGY", comm)
        lammps_energy = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)

        # ani wants a torch tensor with shape (natoms, 3)
        coords_reshape = coords.reshape(1, natoms, 3)

        # create a torch tensor for coords
        coords_torch = torch.tensor(coords_reshape, requires_grad=True, device=device).float()

        # calculate the forces using ANI
        energy = model((elements_torch, coords_torch), cell=cell, pbc=pbc).energies
        derivative = torch.autograd.grad(energy, coords_torch)[0]
        forces = -derivative.squeeze()

        print(f"timestep: {i} {lammps_energy} {energy.item()}")
        
        # get the forces as a numpy array
        forces_np = forces.cpu().detach().numpy()

        # reshape to be 1-dimensional like MDI wants
        forces_np = forces_np.reshape(natoms*3)
        forces_np = forces_np / bohr_to_angstrom

        # Send the forces to the engine
        mdi.MDI_Send_Command(">FORCES", comm)
        mdi.MDI_Send(forces_np, natoms*3, mdi.MDI_DOUBLE, comm)

        # append TorchANI energies
        energies.append(energy.item())

    # write energies
    with open(outputfile, "w") as f:
        for energy in energies:
            f.write(f"{energy}\n")

    # use matplotib to make a plot of energies - will probably remove
    plt.plot(energies)
    plt.xlabel("Timestep")
    plt.ylabel("Energy (Hartree)")
    
    # make figure fit on page
    plt.tight_layout()

    if node == "@INIT_OPTG":
        plt.savefig("minimization_energies.png")
    elif node == "@INIT_MD":
        plt.savefig("md_energies.png")

    # Send the "EXIT" command to each of the engines
    mdi.MDI_Send_Command("EXIT", comm)