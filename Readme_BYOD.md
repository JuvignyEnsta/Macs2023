# BYOD IN203 : programmation parallèle
ENSTA Paris - édition 2022/23

## Installation des outils nécessaires aux TDs

### Linux/Debian 

    sudo apt install build-essential make g++ gdb libopenmpi-dev python3-mpi4py

### Mac

Installer [homebrew](https://brew.sh) en suivant les instructions, puis :

    brew install gcc open-mpi
    brew update # MàJ

A mettre dans le *.bashrc* :

    export OMPI_CC=gcc-10
    export OMPI_CXX=g++-10

### Windows 10/11

Installation d'Ubuntu sous Windows 10/11 : allez sur [ce lien](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview) et suivez les instructions.

Tapez *bash* dans la barre de questionnement.
Une fois sous Linux :

    sudo apt update
    sudo apt install build-essential make g++ gdb libopenmpi-dev python3-mpi4py

__Remarque__ : Sous Windows 11, il ,'est pas nécessaire d'installer un serveur X11. Les fenêtre X11 de linux sont redirigés en directX sur le bureau Windows automatiquement.

## Vérification de l'installation

Vous trouverez sur [ce lien](https://github.com/JuvignyEnsta/Installation_Test) un projet que vous pouvez télécharger afin de tester votre environnement. Les instructions pour installer ce projet
sont inclus dans le projet. Si tout se passe bien, le projet vous affichera des messages de bienvenues.