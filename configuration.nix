{ config, pkgs, ... }:

{
  imports = [
    ./hardware-configuration.nix
  ];

  #######################################
  # Boot (SeaBIOS + GRUB)
  #######################################

  boot.loader.grub = {
    enable = true;
    device = "/dev/sda";   # bei QEMU -hda
    # device = "/dev/vda"; # bei VirtIO
  };


  #######################################
  # System
  #######################################

  networking.hostName = "nixos-dev";

  time.timeZone = "Europe/Berlin";

  networking.networkmanager.enable = true;


  #######################################
  # User
  #######################################

  users.users.dev = {
    isNormalUser = true;
    description = "Developer";

    extraGroups = [
      "wheel"
      "networkmanager"
      "docker"
    ];

    shell = pkgs.zsh;
  };


  security.sudo.wheelNeedsPassword = false;


  #######################################
  # Shell
  #######################################

  programs.zsh = {
    enable = true;

    enableCompletion = true;

    autosuggestions.enable = true;

    syntaxHighlighting.enable = true;
  };


  programs.bash = {
    completion.enable = true;
  };


  #######################################
  # Git
  #######################################

  programs.git = {
    enable = true;

    config = {
      init.defaultBranch = "main";
      pull.rebase = false;
    };
  };


  #######################################
  # Entwicklungs-Pakete
  #######################################

  environment.systemPackages = with pkgs; [

    # Editor
    vim
    neovim
    nano

    # Terminal
    tmux
    screen
    htop
    btop
    tree
    fastfetch

    # File Tools
    file
    which
    unzip
    zip
    rsync

    # Search
    ripgrep
    fd
    jq

    # Netzwerk
    curl
    wget
    openssh
    nmap

    # Compiler
    gcc
    gnumake
    cmake
    pkg-config

    # Python
    python312
    python312Packages.pip
    python312Packages.virtualenv

    # Andere Sprachen
    nodejs
    go
    rustup

    # Git Tools
    git
    gh

    # Container
    docker-compose

    # QEMU Tools
    qemu
  ];


  #######################################
  # Docker
  #######################################

  virtualisation.docker.enable = true;


  #######################################
  # SSH Server
  #######################################

  services.openssh = {
    enable = true;

    settings = {
      PermitRootLogin = "no";
      PasswordAuthentication = true;
    };
  };


  #######################################
  # QEMU Guest
  #######################################

  services.qemuGuest.enable = true;


  #######################################
  # Nix modern
  #######################################

  nix.settings.experimental-features = [
    "nix-command"
    "flakes"
  ];


  #######################################
  # State Version
  #######################################

  system.stateVersion = "25.05";
}
