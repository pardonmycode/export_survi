#!/usr/bin/env bash
set -euo pipefail

DISK="/dev/vda"

echo "=== WARNING ==="
echo "ALL DATA ON ${DISK} WILL BE DELETED!"
read -rp "Type YES to continue: " ANSWER

[ "$ANSWER" = "YES" ] || exit 1

echo "== Partitioning =="

parted -s "${DISK}" \
    mklabel msdos \
    mkpart primary ext4 1MiB 100% \
    set 1 boot on

sleep 2

echo "== Formatting =="

mkfs.ext4 -F -L nixos "${DISK}1"

echo "== Mounting =="

mount "${DISK}1" /mnt

echo "== Generating configuration =="

nixos-generate-config --root /mnt

cat >> /mnt/etc/nixos/configuration.nix <<'EOF'

boot.loader.grub.enable = true;
boot.loader.grub.device = "/dev/vda";

networking.hostName = "nixos";

services.openssh.enable = true;

users.users.nixos = {
  isNormalUser = true;
  extraGroups = [ "wheel" ];
  initialPassword = "nixos";
};

security.sudo.wheelNeedsPassword = false;

environment.systemPackages = with pkgs; [
  git
  vim
  curl
  wget
  htop
];

system.stateVersion = "25.05";
EOF

echo
echo "==============================="
echo "Installing NixOS..."
echo "==============================="

nixos-install

echo
echo "Set root password:"
passwd

echo
echo "Installation finished!"
echo
echo "Shutdown the VM, remove the ISO and boot from the disk."
