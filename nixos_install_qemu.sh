#!/bin/sh

echo "--------------------------------------------------------------------------------"
echo "Your attached storage devices will now be listed."
read -p "Press 'q' to exit the list. Press enter to continue." NULL

sudo fdisk -l | less

echo "--------------------------------------------------------------------------------"
echo "Detected the following devices:"
echo

i=0
for device in $(sudo fdisk -l | grep "^Disk /dev" | awk '{print $2}' | sed 's/://'); do
    echo "[$i] $device"
    i=$((i+1))
    DEVICES[$i]=$device
done

echo
read -p "Which device do you wish to install on? " DEVICE

DEV=${DEVICES[$(($DEVICE+1))]}

read -p "How much swap space do you need in GiB (e.g. 8)? " SWAP

read -p "Will now partition ${DEV} with swap size ${SWAP}GiB. Ok? Type 'go': " ANSWER

if [ "$ANSWER" = "go" ]; then
    echo "Partitioning ${DEV}..."

    (
      echo o          # New DOS partition table

      echo n          # New partition
      echo p          # Primary
      echo 1          # Partition 1
      echo            # Default start
      echo -${SWAP}G  # Leave space for swap

      echo n          # New partition
      echo p          # Primary
      echo 2          # Partition 2
      echo            # Default start
      echo            # Default end

      echo t          # Change type
      echo 2          # Partition 2
      echo 82         # Linux swap

      echo a          # Make bootable
      echo 1          # Partition 1

      echo p
      echo w
    ) | sudo fdisk ${DEV}

else
    echo "Cancelled."
    exit
fi

echo "--------------------------------------------------------------------------------"
echo "Getting created partition names..."

P1=${DEV}1
P2=${DEV}2

echo "Root: ${P1}"
echo "Swap: ${P2}"

echo "--------------------------------------------------------------------------------"
read -p "Press enter to install NixOS."

sudo mkfs.ext4 -L nixos ${P1}

sudo mkswap -L swap ${P2}
sudo swapon ${P2}

sudo mount ${P1} /mnt

echo "Generating configuration..."

sudo nixos-generate-config --root /mnt

echo
echo "Open configuration.nix"
read -p "Press Enter..."

sudo nano /mnt/etc/nixos/configuration.nix

echo
echo "IMPORTANT!"
echo "Make sure configuration.nix contains:"
echo
echo 'boot.loader.grub.enable = true;'
echo "boot.loader.grub.device = \"${DEV}\";"
echo

read -p "Press Enter to install..."

sudo nixos-install

echo
echo "Installation complete."
echo "Remove the ISO and reboot."

reboot
