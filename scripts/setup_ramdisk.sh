#!/bin/bash

# setup_ramdisk.sh
# Ensures /dev/shm has at least 4GB allocation.

RAMDISK_PATH="/dev/shm"
MIN_SIZE_GB=4
MIN_SIZE_BYTES=$((MIN_SIZE_GB * 1024 * 1024 * 1024))

echo "Checking RAMDisk at $RAMDISK_PATH..."

# Check current size and file system type
CURRENT_STATS=$(df -B1 --output=size,fstype "$RAMDISK_PATH" | awk 'NR==2')
CURRENT_SIZE_BYTES=$(echo "$CURRENT_STATS" | awk '{print $1}')
FS_TYPE=$(echo "$CURRENT_STATS" | awk '{print $2}')
CURRENT_SIZE_GB=$((CURRENT_SIZE_BYTES / 1024 / 1024 / 1024))

echo "Current RAMDisk size: ${CURRENT_SIZE_GB}GB (Type: $FS_TYPE)"

if [ "$CURRENT_SIZE_BYTES" -lt "$MIN_SIZE_BYTES" ]; then
    echo "Current size is less than recommended ${MIN_SIZE_GB}GB."
    
    # Check for sudo
    if [ "$EUID" -ne 0 ]; then
        echo "Error: Resizing /dev/shm requires root privileges. Please run with sudo."
        exit 1
    fi

    if [ "$FS_TYPE" == "tmpfs" ]; then
        echo "Attempting to remount /dev/shm to ${MIN_SIZE_GB}GB..."
        mount -o remount,size=${MIN_SIZE_GB}G "$RAMDISK_PATH"
    else
        echo "Warning: $RAMDISK_PATH is not a tmpfs (found $FS_TYPE). Attempting to mount anyway..."
        mount -t tmpfs -o size=${MIN_SIZE_GB}G tmpfs "$RAMDISK_PATH"
    fi
    
    if [ $? -eq 0 ]; then
        echo "Successfully updated $RAMDISK_PATH to ${MIN_SIZE_GB}GB."
    else
        echo "Failed to update $RAMDISK_PATH."
        exit 1
    fi
else
    echo "RAMDisk size is sufficient: ${CURRENT_SIZE_GB}GB."
fi

# Verify free space for the current user safely
FREE_SPACE=$(df -B1 "$RAMDISK_PATH" | awk 'NR==2 {print $4}')
FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024 / 1024))

echo "Current free space on RAMDisk: ${FREE_SPACE_GB}GB"

if [ "$FREE_SPACE" -lt $((2 * 1024 * 1024 * 1024)) ]; then
    echo "Warning: Free space on $RAMDISK_PATH is less than 2GB. Workers may fail."
fi

exit 0
