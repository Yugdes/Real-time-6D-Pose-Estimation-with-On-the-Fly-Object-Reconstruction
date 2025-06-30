#!/bin/bash
set -e  # Exit on error

# Step 1: Run the mask capture script
python3 rec_con_mask.py

# Step 0: Get the object name from the file written by rec_con_mask.py
OBJECT_NAME=$(cat last_object_name.txt)
echo "Using object name: $OBJECT_NAME"

# Step 2: Run XMem evaluation
cd XMem
python3 eval.py
cd ..

# Step 3: Move predicted masks to output location
ANNOTATION_DIR="input/${OBJECT_NAME}/Annotations/video1"
MASKS_DIR="input/${OBJECT_NAME}/masks/video1"

if [ -d "$ANNOTATION_DIR" ]; then
  mkdir -p "input/${OBJECT_NAME}/masks"
  if [ -d "$MASKS_DIR" ]; then
    echo "Removing existing $MASKS_DIR to avoid mv conflict"
    rm -rf "$MASKS_DIR"
  fi
  mv "$ANNOTATION_DIR" "input/${OBJECT_NAME}/masks/"
else
  echo "❌ $ANNOTATION_DIR not found. Skipping mask move."
fi

# Step 4: Clean up temporary folders
rm -rf "input/${OBJECT_NAME}/JPEGImages"
rm -rf "input/${OBJECT_NAME}/Annotations"

# Step 5: Delete top-level masks folder if it exists
TOP_LEVEL_MASKS="masks"
if [ -d "$TOP_LEVEL_MASKS" ]; then
  echo "Removing top-level $TOP_LEVEL_MASKS folder"
  rm -rf "$TOP_LEVEL_MASKS"
fi

# Step 6: Move input/<object>/masks → top-level ./masks
mv "input/${OBJECT_NAME}/masks" "./masks"

# Step 7: Rename input/<object>/video1 → input/<object>/masks
mv "input/${OBJECT_NAME}/video1" "input/${OBJECT_NAME}/masks"


