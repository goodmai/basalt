#!/bin/bash
set -e

# Путь к исходникам шейдеров в чекаутах
KERNEL_DIR=".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
OUTPUT_DIR=".build/debug"
TEMP_DIR=".build/metal_temp"

mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Compiling Metal kernels..."

# Собираем список всех .metal файлов
METAL_FILES=$(find "$KERNEL_DIR" -name "*.metal")
# Также добавляем сгенерированные шейдеры
GEN_KERNEL_DIR=".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
if [ -d "$GEN_KERNEL_DIR" ]; then
    METAL_FILES="$METAL_FILES $(find "$GEN_KERNEL_DIR" -name "*.metal")"
fi

# Компилируем каждый файл в .air
for f in $METAL_FILES; do
    name=$(basename "$f" .metal)
    xcrun -sdk macosx metal -I .build/checkouts/mlx-swift/Source/Cmlx/mlx/ -I .build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/ -c "$f" -o "$TEMP_DIR/$name.air"
done

# Архивация в одну библиотеку
xcrun -sdk macosx metallib "$TEMP_DIR"/*.air -o "$OUTPUT_DIR/default.metallib"
cp "$OUTPUT_DIR/default.metallib" default.metallib

echo "Successfully built $OUTPUT_DIR/default.metallib and copied to root"
rm -rf "$TEMP_DIR"
