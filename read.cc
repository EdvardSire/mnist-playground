#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

int32_t swapBytes(int32_t num) {
  int32_t swapped = ((num >> 24) & 0x000000FF) | ((num >> 8) & 0x0000FF00) |
                    ((num << 8) & 0x00FF0000) | ((num << 24) & 0xFF000000);
  return swapped;
}
struct train_images {
  unsigned char image[28 * 28];
  unsigned char label;
};

int main() {
  FILE *fTrain;
  fTrain = fopen("data/train-images-idx3-ubyte", "rb");
  FILE *fLabels;
  fLabels = fopen("data/train-labels-idx1-ubyte", "rb");

  if ((fTrain || fLabels) == 0) {
    printf("Files not found\n");
    return 0;
  }

  int magic, num_images, rows, cols;
  // Train
  fread(&magic, sizeof(int), 1, fTrain);
  fread(&num_images, sizeof(int), 1, fTrain);
  fread(&rows, sizeof(int), 1, fTrain);
  fread(&cols, sizeof(int), 1, fTrain);
  int magic_label, num_labels;

  // Label
  fread(&magic_label, sizeof(int), 1, fLabels);
  fread(&num_labels, sizeof(int), 1, fLabels);

  struct train_images *ALL_IMAGES =
      (struct train_images *)malloc(60000 * sizeof(struct train_images));

  for (int i = 0; i < swapBytes(num_images); i++) {
    fread(ALL_IMAGES[i].image, sizeof(char), swapBytes(rows) * swapBytes(cols),
          fTrain);
    fread(&ALL_IMAGES[i].label, sizeof(char), 1, fLabels);
  }

  // Test; should be 8
  int INDEX = 60000 - 1;
  printf("%i\n", ALL_IMAGES[INDEX].label);
  for (int i = 0; i < 28 * 28; i++) {
    printf("%3i ", ALL_IMAGES[INDEX].image[i]);
    if (i % 28 == 0) {
      printf("\n");
    }
  };

  return 0;
}
