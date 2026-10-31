# GEO-Bench Dataset Summary

Counts only — the test split's contents are never loaded here, only its sample count read from the partition file metadata.

## m-eurosat (classification_v1.0)

- Train samples: 2000
- Validation samples: 1000
- Test samples: 1000 (count only, split not opened)
- Bands (13): 01 - Coastal aerosol, 02 - Blue, 03 - Green, 04 - Red, 05 - Vegetation Red Edge, 06 - Vegetation Red Edge, 07 - Vegetation Red Edge, 08 - NIR, 08A - Vegetation Red Edge, 09 - Water vapour, 10 - SWIR - Cirrus, 11 - SWIR, 12 - SWIR
- Image shape (H, W, C): (64, 64, 13)
- Label: Classification, 10 classes: Industrial Buildings, Residential Buildings, Annual Crop, Permanent Crop, River, Sea and Lake, Herbaceous Vegetation, Highway, Pasture, Forest

## m-so2sat (classification_v1.0)

- Train samples: 19992
- Validation samples: 986
- Test samples: 986 (count only, split not opened)
- Bands (18): 01 - VH.Real, 02 - VH.Imaginary, 03 - VV.Real, 04 - VV.Imaginary, 05 - VH.LEE Filtered, 06 - VV.LEE Filtered, 07 - VH.LEE Filtered.Real, 08 - VV.LEE Filtered.Imaginary, 02 - Blue, 03 - Green, 04 - Red, 05 - Vegetation Red Edge, 06 - Vegetation Red Edge, 07 - Vegetation Red Edge, 08 - NIR, 08A - Vegetation Red Edge, 11 - SWIR, 12 - SWIR
- Image shape (H, W, C): (32, 32, 18)
- Label: Classification, 17 classes: Compact high rise, Compact mid rise, Compact low rise, Open high rise, Open mid rise, Open low rise, Lightweight low rise, Large low rise, Sparsely built, Heavy industry, Dense trees, Scattered trees, Bush, scrub, Low plants, Bare rock or paved, Bare soil or sand, Water

## m-pv4ger-seg (segmentation_v1.0)

- Train samples: 3000
- Validation samples: 403
- Test samples: 403 (count only, split not opened)
- Bands (3): Red, Green, Blue
- Image shape (H, W, C): (320, 320, 3)
- Label: SegmentationClasses, 2 classes: no solar pv, solar pv

## m-nz-cattle (segmentation_v1.0)

- Train samples: 524
- Validation samples: 66
- Test samples: 65 (count only, split not opened)
- Bands (3): Red, Green, Blue
- Image shape (H, W, C): (500, 500, 3)
- Label: SegmentationClasses, 2 classes: no cattle, cattle
