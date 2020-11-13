# Import necessary packages
library("EBImage")
library("Cardinal")
library("CardinalWorkflows")

# Data from pig206 dataset is downloaded from CardinalWorkflows package
data(pig206, package="CardinalWorkflows")
pig206 <- as(pig206, "MSImagingExperiment") # cast as MSImagingExperiment

# Read in data from disk
dpath <- "/Users/sai/Documents/00-NEU/2-Ind-Study/data/pig206/"
optfname <- "/Users/sai/Documents/00-NEU/2-Ind-Study/data/pig206/pig206.png"
opt <- readImage(optfname)
opt <- preprocess_fixed(opt) # processed optical image
msi <- preprocess_moving(pig206) # processed msi image

# Perform SSC segmentation with all the combinations of parameters below
out <- summarize_moving_ssc(msi, r=c(1,2), k=c(3,4), s=c(0,3,6,9))

# Save and read the segmentation results
# saveRDS(out, file.path(dpath, "pig_ssc_results.rds"))
out <- readRDS(file.path(dpath, "pig_ssc_results.rds"))

out2 <- summarize_moving_ssc(msi, r=c(1,2), k=c(5), s=c(0,3,6,9))
# saveRDS(out2, file.path(dpath, "pig_ssc_results2.rds"))

# BUILD COMPOSITE IMAGES FROM MULTIPLE TOP MZ VALUES

# Build composite from top mz values
top_mz = as.vector(out[[2]])
image(msi, mz=top_mz, superpose=TRUE, key=FALSE)

# Build composite image from multiple mz values
composite_img_mz <- function(msi, top_mz) {
    nrow = dims(msi)[[1]]
    N <- length(top_mz)

    img_l <- lapply(top_mz,
        function(x) as.vector(image(msi, mz=x)[[1]][[1]][[1]][[3]]))
    img_l <- lapply(img_l, function(x) build_image(x, nrow=nrow))

    comp_img <- img_l[[1]]
    for (i in 2:N) {
        comp_img <- comp_img + img_l[[i]]
    }

    comp_img <- preprocess_fixed(comp_img)
}

comp_img_mz <- composite_img_mz(msi, top_mz)
composite <- c(composite, list(comp_img_mz))

display(comp_img_mz, "raster")

top_mz


# Segmentation with only the top_mz
fid <- features(msi, mz=top_mz)
fid = sort(unique(fid))
msi3 = msi[fid, ]
out3 <- summarize_moving_ssc(msi3, r=c(1,2), k=c(7), s=c(0,3,6,9))



fid2 <- c(416, 417)
msi4 <- msi[fid2, ]

z <- npyLoad("/Users/sai/Documents/transformed_fixed.npy")
spectra(msi4)[2, ] <- as.vector(z)
out4 <- summarize_moving_ssc(msi4pi, r=c(1,2), k=c(7), s=c(0,3,6,9))
