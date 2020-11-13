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


# BUILD COMPOSITE IMAGES FROM MULTIPLE SSC RESULTS AND TOP MZ VALUES
composite <- NULL

# Build composite from ssc images
img_cnt <- length(out[[1]])

fac_l <- lapply(out[[1]], function(x) resultData(x, 1, "class"))
fac_l <- lapply(fac_l, function(x) as.numeric(levels(x))[x])

comp_img_sscl <- lapply(fac_l, function(x) build_image(x, nrow=dims(msi)[[1]],
    isIncomplete=F, coords=coord(msi)))

comp_img_ssc <- comp_img_sscl[[1]]
for (i in 2:img_cnt) {
    comp_img_ssc <- comp_img_ssc + comp_img_sscl[[i]]
}

comp_img_ssc <- comp_img_ssc / length(comp_img_sscl) # composite of ssc results
composite <- list(comp_img_ssc)
display(comp_img_ssc, "raster")

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


