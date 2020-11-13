library("Cardinal")
library("EBImage")

# HR2MSI dataset
dpath1 <- "/Users/sai/Documents/00-NEU/2-Ind-Study/data"
dpath2 <- "HR2MSI-mouse-unary-bladder"
dpath3 <- ""
dpath <- file.path(dpath1, dpath2, dpath3)

msifname <- "HR2MSI mouse urinary bladder S096.imzML"
hr2msi <- readMSIData(file.path(dpath, msifname))

optfname <- "HR2MSI mouse urinary bladder S096 - optical image.tiff"
hr2opt <- readImage(file.path(dpath, optfname))

opt <- preprocess_fixed(hr2opt) # processed optical image
msi <- preprocess_moving(hr2msi) # processed msi image


# # PIG206 dataset
# data(pig206, package="CardinalWorkflows")
# pig206 <- as(pig206, "MSImagingExperiment")
# pig206
# str(pig206)

dpath1 <- "/Users/sai/Documents/00-NEU/2-Ind-Study/msi-coreg/data"
optfname <- "pig206-optical.png"
pigopt <- readImage(file.path(dpath1, optfname))

display(pigopt, "raster")
opt <- preprocess_fixed(pigopt)
msi <- preprocess_moving(pig206)


# Simulate MSI image
mse <- simulateImage(preset=1, npeaks=10, nruns=1, baseline=1, by=50*50)

preset_def <- presetImageDef(preset=1, dim=c(50L,50L))
mse <- simulateImage(preset_def$pixelData, preset_def$featureData)


mse <- simulateImage(preset=2, dim=c(50,50), npeaks=10, nruns=1, baseline=1)
mse <- preprocess_moving(mse)
msi <- mse


# Run spatial shrunken centroids and spatial DGMM to find top
# conntributing m/z values
RNGkind("L'Ecuyer-CMRG")
set.seed(2019)
ssc <- spatialShrunkenCentroids(hr2msi, r=1, k=3,
    s=c(0,3), method="gaussian", BPPARAM=MulticoreParam())

ssc2 <- spatialShrunkenCentroids(hr2msi, r=1, k=3,
    s=c(6,9), method="gaussian", BPPARAM=MulticoreParam())

ssc_sim <- spatialShrunkenCentroids(msi, r=1, k=3,
    s=c(6,9), method="gaussian", BPPARAM=MulticoreParam())

image(ssc2, model=list(s=9), values="probability")
image(ssc_sim, model=list(s=9), values="probability")

top_ssc2 <- topFeatures(ssc2, model=list(s=6))
top_ssc_sim <- topFeatures(ssc_sim, model=list(s=6))

image(hr2msi, mz=633)
image(msi, mz=633)

ssc3 <- spatialShrunkenCentroids(msi[tissue], r=1, k=3,
    s=c(6,9), method="gaussian", BPPARAM=MulticoreParam())
image(ssc3, model=list(s=9), values="probability", key=FALSE)


.validate_summarize_input <- function(moving, r, k, s) {
    moving <- .validate_moving_input(moving)

    if ( !all(c(typeof(r), typeof(k), typeof(s)) == "double") |
        !all(c(class(r), class(k), class(s)) == "numeric") ) {
        stop("r, k, and s must be of numeric class and double type",
            call. = FALSE)
    }

    if ( any(k >= 6) ) {
        warning("k above 5 is not recommended")
    }

    if ( any(r > 3) ) {
        warning("r above 3 is not recommended")
    }

    moving
}


.compute_ssc <- function(moving, r, k, s) {
    ssc <- spatialShrunkenCentroids(moving, r=r, k=k, s=s,
        method="adaptive", BPPARAM=MulticoreParam())

    top <- topFeatures(ssc)$mz

    return(list(ssc, top))
}


summarize_moving_ssc <- function(moving, r=c(1,2),
    k=c(3,4,5), s=c(0,3,6,9), seed=210) {

    # moving <- preprocess_moving(moving) # preprocess moving image

    RKS <- sapply(list(r, k, s), function(x) length(x))
    size <- cumprod(RKS)[3]

    RNGkind("L'Ecuyer-CMRG")
    moving <- .validate_summarize_input(moving, r, k, s)

    top <- matrix(rep(0, 10*size), nrow=10)
    colnames(top) <- rep("name", size)

    ssc_list <- c()

    count <- 1

    for (.s in s) {
        for (.k in k) {
            for (.r in r) {
                out <- .compute_ssc(moving, .r, .k, .s)
                ssc_list <- c(ssc_list, out[[1]])
                top[, count] <- out[[2]]

                colnames(top)[count] <- paste0("s=", .s, ", k=", .k, ", r=", .r)
                count <- count + 1
            }
        }
    }

    return(list(ssc_list, top))

}


out <- summarize_moving_ssc(msi, r=c(1,2), k=c(3,4), s=c(0,3,6,9))
# out <- summarize_moving_ssc(mse, r=c(1), k=c(3,4), s=c(0,3,6,9))

# BUILD COMPOSITE IMAGES FROM MULTIPLE SSC RESULTS AND TOP MZ VALUES
composite <- NULL

# Build composite from ssc images
img_cnt <- length(out[[1]])

fac_l <- lapply(out[[1]], function(x) resultData(x, 1, "class"))
fac_l <- lapply(fac_l, function(x) as.numeric(levels(x))[x])

build_image <- function(fac, nrow, isIncomplete=FALSE, coords=NULL) {
    fac[is.na(fac)] <- 0

    if (isIncomplete) {
        mat <- cbind(coords, class=fac)
        r <- max(mat[,1])
        nrow <- r
        c <- max(mat[,2])
        fac <- matrix(rep(0, r*c), nrow=r)

        for (i in 1:nrow(mat)) {
            fac[mat[i,1], mat[i,2]] <- mat[i,3]
        }
    }

    fac <- matrix(fac, nrow=nrow)
    fac_img <- Image(fac)
    fac_img <- preprocess_fixed(fac_img)
    fac_img
}

comp_img_sscl <- lapply(fac_l, function(x) build_image(x, nrow=dims(msi)[[1]],
    isIncomplete=F, coords=coord(msi)))
comp_img_sscl <- lapply(fac_l, function(x) build_image(x, nrow=dims(msi)[[1]],
    isIncomplete=T, coords=coord(msi)))
# comp_img_sscl <- lapply(fac_l, function(x) build_image(x, nrow=dims(mse)[[1]],
    # isIncomplete=F, coords=coord(mse)))
# comp_img_sscl <- lapply(fac_l, function(x) build_image(x, nrow=dims(msi)[[1]]))

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

# IMAGE THRESHOLDING
comp_img_mz <- thresh(comp_img_mz, w=5, h=5, offset=0.0001)
display(comp_img_mz, "raster")

.comp_img_ssc <- thresh(comp_img_ssc, w=1, h=1, offset=0.02)
display((.comp_img_ssc + 3*comp_img_ssc)/4, "raster")

ratio <- 0.25
msi_summarized <- (ratio*.comp_img_ssc + (1-ratio)*comp_img_ssc)
display(msi_summarized, "raster")

# Temporary working variables
fixed <- opt

moving <- comp_img_ssc
moving <- comp_img_sscl[[16]]
display(moving, "raster")

moving <- comp_img_mz
moving <- msi_summarized


