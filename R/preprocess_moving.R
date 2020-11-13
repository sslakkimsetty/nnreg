library("Cardinal")
library("CardinalWorkflows")

# dpath1 <- "/Users/sai/Documents/00-NEU/2-Ind-Study/data"
# dpath2 <- "S042"
# dpath3 <- ""
# dpath <- file.path(dpath1, dpath2, dpath3)

# msifname <- "S042-continuous.imzML"

# # Read MSI data
# msi <- readMSIData(file.path(dpath, msifname))


# # Simulate MSI data
# register(SerialParam())
# register(MulticoreParam())
# set.seed(2020)

# msi <- simulateImage(preset=1, npeaks=50, nruns=1, baseline=2)

.validate_moving_input <- function(moving) {
    valid_classes <- c("MSImagingExperiment",
        "MSContinuousImagingExperiment",
        "MSProcessedImagingExperiment")

    if ( !any(class(moving) %in% valid_classes) ) {
        stop(cat("Class of object should be either of ", valid_classes),
            call. = FALSE)
    }

    moving
}


preprocess_moving <- function(moving,
    process=c("normalize", "smooth", "baseline", "pick", "align", "filter"),
    seed=210) {

    moving <- .validate_moving_input(moving)

    if ("normalize" %in% process) {
        message("Normalizing...")
        moving <- normalize(moving, method="tic")
    }

    if ("smooth" %in% process) {
        message("Smoothing...")
        moving <- smoothSignal(moving, method="gaussian", window=9)
    }

    if ("baseline" %in% process) {
        message("Correcting baseline...")
        moving <- reduceBaseline(moving, method="median", blocks=50)
    }

    if ("pick" %in% process) {
        message("Peak picking...")
        moving <- peakPick(moving, method="adaptive", SNR=2)
    }

    if ("align" %in% process) {
        if ( !("pick" %in% process) ) {
            stop("Peak picking should also be included!", call.=FALSE)
        }
        message("Peak aligning...")
        moving <- peakAlign(moving, method="diff")
    }

    if ("filter" %in% process) {
        if ( !all(c("pick", "align") %in% process) ) {
            stop("Peak pick or align missing and should be included!",
                call.=FALSE)
        }
        message("Peak filtering...")
        # moving <- peakFilter(moving, freq.min=0.2)
    }

    moving <- process(moving)

    moving
}

# msi2 <- preprocess_moving(msi)
# plot(msi2, pixel=2)



# # HR2MSI dataset
# dpath1 <- "/Users/sai/Documents/00-NEU/2-Ind-Study/data"
# dpath2 <- "HR2MSI-mouse-unary-bladder"
# dpath3 <- ""
# dpath <- file.path(dpath1, dpath2, dpath3)

# msifname <- "HR2MSI mouse urinary bladder S096.imzML"
# hr2msi <- readMSIData(file.path(dpath, msifname))

# hr2msi <- smoothSignal(hr2msi, method="gaussian", window=9)
# hr2msi <- peakPick(hr2msi, method="adaptive", SNR=2)
# hr2msi <- peakAlign(hr2msi, method="diff")
# hr2msi <- reduceBaseline(hr2msi, method="median", blocks=50)
# hr2msi <- normalize(hr2msi, method="tic")
# hr2msi <- process(hr2msi)

